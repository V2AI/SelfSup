# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch import distributed as dist

from cvpods.layers import ShapeSpec
from cvpods.structures import ImageList
from cvpods.layers.batch_norm import NaiveSyncBatchNorm1d
from cvpods.utils import comm


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SiMo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, cfg):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SiMo, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.dim = cfg.MODEL.CLR.DIM
        self.T = cfg.MODEL.CLR.TAU
        self.mlp = cfg.MODEL.CLR.MLP
        self.norm = cfg.MODEL.CLR.NORM
        self.m = cfg.MODEL.CLR.MOMENTUM

        alpha = cfg.MODEL.CLR.ALPHA
        K = cfg.MODEL.CLR.K

        # create the encoders
        # num_classes is the output fc dimension
        cfg.MODEL.RESNETS.NUM_CLASSES = self.dim

        self.network = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self.network_k = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        self.size_divisibility = self.network.size_divisibility

        if self.mlp:  # hack: brute-force replacement
            dim_mlp = self.network.linear.weight.shape[1]
            if self.norm == "SyncBN":
                self.network.linear = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp, bias=False),
                    NaiveSyncBatchNorm1d(dim_mlp),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, self.dim, bias=False),
                    NaiveSyncBatchNorm1d(self.dim)
                )
                nn.init.normal_(self.network.linear[0].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.normal_(self.network.linear[3].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.constant_(self.network.linear[1].weight, 1.0)  # bn gamma
                nn.init.constant_(self.network.linear[4].weight, 1.0)  # bn gamma

                self.network_k.linear = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp, bias=False),
                    NaiveSyncBatchNorm1d(dim_mlp),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, self.dim, bias=False),
                    NaiveSyncBatchNorm1d(self.dim)
                )
                nn.init.normal_(self.network_k.linear[0].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.normal_(self.network_k.linear[3].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.constant_(self.network_k.linear[1].weight, 1.0)  # bn gamma
                nn.init.constant_(self.network_k.linear[4].weight, 1.0)  # bn gamma

            else:
                self.network.linear = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, self.dim),
                )
                nn.init.normal_(self.network.linear[0].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.normal_(self.network.linear[2].weight, mean=0.0, std=0.01)  # linear weight

                self.network_k.linear = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, self.dim),
                )
                nn.init.normal_(self.network_k.linear[0].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.normal_(self.network_k.linear[2].weight, mean=0.0, std=0.01)  # linear weight

        for param_q, param_k in zip(self.network.parameters(), self.network_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.loss_evaluator = NT_Xent(cfg.SOLVER.IMS_PER_DEVICE, self.T, alpha, K, self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1, 3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1, 3, 1, 1)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std

        self.to(self.device)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.network.parameters(), self.network_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, batched_inputs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        x_i = self.preprocess_image([bi["image"][0] for bi in batched_inputs])
        x_j = self.preprocess_image([bi["image"][1] for bi in batched_inputs])

        z_i = self.network(x_i)["linear"]
        z_in = F.normalize(z_i, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_j = self.network_k(x_j)["linear"]
            z_jn = F.normalize(z_j, dim=1)

        loss_i, acc1, acc5 = self.loss_evaluator(z_in, z_jn)

        return {
            "loss_NTXent_i": loss_i,
            "Acc@1": acc1,
            "Acc@5": acc5,
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = torch.stack([x for x in batched_inputs]).to(self.device)
        images = self.normalizer(images)

        return images


class NT_Xent(nn.Module):
    def __init__(self, device_size, temperature, alpha, K, device):
        super(NT_Xent, self).__init__()
        self.device_size = device_size
        self.temperature = temperature
        self.alpha = alpha
        self.K = K
        self.device = device

        self.similarity_f = nn.CosineSimilarity(dim=2)

        pos_mask_i, neg_mask_i = \
            self.mask_correlated_samples(comm.get_world_size(), self.device_size)
        self.pos_mask_i = pos_mask_i.to(self.device)
        self.neg_mask_i = neg_mask_i.to(self.device)

    def mask_correlated_samples(self, world_size, device_size):
        batch_size = world_size * device_size

        neg_mask_i = torch.ones((batch_size, batch_size), dtype=bool)
        for rank in range(world_size):
            for idx in range(device_size):
                neg_mask_i[device_size * rank + idx, device_size * rank + idx] = 0  # i
        pos_mask_i = neg_mask_i.clone()

        return ~pos_mask_i, neg_mask_i

    def forward(self, z_i, z_j):
        device_size = z_i.shape[0]
        batch_size = device_size * comm.get_world_size()
        local_rank = comm.get_rank()

        neg_perm = torch.randperm(batch_size - 1)[: self.K]

        if comm.get_world_size() > 1:
            group = comm._get_global_gloo_group()

            zi_large = [torch.zeros_like(z_i) for _ in range(comm.get_world_size())]
            zj_large = [torch.zeros_like(z_j) for _ in range(comm.get_world_size())]

            dist.all_gather(zi_large, z_i, group=group)
            dist.all_gather(zj_large, z_j, group=group)

            choices = [
                torch.zeros_like(neg_perm, dtype=torch.int64) for _ in range(comm.get_world_size())]
            dist.all_gather(choices, neg_perm, group=group)
            neg_perm = choices[0]

        else:
            zi_large = [z_i]
            zj_large = [z_j]

        zi_large[local_rank] = z_i

        zi_large = torch.cat(zi_large)
        zj_large = torch.cat(zj_large)

        sim_i_large = self.similarity_f(
            zi_large.unsqueeze(1), zj_large.unsqueeze(0)) / self.temperature

        positive_samples_i = sim_i_large[self.pos_mask_i].reshape(batch_size, 1)
        negative_samples_i = sim_i_large[self.neg_mask_i].reshape(batch_size, -1)[:, neg_perm]

        labels_i = torch.zeros(batch_size).to(self.device).long()
        logits_i = torch.cat((positive_samples_i, negative_samples_i), dim=1)

        # EqCo
        loss_i = torch.log(
            torch.exp(positive_samples_i) +
            # self.alpha / negative_samples_i.shape[1] *  # uncomment this when negatives != bs
            torch.exp(negative_samples_i).sum(dim=-1, keepdim=True)
        ) - positive_samples_i
        loss_i = loss_i.sum() / device_size

        acc1, acc5 = accuracy(logits_i, labels_i, topk=(1, 5))

        return loss_i, acc1, acc5
