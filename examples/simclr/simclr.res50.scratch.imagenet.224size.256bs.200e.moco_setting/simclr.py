# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from torch.nn import functional as F

from cvpods.layers import ShapeSpec
from cvpods.structures import ImageList
from cvpods.layers.batch_norm import NaiveSyncBatchNorm1d

from nt_xent2 import NT_Xent


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


class SimCLR(nn.Module):
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
        super(SimCLR, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.dim = cfg.MODEL.CLR.DIM
        self.T = cfg.MODEL.CLR.TAU
        self.mlp = cfg.MODEL.CLR.MLP
        self.norm = cfg.MODEL.CLR.NORM

        # create the encoders
        # num_classes is the output fc dimension
        cfg.MODEL.RESNETS.NUM_CLASSES = self.dim

        self.network = cfg.build_backbone(
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
            else:
                self.network.linear = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    nn.Linear(dim_mlp, self.dim),
                )
                nn.init.normal_(self.network.linear[0].weight, mean=0.0, std=0.01)  # linear weight
                nn.init.normal_(self.network.linear[2].weight, mean=0.0, std=0.01)  # linear weight

        # self.loss_evaluator = NTXentLoss(self.device, cfg.SOLVER.IMS_PER_DEVICE, self.T, True)
        self.loss_evaluator = NT_Xent(cfg.SOLVER.IMS_PER_DEVICE, self.T, self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        x_i = self.preprocess_image([bi["image"][0] for bi in batched_inputs]).tensor
        x_j = self.preprocess_image([bi["image"][1] for bi in batched_inputs]).tensor

        z_i = self.network(x_i)["linear"]
        z_j = self.network(x_j)["linear"]

        z_in = F.normalize(z_i, dim=1)
        z_jn = F.normalize(z_j, dim=1)

        loss_i, loss_j = self.loss_evaluator(z_in, z_jn)

        return {
            "loss_nt_xenti": loss_i,
            "loss_nt_xentj": loss_j,
        }

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [x.float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        return images
