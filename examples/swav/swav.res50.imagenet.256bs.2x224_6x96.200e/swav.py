import numpy as np
import torch
import torch.nn as nn

from cvpods.utils import comm

from torch import distributed as dist

import swav_resnet as resnet_models


class SwAV(nn.Module):
    def __init__(self, cfg):
        super(SwAV, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.D = cfg.MODEL.SWAV.D
        self.K = cfg.MODEL.SWAV.K
        self.K_start = cfg.MODEL.SWAV.K_START
        self.P = cfg.MODEL.SWAV.P
        self.T = cfg.MODEL.SWAV.TAU
        self.EPS = cfg.MODEL.SWAV.EPS
        self.SK_ITERS = cfg.MODEL.SWAV.SK_ITERS

        self.improve_numerical_stability = cfg.MODEL.SWAV.NUMERICAL_STABILITY
        self.crops_for_assign = cfg.MODEL.SWAV.CROPS_FOR_ASSIGN
        self.nmb_crops = cfg.MODEL.SWAV.NMB_CROPS

        self.network = resnet_models.__dict__[cfg.MODEL.SWAV.ARCH](
            normalize=True,
            hidden_mlp=cfg.MODEL.SWAV.HIDDEN_MLP,
            output_dim=cfg.MODEL.SWAV.D,
            nmb_prototypes=cfg.MODEL.SWAV.P,
        )

        # create the queue
        self.register_buffer(
            "queue",
            torch.zeros(len(self.crops_for_assign), self.K // comm.get_world_size(), self.D)
        )
        self.use_the_queue = False

        # self.linear_eval = nn.Linear(encoder_dim, 1000)
        # self.loss_evaluator = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if self.epoch >= self.K_start:
            self.use_the_queue = True

        # normalize the prototypes
        with torch.no_grad():
            w = self.network.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.network.prototypes.weight.copy_(w)

        # # 0 Linear evaluation
        # linear_inputs = [bi['linear'] for bi in batched_inputs]
        # x_linear = self.preprocess_image([bi["image"] for bi in linear_inputs]).tensor
        # logits = self.linear_eval(
        #     torch.flatten(self.avgpool(self.network(x_linear)["res5"].detach()), 1)
        # )
        # labels = torch.tensor([gi["category_id"] for gi in linear_inputs]).cuda()
        # linear_loss = self.loss_evaluator(logits, labels)
        # acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        # 1. Preprocessing
        contrastive_inputs = torch.stack(
            [bi['contrastive']["image"] for bi in batched_inputs]
        ).permute(1, 0, 2, 3, 4).to(self.device)
        multiview_inputs = torch.stack(
            [bi['multiview']["image"] for bi in batched_inputs]
        ).permute(1, 0, 2, 3, 4).to(self.device)
        inputs = [ci.squeeze(0) for ci in torch.split(contrastive_inputs, 1)] + \
            [mi.squeeze(0) for mi in torch.split(multiview_inputs, 1)]

        embedding, output = self.network(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)]

                if self.use_the_queue:
                    out = torch.cat((
                        torch.mm(self.queue[i], self.network.prototypes.weight.t()),
                        out,
                    ))

                # fill the queue
                self.queue[i, bs:] = self.queue[i, :-bs].clone()
                self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) * bs]

                # get assignments
                q = out / self.EPS
                if self.improve_numerical_stability:
                    M = torch.max(q)
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q -= M
                q = torch.exp(q).t()
                q = distributed_sinkhorn(q, self.SK_ITERS)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                p = self.softmax(output[bs * v:bs * (v + 1)] / self.T)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)

        loss /= len(self.crops_for_assign)

        self.steps += 1

        return {
            "loss": loss,
            # "loss_linear": linear_loss,
            # "acc@1": acc1,
            # "acc@5": acc5,
        }


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (comm.get_world_size() * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            dist.all_reduce(u)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor
