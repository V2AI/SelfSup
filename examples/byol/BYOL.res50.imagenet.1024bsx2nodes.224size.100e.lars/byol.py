import math
import torch
import torch.nn as nn

from torch.nn import functional as F

from cvpods.layers import ShapeSpec


class EncoderWithProjection(nn.Module):
    def __init__(self, cfg):
        super(EncoderWithProjection, self).__init__()
        self.proj_dim = cfg.MODEL.BYOL.PROJ_DIM
        self.out_dim = cfg.MODEL.BYOL.OUT_DIM

        self.encoder = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(2048, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.out_dim, bias=False),
        )

    def forward(self, x):
        embedding = torch.flatten(self.avgpool(self.encoder(x)["res5"]), 1)
        return self.projector(embedding)


class BYOL(nn.Module):
    def __init__(self, cfg):
        super(BYOL, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.base_mom = cfg.MODEL.BYOL.BASE_MOMENTUM
        self.total_steps = cfg.SOLVER.LR_SCHEDULER.MAX_ITER * cfg.SOLVER.BATCH_SUBDIVISIONS

        self.online_network = EncoderWithProjection(cfg)
        self.target_network = EncoderWithProjection(cfg)

        self.size_divisibility = self.online_network.encoder.size_divisibility

        self.predictor = nn.Sequential(
            nn.Linear(self.online_network.out_dim, self.online_network.proj_dim),
            nn.BatchNorm1d(self.online_network.proj_dim),
            nn.ReLU(),
            nn.Linear(self.online_network.proj_dim, self.online_network.out_dim, bias=False),
        )

        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_parameter("step", nn.Parameter(torch.zeros(1), requires_grad=False))
        self.register_parameter("mom", nn.Parameter(torch.zeros(1), requires_grad=False))

        self.to(self.device)

    def losses(self, preds, targets):
        bz = preds.size(0)
        preds_norm = F.normalize(preds, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
        return loss

    def update_mom(self):
        mom = 1 - (1 - self.base_mom) * (math.cos(math.pi * self.step.item() / self.total_steps) + 1) / 2.
        self.step += 1
        return mom

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        mom = self.update_mom()
        self.mom[0] = mom
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * mom + param_q.data * (1. - mom)

    def forward(self, batched_inputs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q_inputs = [bi["q"] for bi in batched_inputs]
        k_inputs = [bi["k"] for bi in batched_inputs]

        x_i = torch.stack([bi["image"][0] for bi in q_inputs]).to(self.device)
        x_j = torch.stack([bi["image"][0] for bi in k_inputs]).to(self.device)

        online_out_1 = self.predictor(self.online_network(x_i))
        online_out_2 = self.predictor(self.online_network(x_j))

        with torch.no_grad():
            self._momentum_update_key_encoder()

            target_out_1 = self.target_network(x_i)
            target_out_2 = self.target_network(x_j)

        loss_i = self.losses(online_out_1, target_out_2)
        loss_j = self.losses(online_out_2, target_out_1)

        return {
            "loss_i": loss_i,
            "loss_j": loss_j,
            "mom": self.mom,
        }
