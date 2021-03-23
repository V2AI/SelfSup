import torch
import torch.nn as nn

from torch.nn import functional as F

from cvpods.layers import ShapeSpec


class SimSiam(nn.Module):
    def __init__(self, cfg):
        super(SimSiam, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.proj_dim = cfg.MODEL.BYOL.PROJ_DIM
        self.pred_dim = cfg.MODEL.BYOL.PRED_DIM
        self.out_dim = cfg.MODEL.BYOL.OUT_DIM

        self.encoder_q = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(self.out_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, self.pred_dim),
            nn.BatchNorm1d(self.pred_dim),
            nn.ReLU(),
            nn.Linear(self.pred_dim, self.out_dim),
        )

        self.to(self.device)

    def D(self, p, z, version='simplified'):    # negative cosine similarity
        if version == 'original':
            z = z.detach()  # stop gradient
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()
        elif version == 'simplified':  # same thing, much faster.
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

    def forward(self, batched_inputs):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        x1 = torch.stack([bi["image"][0] for bi in batched_inputs]).to(self.device)
        x2 = torch.stack([bi["image"][1] for bi in batched_inputs]).to(self.device)
        z1 = self.projector(torch.flatten(self.avgpool(self.encoder_q(x1)["res5"]), 1))
        z2 = self.projector(torch.flatten(self.avgpool(self.encoder_q(x2)["res5"]), 1))
        p1, p2 = self.predictor(z1), self.predictor(z2)

        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2

        return dict(loss=loss)
