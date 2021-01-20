import math
import torch
import torch.nn as nn

from torch.nn import functional as F

from cvpods.layers import ShapeSpec
from cvpods.structures import ImageList
from cvpods.layers.batch_norm import NaiveSyncBatchNorm1d


class SimSiam(nn.Module):
    def __init__(self, cfg):
        super(SimSiam, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.proj_dim = cfg.MODEL.BYOL.PROJ_DIM
        self.pred_dim = cfg.MODEL.BYOL.PRED_DIM
        self.out_dim = cfg.MODEL.BYOL.OUT_DIM

        self.total_steps = cfg.SOLVER.LR_SCHEDULER.MAX_ITER * cfg.SOLVER.BATCH_SUBDIVISIONS

        # create the encoders
        # num_classes is the output fc dimension
        cfg.MODEL.RESNETS.NUM_CLASSES = self.out_dim

        self.encoder_q = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        self.size_divisibility = self.encoder_q.size_divisibility

        dim_mlp = self.encoder_q.linear.weight.shape[1]

        # Projection Head
        self.encoder_q.linear = nn.Sequential(
            nn.Linear(dim_mlp, self.proj_dim),
            nn.SyncBatchNorm(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.SyncBatchNorm(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.SyncBatchNorm(self.proj_dim),
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.proj_dim, self.pred_dim),
            nn.SyncBatchNorm(self.pred_dim),
            nn.ReLU(),
            nn.Linear(self.pred_dim, self.out_dim),
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(1, 3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(1, 3, 1, 1)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std

        self.to(self.device)

    def D(self, p, z, version='simplified'): # negative cosine similarity
        if version == 'original':
            z = z.detach() # stop gradient
            p = F.normalize(p, dim=1) # l2-normalize 
            z = F.normalize(z, dim=1) # l2-normalize 
            return -(p*z).sum(dim=1).mean()
        elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
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
        x1 = self.preprocess_image([bi["image"][0] for bi in batched_inputs])
        x2 = self.preprocess_image([bi["image"][1] for bi in batched_inputs])

        z1, z2 = self.encoder_q(x1)["linear"], self.encoder_q(x2)["linear"]
        p1, p2 = self.predictor(z1), self.predictor(z2)

        loss = self.D(p1, z2) / 2 + self.D(p2, z1) / 2

        return dict(loss=loss) 

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = torch.stack([x for x in batched_inputs]).to(self.device)
        images = self.normalizer(images)

        return images
