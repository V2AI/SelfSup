import torch
import torch.nn as nn

from cvpods.utils import comm

from cvpods.layers import ShapeSpec


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        self.backbone.linear = nn.Identity()

        self.lambd = cfg.MODEL.BT.LAMBD
        self.scale_loss = cfg.MODEL.BT.SCALE_LOSS

        # projector
        sizes = [2048] + list(map(int, cfg.MODEL.BT.PROJECTOR.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):

        cur_bs = len(batched_inputs)

        t_inputs = [bi["t"] for bi in batched_inputs]
        p_inputs = [bi["t_prime"] for bi in batched_inputs]

        y1 = self.preprocess_image([bi["image"][0] for bi in t_inputs])
        y2 = self.preprocess_image([bi["image"][0] for bi in p_inputs])

        z1 = self.projector(self.backbone(y1)["linear"])
        z2 = self.projector(self.backbone(y2)["linear"])

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(cur_bs * comm.get_world_size())
        torch.distributed.all_reduce(c)

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return dict(loss=loss)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        # images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [x.float().to(self.device) for x in batched_inputs]
        images = torch.stack([self.normalizer(x) for x in images])

        return images
