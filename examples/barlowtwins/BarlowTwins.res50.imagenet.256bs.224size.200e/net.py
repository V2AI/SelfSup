from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone import build_resnet_backbone

from cvpods.utils import comm

from barlow_twins import BarlowTwins


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_model(cfg):

    cfg.build_backbone = build_backbone

    model = BarlowTwins(cfg)
    if comm.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
