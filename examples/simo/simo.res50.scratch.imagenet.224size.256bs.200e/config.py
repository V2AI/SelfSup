import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        AS_PRETRAIN=True,
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=1000,
            NORM="nnSyncBN",
            OUT_FEATURES=["linear"],
            STRIDE_IN_1X1=False,  # default true for msra models
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
        ),
        CLR=dict(
            ALPHA=256,
            K=256,
            DIM=128,
            TAU=0.2,
            MLP=True,
            NORM="nnSyncBN",
            MOMENTUM=0.999,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_nori_train", ),
        TEST=("imagenet_nori_val", ),
    ),
    DATALOADER=dict(NUM_WORKERS=6, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=200,
            WARMUP_ITERS=10,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            LARS=dict(
                ENABLED=False,
                EPS=1e-8,
                TRUST_COEF=1e-3,
            ),
            BASE_LR=0.03,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RepeatList", dict(transforms=[
                    ("Torch_RRC", transforms.RandomResizedCrop(224, scale=(0.2, 1.))),
                    ("Torch_RACJ", transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)),
                    ("GaussianBlur", dict(sigma=[.1, 2.], p=0.5)),
                    ("Torch_RG", transforms.RandomGrayscale(p=0.2)),
                    ("Torch_RHF", transforms.RandomHorizontalFlip()),
                ], repeat_times=2)),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
