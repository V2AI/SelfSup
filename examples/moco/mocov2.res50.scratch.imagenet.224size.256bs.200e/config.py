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
            NORM="BN",
            OUT_FEATURES=["linear"],
            STRIDE_IN_1X1=False,  # default true for msra models
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
        ),
        MOCO=dict(
            DIM=128,
            K=65536,
            MOMENTUM=0.999,
            TAU=0.2,
            MLP=True,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    DATALOADER=dict(NUM_WORKERS=6, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=200,
            WARMUP_ITERS=5,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.03,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RepeatList", dict(transforms=[
                    ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                            transforms.RandomApply([
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomHorizontalFlip(),
                        ])),
                    ("GaussianBlur", dict(sigma=[.1, 2.], p=0.5)),
                ], repeat_times=2)),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("SelfSup/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
