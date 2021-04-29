import numpy as np
import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        PIXEL_MEAN=[0.4465, 0.4822, 0.4914],  # BGR
        PIXEL_STD=[0.2010, 0.1994, 0.2023],
        AS_PRETRAIN=True,  # Automatically convert ckpt to pretrain pkl
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=10,
            STRIDE_IN_1X1=False,  # default true for msra models
            NORM="BN",
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
            OUT_FEATURES=["linear"],
        ),
    ),
    DATASETS=dict(
        TRAIN=("cifar10_train", ),
        TEST=("cifar10_test", ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=6,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            # NAME="CosineAnnealingLR",
            NAME="WarmupCosineLR",
            MAX_EPOCH=200,
            WARMUP_ITERS=0,
            EPOCH_WISE=True,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.1,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
        ),
        CHECKPOINT_PERIOD=50,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RRC", transforms.RandomCrop(32, padding=4)),
                ("Torch_RHF", transforms.RandomHorizontalFlip()),
            ],
            TEST_PIPELINES=[
            ]
        )
    ),
    TEST=dict(
        EVAL_PERIOD=50,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class ClassificationConfig(BaseClassificationConfig):
    def __init__(self):
        super(ClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ClassificationConfig()
