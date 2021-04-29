import numpy as np
import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        PIXEL_MEAN=[0.5, 0.5, 0.5],  # BGR
        PIXEL_STD=[0.5, 0.5, 0.5],
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
        TRAIN=("stl10_train", ),
        TEST=("stl10_test", ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=2,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            # NAME="CosineAnnealingLR",
            NAME="WarmupMultiStepLR",
            STEPS=(80, 120),
            MAX_EPOCH=150,
            WARMUP_ITERS=0,
            EPOCH_WISE=False,
        ),
        OPTIMIZER=dict(
            NAME="Adam",
            BASE_LR=0.001,
            WEIGHT_DECAY=0.0,
        ),
        CHECKPOINT_PERIOD=50,
        IMS_PER_BATCH=1600,
        IMS_PER_DEVICE=200,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_P", transforms.Pad(4)),
                ("Torch_RRC", transforms.RandomCrop(96)),
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
