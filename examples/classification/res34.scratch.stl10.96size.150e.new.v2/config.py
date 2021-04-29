import numpy as np
import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        PIXEL_MEAN=[x/255.0 for x in [113.9, 123.0, 125.3]],  # BGR
        PIXEL_STD=[x/255.0 for x in [66.7, 62.1, 63.0]],
        AS_PRETRAIN=True,  # Automatically convert ckpt to pretrain pkl
        RESNETS=dict(
            DEPTH=34,
            RES2_OUT_CHANNELS=64,
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
            NAME="WarmupMultiStepLR",
            STEPS=(100, 125),
            MAX_EPOCH=150,
            WARMUP_ITERS=0,
            # EPOCH_WISE=False,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.1,
            MOMENTUM=0.9,
            WEIGHT_DECAY=5e-4,
            WEIGHT_DECAY_NORM=5e-4,
            NESTEROV=True,
        ),
        CHECKPOINT_PERIOD=50,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=16,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RRC", transforms.RandomCrop(96, padding=4)),
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
