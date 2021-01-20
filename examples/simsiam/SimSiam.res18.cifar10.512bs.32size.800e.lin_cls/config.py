import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="../SimSiam.res18.cifar10.512bs.32size.800e/log/model_final.pkl",
        PIXEL_MEAN=[0.4465, 0.4822, 0.4914],  # BGR
        PIXEL_STD=[0.2010, 0.1994, 0.2023],
        BACKBONE=dict(FREEZE_AT=0, ),  # freeze all parameters manually in imagenet.py
        RESNETS=dict(
            DEPTH=18,
            RES2_OUT_CHANNELS=64,
            NUM_CLASSES=10,
            NORM="BN",
            OUT_FEATURES=["res5", "linear"],
            STRIDE_IN_1X1=False,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cifar10_train", ),
        TEST=("cifar10_test", ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=2,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            STEPS=(60, 80),
            MAX_EPOCH=90,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            LARC=dict(
                ENABLED=False,
                EPS=1e-8,
                TRUST_COEF=1e-3,
                CLIP=False,
            ),
            BASE_LR=30 / 256 * 256,
            MOMENTUM=0.9,
            WEIGHT_DECAY=0.0,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_Compose", transforms.Compose([
                    transforms.RandomResizedCrop(32),
                    transforms.RandomHorizontalFlip(),
                    ]))
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=10,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("SelfSup/")[-1]
    )
)


class ClassificationConfig(BaseClassificationConfig):
    def __init__(self):
        super(ClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ClassificationConfig()
