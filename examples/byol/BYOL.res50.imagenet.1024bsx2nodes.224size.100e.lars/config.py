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
            OUT_FEATURES=["res5"],
            STRIDE_IN_1X1=False,  # default true for msra models
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
        ),
        BYOL=dict(
            BASE_MOMENTUM=0.99,
            PROJ_DEPTH=2,
            PROJ_DIM=4096,
            OUT_DIM=256,
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
            MAX_EPOCH=100,
            WARMUP_ITERS=5,
        ),
        OPTIMIZER=dict(
            NAME="LARS_SGD",
            EPS=1e-8,
            TRUST_COEF=1e-3,
            CLIP=False,
            # _LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
            # _WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
            # _EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}
            BASE_LR=0.45 * 4,  # 0.3 for bs 256 => 4.8 for 4096
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-6,
            WD_EXCLUDE_BN_BIAS=True,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=1024,
        IMS_PER_DEVICE=128,  # 8 gpus per node
        BATCH_SUBDIVISIONS=1,  # Simulate Batch Size 4096
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                q=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([
                                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=1.0)),
                        ("RandomSolarization", dict(p=0.0)),
                        ("Torch_Compose", transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                        ])),
                    ], repeat_times=1)),
                ],
                k=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([
                                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=0.1)),
                        ("RandomSolarization", dict(p=0.2)),
                        ("Torch_Compose", transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                        ])),
                    ], repeat_times=1)),
                ],
            )
        )),
    TRAINER=dict(FP16=dict(ENABLED=False, OPTS=dict(OPT_LEVEL="O1"))),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
