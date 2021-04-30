import os.path as osp
import torchvision.transforms as transforms

from PIL import Image

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
        BT=dict(
            PROJECTOR="8192-8192-8192",
            LAMBD=3.9e-3,
            SCALE_LOSS=1 / 32,
            # LAMBD=0.0051,
            # SCALE_LOSS=0.024,
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
            MAX_EPOCH=300,
            WARMUP_ITERS=10,
        ),
        OPTIMIZER=dict(
            NAME="LARS_SGD",
            EPS=1e-8,
            TRUST_COEF=1e-3,
            CLIP=False,
            BASE_LR=0.2,  # 0.2 for bs 256 => 4.8 for 4096
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-6,
            EXCLUDE_BIAS_AND_BN=True,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,  # 8 gpus per node
        BATCH_SUBDIVISIONS=1,  # Simulate Batch Size 4096
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                t=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply([
                                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=1.0)),
                        ("RandomSolarization", dict(p=0.0)),
                    ], repeat_times=1)),
                ],
                t_prime=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply([
                                        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=0.1)),
                        ("RandomSolarization", dict(p=0.2)),
                    ], repeat_times=1)),
                ],
            )
        )),
    TRAINER=dict(FP16=dict(ENABLED=False, OPTS=dict(OPT_LEVEL="O1"))),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/self_supervised',
        osp.split(osp.realpath(__file__))[0].split("self_supervised/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
