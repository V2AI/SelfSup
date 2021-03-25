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
        SWAV=dict(
            CANCEL_EPOCHS=1,  # cancel gradient for the first N epoch for prototypes
            NMB_CROPS=[2, 6],
            CROPS_FOR_ASSIGN=[0, 1],
            ARCH="resnet50",
            HIDDEN_MLP=2048,
            D=128,  # Feature Dim
            K=3840,  # Quele Length
            K_START=15,  # Epoch Queue Start
            P=3000,  # Prototypes
            TAU=0.1,
            EPS=0.05,
            SK_ITERS=3,
            NUMERICAL_STABILITY=True,
            NORM="BN1d"
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=200,
            WARMUP_ITERS=0,
            EPOCH_WISE=True,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            # EPS=1e-8,
            # TRUST_COEF=1e-3,
            # CLIP=False,
            BASE_LR=0.6,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-6,
        ),
        CHECKPOINT_PERIOD=5,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    DATALOADER=dict(NUM_WORKERS=6, ),
    TRAINER=dict(
        FP16=dict(ENABLED=False),
        NAME="SWAVRunner",
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                contrastive=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.14, 1.)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply([
                                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=0.5, mode="PIL")),
                        ("Torch_Compose", transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                        ])),
                    ], repeat_times=2)),
                ],
                multiview=[
                    ("RepeatList", dict(transforms=[
                        ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply([
                                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ])),
                        ("RandomGaussianBlur", dict(sigma=[.1, 2.], p=0.5, mode="PIL")),
                        ("Torch_Compose", transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                        ])),
                    ], repeat_times=6)),
                ],
                # linear=[
                #     ("Torch_Compose", transforms.Compose([
                #         transforms.RandomResizedCrop(224),
                #         transforms.RandomHorizontalFlip(),
                #     ])),
                # ],
            )
        )),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
