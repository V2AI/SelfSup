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
            STRIDE_IN_1X1=False,       # default true for msra models
            ZERO_INIT_RESIDUAL=True,   # default false, use true for all subsequent models
        ),
        BYOL=dict(
            PROJ_DIM=2048,
            PRED_DIM=512,
            OUT_DIM=2048,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    DATALOADER=dict(NUM_WORKERS=4, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=100,
            WARMUP_ITERS=0,
            EPOCH_WISE=True,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.05,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RepeatList", dict(transforms=[
                    ("Torch_Compose", transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
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
            ]
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/self_supervised',
        osp.split(osp.realpath(__file__))[0].split("self_supervised/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
