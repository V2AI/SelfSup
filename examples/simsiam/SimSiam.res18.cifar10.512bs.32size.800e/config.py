import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        AS_PRETRAIN=True,
        PIXEL_MEAN=[0.4914, 0.4822, 0.4465],  # RGB
        PIXEL_STD=[0.2023, 0.1994, 0.2010],
        RESNETS=dict(
            DEPTH=18,
            RES2_OUT_CHANNELS=64,
            NUM_CLASSES=10,
            NORM="nnSyncBN",
            OUT_FEATURES=["linear"],
            STRIDE_IN_1X1=False,  # default true for msra models
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
        ),
        BYOL=dict(
            PROJ_DIM=2048,
            PRED_DIM=512,
            OUT_DIM=2048,
        ),
    ),
    DATASETS=dict(
        TRAIN=("cifar10_train", ),
        TEST=("cifar10_test", ),
    ),
    DATALOADER=dict(NUM_WORKERS=4, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=800,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.03 / 256 * 512,
            MOMENTUM=0.9,
            WEIGHT_DECAY=5e-4,
        ),
        CHECKPOINT_PERIOD=50,
        IMS_PER_BATCH=512,
        IMS_PER_DEVICE=64,
    ),
    INPUT=dict(
        FORMAT="RGB",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("RepeatList", dict(transforms=[
                    ("Torch_Compose", transforms.Compose([
                            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                        ]))
                ], repeat_times=2)),
            ]
        )),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/self_supervised',
        osp.split(osp.realpath(__file__))[0].split("self_supervised/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
