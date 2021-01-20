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
            TAU=0.07,
            MLP=False,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    DATALOADER=dict(NUM_WORKERS=6, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(120, 160),
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
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                        transforms.RandomHorizontalFlip(),
                    ]))
                ], repeat_times=2)),
            ],
        )
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/SelfSup',
        osp.split(osp.realpath(__file__))[0].split("SelfSup/")[-1]),
)


class MoCoConfig(BaseClassificationConfig):
    def __init__(self):
        super(MoCoConfig, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoConfig()
