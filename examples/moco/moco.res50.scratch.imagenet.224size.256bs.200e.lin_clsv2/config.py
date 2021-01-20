import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="../simclr.moco_setting/log/model_final_pretrain_weight.pkl",
        BACKBONE=dict(FREEZE_AT=0, ),  # freeze all parameters manually in imagenet.py
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=1000,
            NORM="BN",
            OUT_FEATURES=["res5", "linear"],
            STRIDE_IN_1X1=False,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    DATALOADER=dict(
        NUM_WORKERS=6,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60, 80),
            MAX_EPOCH=90,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            BASE_LR=30,
            MOMENTUM=0.9,
            WEIGHT_DECAY=0.0,
            WEIGHT_DECAY_NORM=0.0,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RRC", transforms.RandomResizedCrop(224)),
                ("Torch_RHF", transforms.RandomHorizontalFlip()),
            ],
            TEST_PIPELINES=[
                ("Torch_R", transforms.Resize(256)),
                ("Torch_CC", transforms.CenterCrop(224)),
            ]
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
