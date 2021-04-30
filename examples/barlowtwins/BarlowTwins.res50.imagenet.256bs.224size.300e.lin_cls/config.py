import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(

    MODEL=dict(
        WEIGHTS="../BarlowTwins.res50.imagenet.256bs.224size.200e/log/model_final.pkl",
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
        NUM_WORKERS=4,
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=100,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.3,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-6,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    INPUT=dict(
        FORMAT="RGB",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_Compose", transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ])),
            ],
            TEST_PIPELINES=[
                ("Torch_Compose", transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                ]))
            ],
        )
    ),
    TEST=dict(
        EVAL_PERIOD=10,
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/self_supervised',
        osp.split(osp.realpath(__file__))[0].split("self_supervised/")[-1]
    )
)


class ClassificationConfig(BaseClassificationConfig):
    def __init__(self):
        super(ClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ClassificationConfig()
