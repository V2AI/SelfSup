import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(

    MODEL=dict(
        WEIGHTS="../SimSiam.res50.imagenet.256bs.224size.100e/log/model_epoch_0020.pkl",
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
            NAME="WarmupMultiStepLR",
            MAX_EPOCH=90,
            STEPS=(60, 80),
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            LARC=dict(
                ENABLED=True,
                EPS=1e-8,
                TRUST_COEF=1e-3,
                CLIP=False,
            ),
            BASE_LR=0.02 * 4096 / 256,
            MOMENTUM=0.9,
            WEIGHT_DECAY=0.0,
        ),
        CHECKPOINT_PERIOD=10,
        IMS_PER_BATCH=4096,
        IMS_PER_DEVICE=512,
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
