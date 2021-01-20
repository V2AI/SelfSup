import numpy as np
import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        AS_PRETRAIN=True,  # Automatically convert ckpt to pretrain pkl
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=1000,
            STRIDE_IN_1X1=False,  # default true for msra models
            NORM="BN",
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
            OUT_FEATURES=["linear"],
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
            STEPS=(30, 60, 90),
            MAX_EPOCH=100,
            WARMUP_ITERS=10,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.1,
            WEIGHT_DECAY=0.0001,
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
                # ("Torch_CJ", transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)),
                # ("Lightning", dict(
                #     alpha_std=0.1,
                #     eig_val=np.array([[0.2175, 0.0188, 0.0045]]),
                #     eig_vec=np.array([
                #         [-0.5675, 0.7192, 0.4009],
                #         [-0.5808, -0.0045, -0.8140],
                #         [-0.5836, -0.6948, 0.4203]
                #     ]),
                # )),
            ],
            TEST_PIPELINES=[
                ("Torch_R", transforms.Resize(256)),
                ("Torch_CC", transforms.CenterCrop(224)),
            ]
        )
    ),
    # TEST=dict(
    #     EVAL_PERIOD=10,
    # ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class ClassificationConfig(BaseClassificationConfig):
    def __init__(self):
        super(ClassificationConfig, self).__init__()
        self._register_configuration(_config_dict)


config = ClassificationConfig()
