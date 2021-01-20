import os.path as osp
import torchvision.transforms as transforms

from cvpods.configs.base_classification_config import BaseClassificationConfig
from torchvision import transforms
from transforms import GaussianBlur, Solarization

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        AS_PRETRAIN=True,
        RESNETS=dict(
            DEPTH=50,
            NUM_CLASSES=1000,
            NORM="SyncBN",
            OUT_FEATURES=["res5"],
            STRIDE_IN_1X1=False,  # default true for msra models
            ZERO_INIT_RESIDUAL=True,  # default false, use true for all subsequent models
        ),
        M2T=dict(
            PARAM_MOMENTUM=0.99,
        ),
    ),
    DATASETS=dict(
        TRAIN=("imagenet_train", ),
        TEST=("imagenet_val", ),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="WarmupCosineLR",
            MAX_EPOCH=100,
            WARMUP_ITERS=10,
            WARMUP_METHOD="linear",
            WARMUP_FACTOR=1e-6/0.05,
            EPOCH_WISE=False,
        ),
        OPTIMIZER=dict(
            NAME="SGD",
            BASE_LR=0.05,
            MOMENTUM=0.9,
            WEIGHT_DECAY=1e-4,
        ),
        CHECKPOINT_PERIOD=5,
        IMS_PER_BATCH=256,
        IMS_PER_DEVICE=32,
    ),
    DATALOADER=dict(NUM_WORKERS=8, ),
    TRAINER=dict(FP16=dict(ENABLED=False),),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=dict(
                q=[
                    ("Torch_Compose", transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])),
                ],
                k=[
                    ("Torch_Compose", transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([Solarization()], p=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]))
                ],
            )
        )),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground/SelfSup',
        osp.split(osp.realpath(__file__))[0].split("SelfSup/")[-1]))


class MoCoV2Config(BaseClassificationConfig):
    def __init__(self):
        super(MoCoV2Config, self).__init__()
        self._register_configuration(_config_dict)


config = MoCoV2Config()
