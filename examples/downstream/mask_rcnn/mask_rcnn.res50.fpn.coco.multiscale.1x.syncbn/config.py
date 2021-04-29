import os.path as osp

from cvpods.configs.rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # RGB
        PIXEL_STD=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        WEIGHTS="/path/to/your/pre_trained_weights.pkl",
        MASK_ON=True,
        BACKBONE=dict(
            FREEZE_AT=0,
        ),
        RESNETS=dict(
            DEPTH=50,
            NORM="nnSyncBN",
            STRIDE_IN_1X1=False,  # True only for msra weights
        ),
        FPN=dict(
            NORM="nnSyncBN",
        ),
        ROI_BOX_HEAD=dict(
            NORM="nnSyncBN",
            NUM_CONV=4,
            NUM_FC=1,
        ),
        ROI_MASK_HEAD=dict(
            NORM="nnSyncBN",
        ),
    ),
    DATASETS=dict(
        TRAIN=("coco_2017_train",),
        TEST=("coco_2017_val",),
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(60000, 80000),
            MAX_ITER=90000,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.02,
        ),
        IMS_PER_BATCH=16,
    ),
    INPUT=dict(
        FORMAT="RGB",
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(640, 672, 704, 736, 768, 800),
                      max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
    ),
    TEST=dict(
        EVAL_PEROID=10000,
        PRECISE_BN=dict(
            ENABLED=True,
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/data/Outputs/model_logs/cvpods_playground',
        osp.split(osp.realpath(__file__))[0].split("playground/")[-1]),
)


class MaskRCNNConfig(RCNNFPNConfig):
    def __init__(self):
        super(MaskRCNNConfig, self).__init__()
        self._register_configuration(_config_dict)


config = MaskRCNNConfig()
