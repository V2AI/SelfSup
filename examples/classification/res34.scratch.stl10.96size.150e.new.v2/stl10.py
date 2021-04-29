import math
from copy import deepcopy
import torch
import torch.nn as nn

from torch.nn import functional as F

import cvpods
from cvpods.data.registry import DATASETS, PATH_ROUTES
from torchvision import datasets

import os.path as osp
import numpy as np


_PREDEFINED_SPLITS_STL10 = {
    "dataset_type": "STLDatasets",
    "evaluator_type": {"stl10": "classification"},
    "stl10": {
        "stl10_train": ("stl10", "train"),
        "stl10_unlabeled": ("stl10", "unlabeled"),
        "stl10_test": ("stl10", "test"),
    },
}
PATH_ROUTES.register(_PREDEFINED_SPLITS_STL10, "STL10")


@DATASETS.register()
class STLDatasets(datasets.STL10):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):

        self.meta = {"evaluator_type": "classification"}
        image_root, split = _PREDEFINED_SPLITS_STL10["stl10"][dataset_name]
        self.data_root = osp.join(osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
        self.image_root = osp.join(self.data_root, image_root)
        super(STLDatasets, self).__init__(self.image_root, split=split, download=True, transform=None)
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        self.transforms = transforms
        self.is_train = is_train

    def _apply_transforms(self, image, annotations=None):

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                img = deepcopy(image)
                annos = deepcopy(annotations)
                for tfm in tfms:
                    img, annos = tfm(img)
                dataset_dict[key] = (img, annos)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                image, annos = tfm(image)

            return image, annotations

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        dataset_dict = {"image_id": index, "category_id": annotations}

        image = image.convert("RGB")
        image = np.asarray(image)
        image = image[:, :, ::-1]
        images, anno = self._apply_transforms(image, annotations)

        def process(dd, img):

            if len(img.shape) == 3:
                image_shape = img.shape[:2]  # h, w
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif len(img.shape) == 4:
                image_shape = img.shape[1:3]
                # NHWC -> NCHW
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(0, 3, 1, 2)))

            return dd

        if isinstance(images, dict):
            ret = {}
            # multiple input pipelines
            for desc, item in images.items():
                img, anno = item
                ret[desc] = process(deepcopy(dataset_dict), img)
            return ret
        else:
            return process(dataset_dict, images)
