# SelfSup

Collections of self-supervised methods (MoCo series, SimCLR, **SiMo**, BYOL, SimSiam, SwAV, PointContrast, etc.). 


## Get Started

### Install cvpods following the instructions.

Install cvpods from https://github.com/Megvii-BaseDetection/cvpods.git .

### Prepare Datasets

```shell
cd cvpods
ln -s /path/to/your/ImageNet datasets/imagenet
```

### Train your own models

```
cd /path/to/your/SelfSup/examples/simclr/simclr.res50.scratch.imagenet.224size.256bs.200e
# pre-train
pods_train --num-gpus 8
# convert to weights
python convert.py simclr.res50.scratch.imagenet.224size.256bs.200e/log/model_final.pth weights.pkl
# downstream evaluation
cd /path/to/your/simclr.res50.scratch.imagenet.224size.256bs.200e.lin_cls
pods_train --num-gpus 8 MODEL.WEIGHTS /path/to/your/weights.pkl

```

## Model Zoo

### Supervised Models 

| Methods | Epochs | ImageNet Top 1 |
| ------- | ------ | ------------------ |
| Res50   | 100    | 76.4               |

### Unsupervised Models 

| Methods | Epochs | Batch Size | ImageNet Top 1 |  CIFAR10 Top 1 |
| ------- | ------ | ---------- | ------------------ | -------------- |
| MoCo    | 200    |     256    | 60.5 (paper: 60.5) |      -         |
| MoCov2  | 200    |     256    | 67.6 (paper: 67.5) |      -         |
| SimCLR  | 200    |     256    | 63.2 (paper: 61.9) |      -         |
| **SimCLR*** | 200    |     256    | 67.3 (**Ours**)|      -         |
| **SiMo**    | 200    |     256    | 68.1 (**Ours**)|      -         |
| SimSiam | 100    |     256    | 67.6 (paper: 67.7) |     800e: 90.7 (paper: 91.8) |
| SwAV    | 200    |     256    | Comming Soon.      |    -          |
| BYOL    | 200    |     256    | Comming Soon.      |      -         |
| BarlowTwins| 300    |     256    | Comming Soon.      |      -         |
| SCRL    | 200    |     256    | Comming Soon.      |      -         |

### 3D Unsupervised Models 

| Methods       | Steps | Downstream task |
| ------------- | ----- | --------------- |
| PointContrast | -     | Comming Soon.   |


## Citation

SelfSup is a part of [cvpods](https://github.com/Megvii-BaseDetection/cvpods), so if you find this repo useful in your research, or if you want to refer the implementations in this repo, please consider cite:

```BibTeX

@article{zhu2020eqco,
  title={EqCo: Equivalent Rules for Self-supervised Contrastive Learning},
  author={Zhu, Benjin and Huang, Junqiang and Li, Zeming and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2010.01929},
  year={2020}
}

@misc{zhu2020cvpods,
  title={cvpods: All-in-one Toolbox for Computer Vision Research},
  author={Zhu*, Benjin and Wang*, Feng and Wang, Jianfeng and Yang, Siwei and Chen, Jianhu and Li, Zeming},
  year={2020}
}
```
