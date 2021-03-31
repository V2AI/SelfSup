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

### Supervised Classification 

#### ImageNet
| Methods | Training Schedule | Top 1  Acc |
| ------- | ------ | ------------------ |
| Res50   | 100e    | 76.4               |

#### CIFAR 10
| Methods | Training Schedule | Top 1  Acc |
| ------- | ------ | ------------------ |
| Res50   | 200e    | 95.4              |

#### STL 10
| Methods | Training Schedule | Top 1  Acc |
| ------- | ------ | ------------------ |
| Res50   | 150e    | 86.1              |


### Self-Supervised Learning - Classification

| Methods | Training Schedule | Batch Size | ImageNet Top 1 |
| ------- | ------ | ---------- | ------------------ |
| MoCo    | 200e    |     256    | 60.5 (paper: 60.5) | 
| MoCov2  | 200e  |     256    | 67.6 (paper: 67.5) | 
| SimCLR  | 200e    |     256    | 63.2 (paper: 61.9) |
| **SimCLR*** | 200e    |     256    | 67.3 (**Ours**)|
| **SiMo**    | 200e    |     256    | 68.1 (**Ours**)|
| SimSiam | 100e    |     256    | 67.6 (paper: 67.7) |
| SwAV    | 200e    |     256    | 73.0 (paper 72.7)  |
| BYOL    | 200e    |     256    | Comming Soon.      |
| BarlowTwins| 300e |     256    | Comming Soon.      |

### Self-Supervised Learning - Detection (2D)

| Methods | Training Schedule | Batch Size | ImageNet Top 1 |
| SCRL    | 200    |     256    | Comming Soon.      | 
| DetCon    | 200    |     256    | Comming Soon.      |

### Self-Supervised Learning - 3D Scene Understanding

| Methods       | Training Schedule | Downstream task |
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
