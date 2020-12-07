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

### ImageNet

| Methods | Epochs | ImageNet Top 1 Acc |
| ------- | ------ | ------------------ |
| MoCo    | 200    | 60.5               |
| MoCov2  | 200    | 67.6               |
| SimCLR  | 200    | 63.2               |
| SimCLR* | 200    | 67.3               |
| SiMo    | 200    | 68.1               |
| BYOL    | 200e   | Comming Soon.      |
| SimSiam | 200e   | Comming Soon.      |
| SwAV    | 200e   | Comming Soon.      |

### 3D

| Methods       | Steps | Downstream task |
| ------------- | ----- | --------------- |
| PointContrast | -     | Comming Soon.   |



