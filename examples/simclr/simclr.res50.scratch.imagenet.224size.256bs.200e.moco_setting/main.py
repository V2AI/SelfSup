import matplotlib
matplotlib.use('Agg')

import torch
import cv2
import numpy as np
import pdb
import pickle

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE

from cvpods.data.build import build_dataset, build_transform_gen

from config import config as cfg
from net import build_model


def visualize_data(per_image, meta, save_to_file=False):
    # Pytorch tensor is in (C, H, W) format
    img = per_image["image"].permute(1, 2, 0)

    cv2.imwrite(str(per_image["image_id"]) + ".jpg", img.cpu().numpy())

    if cfg.INPUT.FORMAT == "BGR":
        img = img[:, :, [2, 1, 0]]
    else:
        img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))


train_datasets = cfg.DATASETS.TRAIN
test_datasets = cfg.DATASETS.TEST

train_gens = cfg.INPUT.AUG.TRAIN_PIPELINES
test_gens = cfg.INPUT.AUG.TEST_PIPELINES

train_transforms = build_transform_gen(train_gens)
test_transforms = build_transform_gen(test_gens)

trainset = build_dataset(cfg, train_datasets, train_transforms)
testset = build_dataset(cfg, test_datasets, test_transforms)

toyset = build_dataset(cfg, train_datasets, test_transforms)

model = build_model(cfg)
ckpt = torch.load("./model_final.pth")
ret = model.load_state_dict(ckpt['model'], strict=True)
model.training = False

if not Path("image_features_res5.pkl").is_file():
    for item in tqdm(toyset):
        item["image"] = item["image"].cuda()
        pred = model([item])
    features = []
    instance_nums = {}
    for key, items in model.features.items():
        instance_nums[key] = len(items)
        features.extend(items)
        model.features[key] = np.stack(items, axis=0).tolist()
    with open("image_features_res5.pkl", "wb") as fout:
        pickle.dump(model.features, fout, protocol=pickle.HIGHEST_PROTOCOL)
    loaded_features = model.features
else:
    with open("image_features_res5.pkl", "rb") as fin:
        loaded_features = pickle.load(fin)
    features = []
    instance_nums = {}
    for key, items in loaded_features.items():
        instance_nums[key] = len(items)
        features.extend(items)

selected_features = []
labels = []
selected_keys = np.random.choice(list(instance_nums.keys()), 5, replace=False)
for k in selected_keys:
    labels.extend(np.repeat([k], len(loaded_features[k])))
    selected_features.extend(loaded_features[k])
labels = np.stack(labels)
features = np.stack(selected_features, axis=0)
no_of_images = features.shape[0]

tsne = TSNE(
    # perplexity=50,
    # init="pca",
    # learning_rate=20,
    # n_iter=3000,
    verbose=True)
reduced = tsne.fit_transform(features)
reduced_transformed = reduced - np.min(reduced, axis=0)
reduced_transformed /= np.max(reduced_transformed, axis=0)

# initialize a matplotlib plot
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

# for every class, we'll add a scatter plot separately
for idx, label in enumerate(selected_keys):
    # extract the coordinates of the points of this class only
    indices = [i for i, l in enumerate(labels) if l == label]

    current_tx = np.take(selected_features[0], indices)
    current_ty = np.take(selected_features[1], indices)

    # convert the class color to matplotlib format
    color = np.random.uniform(low=0.0, high=1.0, size=3)
    # color = np.array([(idx+1) * 10, (idx+1) * 24, 255 - (idx+1) * 24], dtype=np.float) / 255

    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)

# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
# plt.show()
plt.savefig('debug.png')
