from stl10 import *
from imagenet import Classification


def build_model(cfg):

    model = Classification(cfg)

    return model
