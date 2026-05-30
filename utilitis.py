"""
    setup model and datasets
"""


import copy

import numpy as np
import torch
from dataset import *

# from models import *

from ResNet import (resnet18, resnet34, resnet50, resnet101, resnet152,
                    NormalizeByChannelMeanStd)
from ResNets import resnet20s, resnet32s, resnet44s, resnet56s, resnet110s, resnet1202s
from VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from VGG_LTH import vgg16_bn_lth



# from advertorch.utils import NormalizeByChannelMeanStd

model_dict = {
    # ResNet variants
    'resnet18'    : resnet18,
    'resnet34'    : resnet34,
    'resnet50'    : resnet50,
    'resnet101'   : resnet101,
    'resnet152'   : resnet152,
    # ResNets small variants
    'resnet20s'   : resnet20s,
    'resnet32s'   : resnet32s,
    'resnet44s'   : resnet44s,
    'resnet56s'   : resnet56s,
    'resnet110s'  : resnet110s,
    'resnet1202s' : resnet1202s,
    # VGG variants
    'vgg11'       : vgg11,
    'vgg11_bn'    : vgg11_bn,
    'vgg13'       : vgg13,
    'vgg13_bn'    : vgg13_bn,
    'vgg16'       : vgg16,
    'vgg16_bn'    : vgg16_bn,
    'vgg19'       : vgg19,
    'vgg19_bn'    : vgg19_bn,
    # VGG LTH variant
    'vgg16_bn_lth': vgg16_bn_lth,
}


__all__ = ["setup_model_dataset"]


def setup_model_dataset(args):

    # NOTE : Added the below line of code

    only_mark = getattr(args, 'only_mark', False)

    if args.dataset == "cifar10":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        train_set_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    elif args.dataset == "cifar100":
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
        )
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
        )

    # else:
    #     raise ValueError("Dataset not supprot yet !")


    # Added this below lines of code in place of above else code 
    
    elif args.dataset == "dermamnist":
        classes = 7
        normalization = NormalizeByChannelMeanStd(
            mean=[0.7631, 0.5380, 0.5614],
            std= [0.1366, 0.1543, 0.1692]
        )
        train_set_loader, val_loader, test_loader = dermamnist_dataloaders(
            batch_size=args.batch_size,
            data_dir=args.data,
            num_workers=args.workers,
            class_to_replace=args.class_to_replace
                if hasattr(args, "class_to_replace") else None,
            num_indexes_to_replace=args.num_indexes_to_replace
                if hasattr(args, "num_indexes_to_replace") else None,
            sample_forget_type=args.sample_forget_type
                if hasattr(args, "sample_forget_type") else "random",
            seed=args.seed,
            # only_mark=True,
            only_mark=only_mark,
        )
    else:
        raise ValueError("Dataset not supported yet!")

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader



    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    return model, train_set_loader, val_loader, test_loader


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)
