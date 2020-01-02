import torch
import torch.nn as nn

def segmentation_loss_dict():
    from . import segmentation
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(reduction='none'),
        'lovasz_softmax': segmentation.LovaszSoftmaxLoss(),
        'focal': segmentation.FocalLoss(reduce=False),
        'weighted_cross_entropy': segmentation.WeightedFocalLoss(reduce=False),
    }
    return losses

def segmentation_loss_construct():
    losses = segmentation_loss_dict()
    if name not in losses:
        raise Exception("Unknown loss function name provided")
    return losses[name]