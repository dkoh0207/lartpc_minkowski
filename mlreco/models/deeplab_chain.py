import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.backbone.deeplab import DeepLabUNet
from collections import defaultdict


class DeepLab_Chain(nn.Module):

    def __init__(self, cfg, name='deeplab_chain'):
        super(DeepLab_Chain, self).__init__()
        self.model_cfg = cfg
        self.net = DeepLabUNet(cfg, name='deeplab_unet')
        self.F = self.model_cfg.get('num_filters', 16)
        self.num_classes = self.model_cfg.get('num_classes', 5)
        self.segmentation = ME.MinkowskiLinear(self.F * 2, self.num_classes)

    def forward(self, input):
        out = defaultdict(list)
        num_gpus = len(input)
        for igpu, x in enumerate(input):
            res = self.net(x)
            seg = res['finalFeatures']
            seg = self.segmentation(seg)
            out['segmentation'].append(seg.F)
        return out
