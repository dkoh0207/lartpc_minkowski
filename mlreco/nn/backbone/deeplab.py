import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.layers.misc import *
from mlreco.nn.layers.network_base import NetworkBase
from .uresnext import UResNeXt

class DeepLabUNet(UResNeXt):

    def __init__(self, cfg, name='deeplab_unet'):
        super(DeepLabUNet, self).__init__(cfg, name=name)

    def forward(self, input):
        pass
