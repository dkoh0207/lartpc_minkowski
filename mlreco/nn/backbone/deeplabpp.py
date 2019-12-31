###########################################################
#   Modified version of DeepLabV3+, ideas aggregated from
#   DeepLabV3+, ResNext, ACNN, PSPNet, etc.
#   DeepLabV3+:
#   ResNeXt:
#   ACNN: https://arxiv.org/pdf/1901.09203.pdf
#   PSPNet:
###########################################################

import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.layers.misc import *
from mlreco.nn.layers.network_base import NetworkBase


class DeepLabPP(NetworkBase):

    def __init__(self, cfg, name='deeplabpp'):
        super(DeepLabPP, self).__init__(cfg)
        model_cfg = cfg['modules'][name]

        # Configurations
        self.reps = model_cfg.get('reps', 2)
        self.depth = model_cfg.get('depth', 5)
        self.num_filters = model_cfg.get('num_filters', 16)
        self.cardinality = model_cfg.get('cardinality', 4)
        assert (self.num_filters % self.cardinality == 0)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.input_kernel = model_cfg.get('input_kernel', 3)

        # Initialize Input Layer
        self.input_layer = AtrousIIBlock(num_input, num_filters,
                                         leakiness=self.leakiness)

    def encoder(self, x):
        pass

    def decoder(self, final, encoderTensors):
        pass

    def forward(self, input):
        pass
