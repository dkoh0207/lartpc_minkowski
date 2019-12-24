import torch
from torch.nn import Module

# For MinkowskiEngine
import ROOT
ROOT.gSystem.Load("/usr/local/cuda/lib64/libcusparse.so")
import MinkowskiEngine as ME

from mlreco.utils.misc import AtrousIIBlock
from mlreco.layers.network_base import NetworkBase


class ACNN(NetworkBase):
    '''
    <ACNN: a Full Resolution DCNN for Medical Image Segmentation>
    Original Paper: https://arxiv.org/pdf/1901.09203.pdf
    '''
    def __init__(self, model_cfg, name='acnn'):
        super(ACNN, self).__init__(model_cfg)
        cfg = model_cfg[name]
        # Depth of (RF - 1) / 8 is required for receptive field of "RF".
        self.depth = cfg.get('depth', 64)
        self.num_features = cfg.get('num_features', 4)

        net = []
        nIn = num_features
        for i in range(self.depth):
            nOut = num_features * (i // 2 + 1)
            net.append(AtrousIIBlock(nIn, nOut, self.D, self.leakiness))
            nIn = nOut
        self.net = nn.Sequential(*net)

    def forward(self, input):
        out = self.net(input)
        return out
