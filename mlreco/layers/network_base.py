import torch
from torch.nn import Module

# For MinkowskiEngine
import ROOT
ROOT.gSystem.Load("/usr/local/cuda/lib64/libcusparse.so")
import MinkowskiEngine as ME
from SparseTensor import SparseTensor
from MinkowskiNonlinearity import MinkowskiModuleBase

from mlreco.utils.misc import ResNetBlock

class NetworkBase(Module):
    '''
    Abstract Base Class for global network parameters.
    '''
    def __init__(self, model_cfg, name='base'):
        super(NetworkBase, self).__init__()
        cfg = self.model_cfg[name]
        self.D = cfg.get('D', 3)
        self.num_input = cfg.get('num_input', 1)
        self.allow_bias = cfg.get('allow_bias', True)
        self.leakiness = cfg.get('leakiness', 0.0)

    def forward(self, input):
        raise NotImplementedError
