import torch
from torch.nn import Module

# For MinkowskiEngine
import MinkowskiEngine as ME
from SparseTensor import SparseTensor
from MinkowskiNonlinearity import MinkowskiModuleBase

class NetworkBase(Module):
    '''
    Abstract Base Class for global network parameters.
    '''
    def __init__(self, cfg, name='network_base'):
        super(NetworkBase, self).__init__()
        model_cfg = cfg['modules'][name]
        self.D = model_cfg.get('D', 3)
        self.num_input = model_cfg.get('num_input', 1)
        self.allow_bias = model_cfg.get('allow_bias', True)
        self.leakiness = model_cfg.get('leakiness', 0.0)

    def forward(self, input):
        raise NotImplementedError
