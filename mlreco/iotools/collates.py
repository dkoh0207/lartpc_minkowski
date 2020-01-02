from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import MinkowskiEngine as ME
import torch

def CollateMinkowski(batch):
    '''
    INPUTS:
        - batch: tuple of dictionary?
    '''
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], tuple):
            data_list = []
            coords = [sample[key][0] for sample in batch]
            features = [sample[key][1] for sample in batch]
            coords, features = ME.utils.sparse_collate(coords, features)
            result[key] = torch.cat([coords.float(), features], dim=1)
        else:
            result[key] = [sample[key] for sample in batch]
    return result
