from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import MinkowskiEngine as ME
import torch

def CollateSparse(batch):
    """
    INPUTS:
      batch - a tuple of dictionary. Each tuple element (single dictionary)
      is a minibatch data = key-value pairs where a value is a parser function return.
    OUTPUT:
      return - a dictionary of key-value pair where key is same as keys in the
      input batch, and the value is a list of data elements in the input.
    ASSUMES:
      - The input batch is a tuple of length >=1.
        Length 0 tuple will fail (IndexError).
      - The dictionaries in the input batch tuple are
        assumed to have identical list of keys.
    """
    concat = np.concatenate
    result = {}
    print("BATCH")
    print(batch)
    for key in batch[0].keys():
        if isinstance(batch[0][key], tuple) and isinstance(batch[0][key][0], np.ndarray) and len(batch[0][key][0].shape)==2:
            # handle SCN input batch
            voxels = concat( [ concat( [sample[key][0],
                                        np.full(shape=[len(sample[key][0]),1], fill_value=batch_id, dtype=np.int32)],
                                       axis=1 ) for batch_id, sample in enumerate(batch) ],
                             axis = 0)
            data = concat([sample[key][1] for sample in batch], axis=0)
            result[key] = concat([voxels, data], axis=1)
        elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==1:
            result[key] = concat( [ concat( [np.expand_dims(sample[key],1),
                                             np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
                                            axis=1 ) for batch_id,sample in enumerate(batch) ],
                                  axis=0)
        elif isinstance(batch[0][key],np.ndarray) and len(batch[0][key].shape)==2:
            result[key] =  concat( [ concat( [sample[key],
                                              np.full(shape=[len(sample[key]),1],fill_value=batch_id,dtype=np.float32)],
                                             axis=1 ) for batch_id,sample in enumerate(batch) ],
                                   axis=0)
        elif isinstance(batch[0][key], list) and isinstance(batch[0][key][0], tuple):
            result[key] = [
                concat([
                    concat( [ concat( [sample[key][depth][0],
                                                np.full(shape=[len(sample[key][depth][0]),1], fill_value=batch_id, dtype=np.int32)],
                                               axis=1 ) for batch_id, sample in enumerate(batch) ],
                                     axis = 0),
                    concat([sample[key][depth][1] for sample in batch], axis=0)
                ], axis=1) for depth in range(len(batch[0][key]))
            ]
        else:
            result[key] = [sample[key] for sample in batch]
    return result

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
