import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.nn.layers.misc import *
from mlreco.nn.layers.network_base import NetworkBase


class UResNeXt(NetworkBase):
    '''
    UNet Type encoder-decoder network, with atrous convolutions and
    resnext-type blocks.
    '''

    def __init__(self, cfg, name='uresnext'):
        super(UResNeXt, self).__init__(cfg)
        self.model_cfg = cfg['modules'][name]

        # Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('depth', 5)
        self.num_filters = self.model_cfg.get('num_filters', 16)
        self.cardinality = self.model_cfg.get('cardinality', 4)
        assert (self.num_filters % self.cardinality == 0)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth + 1)]
        self.input_kernel = self.model_cfg.get('input_kernel', 3)

        # Initialize Input Layer
        self.input_layer = AtrousIIBlock(self.num_input,
                                         self.num_filters,
                                         leakiness=self.leakiness)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNeXtBlock(F, F, dimension=self.D,
                                      leakiness=self.leakiness,
                                      cardinality=4,
                                      dilations=[1, 1, 3, 9]))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth - 1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(MinkowskiLeakyReLU(negative_slope=self.leakiness))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i + 1],
                    kernel_size=2, stride=2, dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth - 2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i + 1]))
            m.append(MinkowskiLeakyReLU(negative_slope=self.leakiness))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i + 1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNeXtBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                      self.nPlanes[i],
                                      dimension=self.D,
                                      cardinality=4,
                                      dilations=[1, 1, 3, 9],
                                      leakiness=self.leakiness))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        print('Total Number of Trainable Parameters = {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def encoder(self, x):
        '''
        UResNeXt Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        x = self.input_layer(x)
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result

    def decoder(self, final, encoderTensors):
        '''
        UResNeXt Decoder

        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i - 2]
            x = layer(x)
            x = ME.cat((eTensor, x))
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
        return decoderTensors

    def forward(self, input):
        coords = input[:, 0:self.D + 1].cpu().int()
        features = input[:, self.D + 1:].float()

        x = ME.SparseTensor(features, coords=coords)
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': encoderTensors,
            'decoderTensors': decoderTensors
        }
        return res
