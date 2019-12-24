import numpy as np
import torch
import torch.nn

import ROOT
ROOT.gSystem.Load("/usr/local/cuda/lib64/libcusparse.so")
import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.utils.misc import ResNetBlock, MinkowskiLeakyReLU
from mlreco.layers.network_base import NetworkBase

class UResNet(NetworkBase):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configurations
    --------------
    depth : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    num_filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    input_kernel : int, optional
        Receptive field size for very first convolution after input layer.
    '''
    def __init__(self, model_cfg, name='uresnet'):
        super(UResNet, NetworkBase).__init__(model_cfg)
        cfg = model_cfg[name]

        # UResNet Configurations
        self.reps = cfg.get('reps', 2)
        self.depth = cfg.get('depth', 5)
        self.num_filters = cfg.get('num_filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        # self.kernel_size = cfg.get('kernel_size', 3)
        # self.downsample = cfg.get(downsample, 2)
        self.input_kernel = cfg.get('input_kernel', 3)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F, dimension=self.D,
                                     leakiness=self.leakiness))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(ME.MinkowskiBatchNorm(F))
                m.append(MinkowskiLeakyReLU(F, negative_slope=self.leakiness))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i]))
            m.append(MinkowskiLeakyReLU(self.nPlanes[i],
                                        negative_slope=self.leakiness))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i],
                out_channels=self.nPlanes[i-1],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                    self.nPlanes[i],
                                     dimension=self.D,
                                     leakiness=self.leakiness))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder.

        INPUTS:
            - x (SparseTensor): MinkowskiEngine SparseTensor

        RETURNS:
            - result (dict): dictionary of encoder output with
            intermediate feature planes:
              1) encoderTensors (list): list of intermediate SparseTensors
              2) finalTensor (SparseTensor): feature tensor at
              deepest layer.
        '''
        encoderTensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderFeatures.append(x)
            x = self.encoding_conv[i](x)

        result = {
            "encoderTensors": encoderTensors,
            "finalTensor": x
        }
        return result


    def decoder(self, final, encoderTensors):
        '''
        Vanilla UResNet Decoder
        INPUTS:
            - encoderTensors (list of SparseTensor): output of encoder.
        RETURNS:
            - decoderTensors (list of SparseTensor):
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i-2]
            x = layer(x)
            x = ME.cat((eTensor, x))
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
        return {"decoderTensors": decoderTensors}

    def forward(self, input):
        coords = input[:, 0:self.dimension+1].int()
        features = input[:, self.dimension+1:].float()

        x = SparseTensor(features, coords=coords)
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': [encoderTensors],
            'decoderTensors': [decoderTensors]
        }
