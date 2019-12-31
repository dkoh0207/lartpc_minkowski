import torch
from torch.nn import Module

# For MinkowskiEngine
import MinkowskiEngine as ME
from SparseTensor import SparseTensor
from MinkowskiNonlinearity import MinkowskiModuleBase


# Custom Nonlinearities
class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = torch.nn.LeakyReLU


# Custom Network Units/Blocks
class Identity(Module):
    def forward(self, input):
        return input


class ConcatTable(Module):
    '''
    Util function for concatenating feature planes of sparse tensors.
    '''
    def __init__(self, D=3):
        super(ConcatTable, self).__init__()
    def forward(self, x, y):
        '''
        WARNING: Input coordinates of x and y must agree exactly!
        '''
        outputFeatures = [x.F, y.F]
        outputFeatures = torch.cat(outputFeatures, dim=1)
        coords = x.C
        output = SparseTensor(outputFeatures, coords=coords)
        return output


class AddTable(Module):
    '''
    Util function for adding feature planes of sparse tensors.
    '''
    def __init__(self, D=3):
        super(AddTable, self).__init__()
    def forward(self, x, y):
        '''
        WARNING: Input coordinates AND dimensions of x and y must agree exactly!
        '''
        outputFeatures = x.F + y.F
        coords = x.coords
        output = SparseTensor(outputFeatures, coords=coords)
        return output


class ResNetBlock(Module):
    '''
    ResNet Block with Leaky ReLU nonlinearities.
    '''
    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 leakiness=0.0,
                 dimension=-1):
        super(ResNetBlock, self).__init__()
        assert dimension > 0

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=3,
            stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(out_features, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(out_features, momentum=bn_momentum)
        self.leaky_relu = MinkowskiLeakyReLU(negative_slope=leakiness)

    def forward(self, x):

        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.leaky_relu(out)
        return out


class AtrousIIBlock(Module):
    '''
    ResNet-type block with Atrous Convolutions, as developed in ACNN paper:
    <ACNN: a Full Resolution DCNN for Medical Image Segmentation>
    Original Paper: https://arxiv.org/pdf/1901.09203.pdf
    '''
    def __init__(self, in_features, out_features, dimension=3, leakiness=0.0):
        super(AtrousIIBlock, self).__init__()
        assert dimension > 0
        self.D = dimension

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()
        self.conv1 = ME.MinkowskiConvolution(
            in_features, out_features,
            kernel_size=3, stride=1, dilation=1, dimension=self.D)
        self.norm1 = ME.MinkowskiInstanceNorm(out_features)
        self.conv1 = ME.MinkowskiConvolution(
            out_features, out_features,
            kernel_size=3, stride=1, dilation=3, dimension=self.D)
        self.norm2 = ME.MinkowskiInstanceNorm(out_features)
        self.leaky_relu = MinkowskiLeakyReLU(negative_slope=leakiness)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.leaky_relu(out)
        return out


class SPP(Module):
    '''
    Spatial Pyramid Pooling Module
    '''
    def __init__(self):
        pass

class ASPP(Module):
    '''
    Atrous Spatial Pyramid Pooling Module
    '''
    def __init__(self):
        pass

class ResNeXtBlock(Module):
    '''
    ResNeXt-type block with leaky relu nonlinearities.
    '''

class AtrousResNeXtBlock(Module):
    '''
    ResNeXt-type block with leaky relu nonlinearities.
    '''
