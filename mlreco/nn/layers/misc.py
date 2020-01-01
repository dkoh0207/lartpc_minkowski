import torch
import torch.nn as nn

# For MinkowskiEngine
import MinkowskiEngine as ME
from SparseTensor import SparseTensor
from MinkowskiNonlinearity import MinkowskiModuleBase


# Custom Nonlinearities
class MinkowskiLeakyReLU(MinkowskiModuleBase):
    MODULE = torch.nn.LeakyReLU


# Custom Network Units/Blocks
class Identity(nn.Module):
    def forward(self, input):
        return input


class ConcatTable(nn.Module):
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


class AddTable(nn.Module):
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


class ResNetBlock(nn.Module):
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


class AtrousIIBlock(nn.Module):
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
        self.norm1 = ME.MinkowskiInstanceNorm(out_features, dimension=self.D)
        self.conv2 = ME.MinkowskiConvolution(
            out_features, out_features,
            kernel_size=3, stride=1, dilation=3, dimension=self.D)
        self.norm2 = ME.MinkowskiInstanceNorm(out_features, dimension=self.D)
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


class ResNeXtBlock(nn.Module):
    '''
    ResNeXt block with leaky relu nonlinearities and atrous convs.

    CONFIGURATIONS:
    -------------------------------------------------------
        - in_features (int): total number of input features

        - out_features (int): total number of output features
          NOTE: if in_features != out_features, then the identity skip
          connection is replaced with a 1x1 conv layer.

        - dimension (int): dimension of dataset.

        - leakiness (float): leakiness for LeakyReLUs.

        - cardinality (int): number of different paths, see ResNeXt paper.

        - depth (int): number of convolutions + BN + LeakyReLU layers inside
        each cardinal path.

        - dilations (int or list of ints): dilation rates for atrous
        convolutions.

        - kernel_sizes (int or list of ints): kernel sizes for each conv layers
        inside cardinal paths.

        - strides (int or list of ints): strides for each conv layers inside
        cardinal paths.
    -------------------------------------------------------
    NOTE: For vanilla resnext blocks, set dilation=1 and others to default.
    '''

    def __init__(self, in_features, out_features, dimension=3,
                 leakiness=0.0, cardinality=4, depth=1,
                 dilations=None, kernel_sizes=3, strides=1):
        super(ResNeXtBlock, self).__init__()
        assert dimension > 0
        assert cardinality > 0
        assert (in_features % cardinality == 0 and
                out_features % cardinality == 0)
        self.D = dimension
        nIn = in_features // cardinality
        nOut = out_features // cardinality

        self.dilations = []
        if dilations is None:
            # Default
            self.dilations = [3**i for i in range(cardinality)]
        elif isinstance(dilations, int):
            self.dilations = [dilations for _ in range(cardinality)]
        elif isinstance(dilations, list):
            assert len(dilations) == cardinality
            self.dilations = dilations
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        self.kernels = []
        if isinstance(kernel_sizes, int):
            self.kernels = [kernel_sizes for _ in range(cardinality)]
        elif isinstance(kernel_sizes, list):
            assert len(kernel_sizes) == cardinality
            self.kernels = kernel_sizes
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        self.strides = []
        if isinstance(strides, int):
            self.strides = [strides for _ in range(cardinality)]
        elif isinstance(strides, list):
            assert len(strides) == cardinality
            self.strides = strides
        else:
            raise ValueError(
                'Invalid type for input strides, must be int or list!')

        # For each path, generate sequentials
        self.paths = []
        for i in range(cardinality):
            m = []
            m.append(ME.MinkowskiLinear(in_features, nIn))
            for j in range(depth):
                in_C = (nIn if j == 0 else nOut)
                m.append(ME.MinkowskiConvolution(
                    in_channels=in_C, out_channels=nOut,
                    kernel_size=self.kernels[i], stride=self.strides[i],
                    dilation=self.dilations[i], dimension=self.D))
                m.append(ME.MinkowskiBatchNorm(nOut))
                m.append(MinkowskiLeakyReLU(negative_slope=leakiness))
            m = nn.Sequential(*m)
            self.paths.append(m)
        self.paths = nn.Sequential(*self.paths)
        self.linear = ME.MinkowskiLinear(out_features, out_features)

        # Skip Connection
        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = Identity()

    def forward(self, x):
        residual = self.residual(x)
        cat = tuple([layer(x) for layer in self.paths])
        out = ME.cat(cat)
        out = self.linear(out)
        out += residual
        return out


class SPP(nn.Module):
    '''
    Spatial Pyramid Pooling Module
    '''

    def __init__(self):
        pass


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module
    '''

    def __init__(self):
        pass
