import os
import pickle
import tempfile
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from typing import Tuple, Union
from collections import OrderedDict

from .utils import check_or_create_dir, same_type


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.contiguous().view(*self.shape)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input):
        return input.squeeze()


class Identity(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()
        self.__name__ = "identity"

    def forward(self, x):
        return x


class OnePlus(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()

    def forward(self, x):
        return F.softplus(x, beta=1)


class Swish(nn.Module):
    def __init__(self, beta=1, trainable_beta=False):
        super(Swish, self).__init__()
        self.beta = torch.zeros(1) + beta
        if trainable_beta:
            self.beta = self.beta.requires_grad_()

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class BWtoRGB(nn.Module):
    def __init__(self):
        super(BWtoRGB, self).__init__()

    def forward(self, x):
        assert len(list(x.size())) == 4
        chans = x.size(1)
        if chans < 3:
            return torch.cat([x, x, x], 1)

        return x


class DistributedDataParallelPassthrough(nn.parallel.DistributedDataParallel):
    """Simple wrapper to still access underlying members of a data-paralleled module.
       Sourced from: https://bit.ly/2VLphRn
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class EvoNormS0(nn.Module):
    """Implementes Evolving Normalization-Activation Layers, Liu et. at 2020."""

    def __init__(self, num_features, groups=32, eps=1e-5):
        super(EvoNormS0, self).__init__()
        assert num_features % groups == 0, "{} % {} != 0 ".format(num_features, groups)
        self.eps = eps
        self.groups = groups
        self.v = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    @staticmethod
    def group_std(x, eps, groups):
        """Calculates groupnorm std-dev."""
        N, C, H, W = x.shape
        orig_type = x.dtype
        x = x.view([N, groups, C // groups, H, W])
        var = torch.var(x.float(), [2, 3, 4], keepdim=True)
        var = var.add_(eps).sqrt_().view(N, -1, 1, 1).type(orig_type)
        # var = var.add_(eps).sqrt_().type(orig_type)
        print("var = ", var.shape)
        return var

    def forward(self, x):
        if self.training:
            num = x * torch.sigmoid(self.v.view(1, -1, 1, 1) * x)
            print("num = ", num.shape)
            return (num / self.group_std(x, self.eps, self.groups) * self.gamma.view(1, -1, 1, 1)
                    + self.beta.view(1, -1, 1, 1))

        return x * self.gamma + self.beta


class Upsample(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=True):
        """ A simple upsampler that also upsamples batches, i.e. [b_j, b_i, c, w, h]
            in addition to the normal 4d images.

        :param size: the required output shape, 2d [w, h]
        :param mode: upsample type
        :param align_corners: whether or not to align the corners from the upsample.
        :returns: upsampled tensor
        :rtype: torch.Tensor

        """
        super(Upsample, self).__init__()
        self.output_shape = size
        self.upsampler = nn.Upsample(size=size, mode=mode, align_corners=align_corners)

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 5:
            x = x.contiguous().view(-1, *original_shape[-3:])
            upsampled = self.upsampler(x)
            return upsampled.contiguous().view(*original_shape[0:3], *self.output_shape)

        assert x.dim() == 4, "input image was neither 5d (batch) nor 4d"
        return self.upsampler(x)


class EMA(nn.Module):
    def __init__(self, decay=0.999):
        """ Simple helper to keep track of exponential moving mean and variance.

        :param decay: the decay, default is decent.
        :returns: EMA module
        :rtype: nn.Module

        """
        super(EMA, self).__init__()
        self.decay = decay
        self.register_buffer('ema_val', None)  # running mean
        self.register_buffer('ema_var', None)  # running variance

    def sample(self):
        """ Return mu + sigma^2 * eps

        :returns: a sample from the running EMA
        :rtype: torch.Tensor

        """
        epsilon = torch.randn_like(self.ema_var)
        return self.ema_val + self.ema_var * epsilon

    def forward(self, x):
        """ Takes an input and creates a variance and mean value if they dont exist and compute EMA & EMA-Var

        :param x: input tensor
        :returns: input tensor itself, just keeps the running variables internally.
        :rtype: torch.Tensor

        """
        if self.ema_val is None:
            self.ema_val = torch.zeros_like(x)
            self.ema_var = torch.zeros_like(x)

        if self.training:  # only update the values if we are in a training state.
            self.ema_val = (1 - self.decay) * x.detach() + self.decay * self.ema_val
            # self.ema_var = (1 - self.decay) * (self.ema_var + self.decay * self.ema_val**2)
            variance = (x.detach() - self.ema_val) ** 2
            self.ema_var = (1 - self.decay) * variance + self.decay * self.ema_var

        return x


class Rotate(nn.Module):
    def __init__(self, angle, resize_shape=None, resize_mode='bilinear', align_corners=True):
        ''' Accepts a batch of tensors, rotates by angle and returns a resized image,
            NOTE: resize_shape is [C, H, W] '''
        super(Rotate, self).__init__()
        self.resize_shape = resize_shape
        self.resize_mode = resize_mode
        self.align_corners = align_corners
        self.angle = angle
        rads = np.pi / 180. * angle
        self.rotation_matrix = torch.zeros(1, 2, 3)
        self.rotation_matrix[:, :, :2] = torch.tensor([[np.cos(rads), -np.sin(rads)],
                                                       [np.sin(rads), np.cos(rads)]],
                                                      dtype=torch.float32)

    def forward(self, x):
        if x.is_cuda and not self.rotation_matrix.is_cuda:
            self.rotation_matrix = self.rotation_matrix.cuda()

        grid = F.affine_grid(self.rotation_matrix.expand(x.size(0), -1, -1), x.size())
        resize_shape = self.resize_shape[-2:] if self.resize_shape is not None else x.shape[-2:]
        return F.interpolate(F.grid_sample(x, grid), size=resize_shape,
                             mode=self.resize_mode,
                             align_corners=self.align_corners)


class AddCoordinates(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)

        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()
        if image_height == image_width == 1:  # handle [B, C, 1, 1] case
            y_coords = torch.tensor([[1]])
            x_coords = torch.tensor([[1]])
        else:
            y_coords = 2.0 * torch.arange(image_height).unsqueeze(
                1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
            x_coords = 2.0 * torch.arange(image_width).unsqueeze(
                0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        coords = coords.type(image.dtype)
        return torch.cat((coords.to(image.device), image), dim=1)


class CoordConv(nn.Module):

    r"""2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)

        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)

        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=True):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)
        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        return self.conv_layer(x)


class CoordConvTranspose(nn.Module):

    r"""2D Transposed Convolution Module Using Extra Coordinate Information
    as defined in 'An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).

    Args:
        Same as `torch.nn.ConvTranspose2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv_tr(input)

        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv_tr(input)

        >>> device = torch.device("cuda:0")
        >>> coord_conv_tr = CoordConvTranspose(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv_tr(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, with_r=True):
        super(CoordConvTranspose, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_tr_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                                kernel_size, stride=stride,
                                                padding=padding,
                                                output_padding=output_padding,
                                                groups=groups, bias=bias,
                                                dilation=dilation)
        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        return self.conv_tr_layer(x)


class BatchGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features):
        """ Batch version of groupnorm, flattens everything to batch and then operates over channels like usual.

        :param num_groups: number of groups to use with batch groupnorm
        :param num_features: number of features in batch groupnorm
        :returns: groupnormed tensor
        :rtype: torch.Tensor

        """
        super(BatchGroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_features)

    def forward(self, x):
        assert len(x.shape) == 5, "batchGroupNorm expects a 5d [{}] tensor".format(x.shape)
        b_i, b_j, c, h, w = x.shape
        out = self.gn(x.contiguous().view(b_i * b_j, c, h, w))
        return out.view([b_i, b_j] + list(out.shape[1:]))


class BatchConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        """ Batch convolve a set of inputs by groups in parallel. Similar to bmm.

        :param in_channels: (b_j, b_i, c_in, h, w) where b_j are the parallel convs to run
        :param out_channels: output channels from conv
        :param kernel_size: size of conv kernel
        :param stride: the stride of the filter
        :param padding: the padding around the input
        :param dilation: the filter dilation
        :param groups: number of parallel ops
        :param bias: whether of not to include a bias term in the conv (bool)
        :returns: tensor of (b_j, b_i, c_out, kh, kw) with batch convolve done
        :rtype: torch.Tensor

        """
        super(BatchConv2D, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels*groups, out_channels*groups,
                              kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x):
        """ (b_j, b_i, c_in, h, w) -> (b_j, b_i * c_in, h, w) --> (b_j, b_i, c_out, h, w)

        :param x: accepts an input of (b_j, b_i, c_in, h, w) where b_j are the parallel groups
        :returns: (b_j, b_i , c_out, kh, kh, kw)
        :rtype: torch.Tensor

        """
        assert len(x.shape) == 5, "batchconv2d expects a 5d [{}] tensor".format(x.shape)
        b_i, b_j, c, h, w = x.shape
        out = self.conv(x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w))
        return out.view(b_j, b_i, self.out_channels,
                        out.shape[-2], out.shape[-1]).permute([1, 0, 2, 3, 4])


class BatchConvTranspose2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1):
        """ Batch conv-transpose a set of inputs by groups in parallel. Similar to bmm.

        :param in_channels: (b_j, b_i, c_in, h, w) where b_j are the parallel convs to run
        :param out_channels: output channels from conv
        :param kernel_size: size of conv kernel
        :param stride: the stride of the filter
        :param padding: the padding around the input
        :param output_padding: the padding of the output volume
        :param groups: number of parallel ops
        :param bias: whether of not to include a bias term in the conv (bool)
        :param dilation: the filter dilation
        :returns: tensor of (b_j, b_i, c_out, kh, kw) with batch convolve done
        :rtype: torch.Tensor

        """
        super(BatchConvTranspose2D, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(in_channels*groups, out_channels*groups,
                                       kernel_size, stride=stride,
                                       padding=padding,
                                       output_padding=output_padding,
                                       groups=groups, bias=bias,
                                       dilation=dilation)

    def forward(self, x):
        """ (b_j, b_i, c_in, h, w) -> (b_j, b_i * c_in, h, w) --> (b_j, b_i, c_out, h, w)

        :param x: accepts an input of (b_j, b_i, c_in, h, w) where b_j are the parallel groups
        :returns: (b_j, b_i , c_out, kh, kh, kw)
        :rtype: torch.Tensor

        """
        assert len(x.shape) == 5, "batchconv2d expects a 5d [{}] tensor".format(x.shape)
        b_i, b_j, c, h, w = x.shape
        out = self.conv(x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w))
        return out.view(b_j, b_i, self.out_channels,
                        out.shape[-2], out.shape[-1]).permute([1, 0, 2, 3, 4])


class MaskedResUnit(nn.Module):
    ''' from jmtomczak's github '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedResUnit, self).__init__()

        self.act = nn.ReLU(True)

        self.h1 = MaskedConv2d(mask_type, *args, **kwargs)
        self.h2 = MaskedConv2d(mask_type, *args, **kwargs)

        self.bn1 = nn.BatchNorm2d(args[0])
        self.bn2 = nn.BatchNorm2d(args[1])

    def forward(self, x):
        h1 = self.bn1(x)
        h1 = self.act(h1)
        h1 = self.h1(h1)

        h2 = self.bn2(h1)
        h2 = self.act(h2)
        h2 = self.h2(h2)
        return x + h2


class MaskedGatedConv2d(nn.Module):
    ''' from jmtomczak's github '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedGatedConv2d, self).__init__()

        self.h = MaskedConv2d(mask_type, *args, **kwargs)
        self.g = MaskedConv2d(mask_type, *args, **kwargs)

    def forward(self, x):
        h = self.h(x)
        g = torch.sigmoid(self.g(x))
        return h * g


class MaskedConv2d(nn.Conv2d):
    ''' from jmtomczak's github '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class GatedConv2d(nn.Module):
    '''from jmtomczak's github '''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, groups=1, activation=None,
                 bias=True, layer_type=nn.Conv2d):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = layer_type(input_channels, output_channels, kernel_size,
                            stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)
        self.g = layer_type(input_channels, output_channels, kernel_size,
                            stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class GatedDense(nn.Module):
    '''similar to gatedconv2d which is from jmtomczak's github '''
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

        # for weight-norm applications
        self.weight = self.h.weight

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class RNNImageClassifier(nn.Module):
    def __init__(self, input_shape, output_size, latent_size=256,
                 n_layers=2, bidirectional=False, rnn_type="lstm",
                 conv_normalization_str="none",
                 dense_normalization_str="none",
                 bias=True, dropout=0,
                 cuda=False, half=False):
        super(RNNImageClassifier, self).__init__()
        self.cuda = cuda
        self.half = half
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.chans = input_shape[0]
        self.bidirectional = bidirectional

        # build the models
        self.feature_extractor = nn.Sequential(
            get_encoder(input_shape, encoder_layer_type='conv',
                        conv_normalization=conv_normalization_str)(output_size=latent_size),
            nn.SELU()
        )
        self.rnn = self._build_rnn(latent_size, bias=bias, dropout=dropout)
        self.output_projector = _build_dense(input_size=latent_size,
                                             output_size=output_size,
                                             normalization_str=dense_normalization_str)
        self.state = None

    def _build_rnn(self, latent_size, model_type='lstm', bias=True, dropout=False):
        if self.half:
            import apex

        model_fn_map = {
            'gru': torch.nn.GRU if not self.half else apex.RNN.GRU,
            'lstm': torch.nn.LSTM if not self.half else apex.RNN.LSTM,
        }
        rnn = model_fn_map[model_type](
            input_size=latent_size,
            hidden_size=latent_size,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            bias=bias, dropout=dropout
        )

        if self.cuda and not self.half:
            rnn.flatten_parameters()

        return rnn

    def forward_rnn(self, x, reset_state=False, return_outputs=False):
        if self.state is None or reset_state:
            self.state = init_state(n_layers=self.n_layers,
                                    batch_size=x.size(0),
                                    hidden_size=self.latent_size,
                                    rnn_type=self.rnn_type,
                                    bidirectional=self.bidirectional,
                                    half=self.half, cuda=self.cuda)

        features = self.feature_extractor(x)
        if features.dim() < 3:
            features = features.unsqueeze(1)

        outputs, self.state = self.rnn(features, self.state)
        return outputs if return_outputs else None

    def forward_prediction(self, reduce_mean=True):
        ''' predicts from the state, but first does a reduce over n_layers [ind 0]'''
        reduce_fn = torch.mean if reduce_mean else torch.sum
        state = self.state[0] if self.rnn_type == 'lstm' else self.state
        return self.output_projector(reduce_fn(state, 0))

    def forward(self, x, reset_state=False, reduce_mean=True):
        ''' helper that does forward rnn and forward prediction '''
        _ = self.forward_rnn(x, reset_state, return_outputs=False)
        return self.forward_prediction(reduce_mean=reduce_mean)


def init_state(n_layers, batch_size, hidden_size, rnn_type='lstm',
               bidirectional=False, noisy=False, half=False, cuda=False):
    ''' return a single initialized state'''
    def _init_state():
        num_directions = 2 if bidirectional else 1
        if noisy:
            # nn.init.xavier_uniform_(
            return same_type(half, cuda)(
                num_directions * n_layers, batch_size, hidden_size
            ).normal_(0, 0.01).requires_grad_(),

        # return zeros if not a noisy state
        return same_type(half, cuda)(
            num_directions * n_layers, batch_size, hidden_size
        ).zero_().requires_grad_()

    if rnn_type == 'lstm':
        return (  # LSTM state is (h, c)
            _init_state(),
            _init_state()
        )

    return _init_state()


class ModelSaver(object):
    def __init__(self, model, early_stop=False, burn_in_interval=20,
                 larger_is_better=False, rank=0, **kwargs):
        """Creates earlystopping or simple best-model storer.
           kwargs contains extra info for EarlyStopping model.

        :param model: nn.Module with save and load fns
        :param early_stop: uses early stopping instead of best-model saver.
        :param burn_in_interval: dont save for at least this many epochs.
        :param larger_is_better: are we maximizing or minimizing?
        :param rank: rank in a DDP setting or 0.
        :returns: ModelSaver Object
        :rtype: object

        """
        self.rank = rank
        self.epoch = 1
        self.model = model
        self.burn_in_interval = burn_in_interval
        self.best_loss = -np.inf if larger_is_better else np.inf
        self.larger_is_better = larger_is_better
        self.saver = EarlyStopping(**kwargs) if early_stop else BestModelSaver(**kwargs)
        print("\nModelSaver: {}\n".format(self.saver))

    def save(self, **kwargs):
        kwargs.setdefault('epoch', self.epoch)
        if self.rank == 0:  # Only save on first DDP rank node.
            self.model.save(**kwargs)

    def restore(self):
        """ Restores the model, optimizer params, set the current epoch and returns

        :returns: state dict with test predicitions dict, test loss dict, epoch
        :rtype: dict

        """
        restore_dict = self.model.load()
        self.epoch = restore_dict['epoch'] + 1
        self.best_loss = restore_dict.get('best_loss', self.best_loss)
        return restore_dict

    def __call__(self, loss, **kwargs):
        """ Calls the underlying save object, but does comparisons here.

        :param loss: current loss
        :returns: early stopping flag (False always for BestModelsaver)
        :rtype: bool

        """
        save_flag = False
        if self.epoch > self.burn_in_interval:
            is_best = self.is_best_loss(loss)
            save_flag = self.saver(loss, is_best)
            if is_best:
                self.best_loss = loss
                self.save(**kwargs)

        self.epoch += 1
        return save_flag

    def is_best_loss(self, loss):
        """ Simply checks whether our new loss is better than the previous best.

        :param loss: current loss
        :returns: flag
        :rtype: bool

        """
        if self.larger_is_better:
            return loss > self.best_loss

        return loss < self.best_loss


class BestModelSaver(object):
    def __repr__(self):
        return 'BestSaver()'

    def __init__(self, **kwargs):
        pass

    def __call__(self, loss, is_best):
        """ Returns false here because we don't want to early stop

        :param loss: current loss
        :param is_best: is it the best so far? (bool)
        :returns: False
        :rtype: bool

        """
        return False


class EarlyStopping(object):
    def __repr__(self):
        return 'EarlyStopping(max_early_stop_steps={})'.format(self.max_steps)

    def __init__(self, max_early_stop_steps=10, **kwargs):
        """ Returns True when loss doesn't change for max_early_stop_steps

        :param max_early_stop_steps: number of steps to observe loss changes
        :returns: EarlyStopping object
        :rtype: object

        """
        self.max_steps = max_early_stop_steps
        self.stopping_steps = 0

    def __call__(self, loss, is_best):
        if is_best:  # reset the counter
            self.stopping_steps = 0
        else:
            self.stopping_steps += 1

        # ES Core Logic
        if self.stopping_steps > self.max_steps:
            self.stopping_steps = 0
            print("Early stopping is triggered:  loss:{:4f}".format(loss))
            return True

        return False


def flatten_layers(model, base_index=0, is_recursive=False):
    """ flattens sequential - sequentials

    :param model: the wrapped sequential model
    :param base_index: the current layer index
    :param is_recursive: an internal param for recursive calls
    :returns: an nn.Sequential that is unrolled
    :rtype: nn.Sequential

    """
    layers = []
    for l in model.children():
        if isinstance(l, nn.Sequential):
            sub_layers, base_index = flatten_layers(l, base_index, is_recursive=True)
            layers.extend(sub_layers)
        else:
            layers.append(('layer_%d' % base_index, l))
            base_index += 1

    if not is_recursive:
        return nn.Sequential(OrderedDict(layers)), base_index

    return layers, base_index


def init_weights(module, init='orthogonal'):
    """Initialize all the weights and biases of a model.

    :param module: any nn.Module or nn.Sequential
    :param init: type of initialize, see dict below.
    :returns: same module with initialized weights
    :rtype: type(module)

    """
    if init is None:  # Base case, no change to default.
        return module

    init_dict = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'orthogonal': nn.init.orthogonal_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
    }

    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # print("initializing {} with {} init.".format(m, init))
            init_dict[init](m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                # print("initial bias from ", m, " with zeros")
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            for mod in m:
                init_weights(mod, init)

    return module


def _compute_group_norm_planes(normalization_str, planes):
    """ Helper to compute group-norm planes

    :param normalization_str: the type of normalization
    :param planes: the output channels
    :returns: number to use for group norm
    :rtype: int

    """
    if planes % 2 == 0 and normalization_str == 'groupnorm':
        gn_planes = max(int(min(np.ceil(planes / 2), 32)), 1)
    elif planes % 3 == 0 and normalization_str == 'groupnorm':
        gn_planes = max(int(min(np.ceil(planes / 3), 32)), 1)
    elif normalization_str != 'groupnorm':
        gn_planes = 0
    else:  # in the case where it's prime
        gn_planes = planes

    return gn_planes


class UpsampleConvLayer(nn.Module):
    '''Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/ '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample,
                                              mode='nearest',
                                              align_corners=None)

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class Annealer(nn.Module):
    def __init__(self, initial_temp=10, min_temp=1e-6, anneal_rate=3e-6, interval=10, use_hard=False):
        """ A simple module that anneals a temperature.

        :param initial_temp: the initial starting temperature
        :param min_temp: the min temp to go to
        :param anneal_rate: the rate of decay
        :param interval: every i-th minibatch to anneal
        :param use_hard: hard decay or smooth exponential decay
        :returns: Annealer object
        :rtype: nn.Module

        """
        super(Annealer, self).__init__()
        self.tau, self.tau0 = initial_temp, initial_temp
        self.anneal_rate = anneal_rate
        self.min_temp = min_temp
        self.anneal_interval = interval
        self.use_hard = use_hard
        self.iteration = 0

    def forward(self):
        """ Returns the current temperature

        :returns: float temp
        :rtype: float

        """
        if self.training \
           and self.iteration > 0 \
           and self.iteration % self.anneal_interval == 0:

            if not self.use_hard:
                # smoother annealing
                rate = -self.anneal_rate * self.iteration
                self.tau = np.maximum(self.tau0 * np.exp(rate), self.min_temp)
                if self.tau < 1e-4:
                    self.tau = self.min_temp
            else:
                # hard annealing
                self.tau = np.maximum(0.9 * self.tau, self.min_temp)

        self.iteration += 1
        return float(self.tau)


class LinearWarmupWithCosineAnnealing(nn.Module):
    def __init__(self, decay_steps, warmup_steps, total_steps, min_value=0.0, constant_for_last_k_steps=0):
        """Linear from [0.0, value_to_scale] followed by cosine decay of [value_to_scale, min_value] over decay_steps.

        :param decay_steps: period over which to decay the cosine wave.
        :param warmup_steps: number of steps to linearly increase between.
        :param total_steps: total steps for model training.
        :param min_value: minimum value as a fraction of value_to_scale
        :param constant_for_last_k_steps: number of steps at the end to stay constant.
        :returns: scaled value_to_scale
        :rtype: float32

        """
        super(LinearWarmupWithCosineAnnealing, self).__init__()
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_value = min_value
        self.is_warming_up = False

        # Used for constant at end logic
        self.constant_for_last_k_steps = constant_for_last_k_steps
        self.total_steps = total_steps
        self.step = 0

        # Pre-compute both the linear warmup and cosine annealing
        self.linear_rate = [max(i / warmup_steps, min_value) for i in range(warmup_steps)]
        self.cosine_rate = [self._cosine_anneal(1.0, i) for i in range(decay_steps)]
        self.lin_idx = 0
        self.cos_idx = 0

    def extra_repr(self):
        """Adds to __repr__ via nn.Module to print some more useful stuff."""
        return 'total_steps={}, decay_steps={}, warmup_steps={}, min_value={}, constant_for_last_k_steps={}'.format(
            self.total_steps, self.decay_steps, self.warmup_steps, self.min_value, self.constant_for_last_k_steps
        )

    def _cosine_anneal(self, value_to_scale, step):
        """Runs cosine annealing given the """
        step = np.minimum(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps))
        decay = (1 - self.min_value) * cosine_decay + self.min_value
        return decay

    def _linear_warmup(self, step):
        """Simple linear warmup given a step"""
        return float(step) / float(max(1.0, self.warmup_steps))

    def forward(self, value_to_scale):
        if self.step >= self.total_steps - self.constant_for_last_k_steps:
            # Handle last K steps where we want it linear.
            return value_to_scale

        if self.is_warming_up:
            # Handle the linear part
            lin_scalar = self.linear_rate[self.lin_idx]
            self.lin_idx = self.lin_idx + 1 if self.training else self.lin_idx
            update = value_to_scale * lin_scalar
        else:
            # Handle the cosine part
            cos_scalar = self.cosine_rate[self.cos_idx]
            self.cos_idx = self.cos_idx + 1 if self.training else self.cos_idx
            update = value_to_scale * cos_scalar

        # Reset the corresponding index if needed
        if self.is_warming_up and self.lin_idx == self.warmup_steps:
            # Reset the linear index; we are now going to cos-part
            self.is_warming_up = False
            self.lin_idx = 0
        elif not self.is_warming_up and self.cos_idx == self.decay_steps:
            # Reset the cos-index; we are now going back to linear
            self.is_warming_up = True
            self.cos_idx = 0

        # increment the global counter (if training) and return the update.
        self.step = self.step + 1 if self.training else self.step
        return update


class LinearWarmupWithFixedInterval(nn.Module):
    def __init__(self, fixed_steps, warmup_steps):
        """Linear from [0.0, value_to_scale] followed by a fixed rate of value_to_scale over fixed_steps.
           Then repeat the same process; produces multiple cycles.

        :param fixed_steps: period over which to keep the fixed value for.
        :param warmup_steps: number of steps to linearly increase between.
        :returns: scaled value_to_scale
        :rtype: float32

        """
        super(LinearWarmupWithFixedInterval, self).__init__()
        self.fixed_steps = fixed_steps
        self.warmup_steps = warmup_steps
        self.is_warming_up = True

        # Pre-compute both the linear warmup and fixed rate
        self.linear_rate = [i / warmup_steps for i in range(warmup_steps)]
        self.lin_idx = 0
        self.fixed_idx = 0

    def extra_repr(self):
        """Adds to __repr__ via nn.Module to print some more useful stuff."""
        return 'fixed_steps={}, warmup_steps={}'.format(
            self.fixed_steps, self.warmup_steps
        )

    def _linear_warmup(self, step):
        """Simple linear warmup given a step"""
        return float(step) / float(max(1.0, self.warmup_steps))

    def forward(self, value_to_scale):
        if self.is_warming_up:
            lin_scalar = self.linear_rate[self.lin_idx]
            self.lin_idx = self.lin_idx + 1 if self.training else self.lin_idx
            update = value_to_scale * lin_scalar
        else:
            self.fixed_idx = self.fixed_idx + 1 if self.training else self.fixed_idx
            update = value_to_scale

        # Update whether we are warming up or not
        if self.is_warming_up and self.lin_idx == self.warmup_steps:
            self.is_warming_up = False
            self.lin_idx = 0
        elif not self.is_warming_up and self.fixed_idx == self.fixed_steps:
            self.is_warming_up = True
            self.fixed_idx = 0

        return update


class DenseResnet(nn.Module):
    def __init__(self, input_size, output_size, normalization_str="none", activation_fn=nn.ReLU):
        """ Resnet, but with dense layers

        :param input_size: the input size
        :param output_size: output size
        :param normalization_str: what type of normalization to use
        :param activation_fn: the activation function module
        :returns: DenseResnet object
        :rtype: nn.Module

        """
        super(DenseResnet, self).__init__()
        self.dense1 = add_normalization(nn.Linear(input_size, output_size), normalization_str, 1, output_size)
        self.dense2 = add_normalization(nn.Linear(output_size, output_size), normalization_str, 1, output_size)
        self.downsampler = add_normalization(nn.Linear(input_size, output_size),
                                             normalization_str, 1, output_size)
        self.act = activation_fn()

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.dense1(x.view(batch_size, -1))
        out = self.act(out)
        out = self.dense2(out)

        # add residual part and return
        residual = self.downsampler(x)
        out += residual
        return self.act(out)


class AttentionBlock(nn.Module):
    def __init__(self, input_shape, embedding_size, num_heads=1, normalization_str="none", **kwargs):
        """ An attention block that takes an input [B, input_size] and returns [B, embedding_size].
            S: source seq len (assumed 1)
            L: target seq len (assumed 1)
            N: batch_size
            E: embedding size

        :param input_shape: the input shape of the tensor
        :param embedding_size: the output size (equivalent to embedding size)
        :param num_heads: the number of heads to use in parallel, NOTE: needs to be divisible by embedding_size
        :param normalization_str: the type of dense normalization to use
        :returns: AttentionBlock Module
        :rtype: nn.Module

        """
        super(AttentionBlock, self).__init__()
        # (S, N, E)
        self.key_net = nn.Sequential(
            _build_dense(input_shape, embedding_size, num_layers=3,
                         normalization_str=normalization_str, **kwargs),
            View([1, -1, embedding_size])
        )

        # (L, N, E)
        self.query_net = nn.Sequential(
            _build_dense(input_shape, embedding_size, num_layers=3,
                         normalization_str=normalization_str, **kwargs),
            View([1, -1, embedding_size])
        )

        # (S, N, E)
        self.value_net = nn.Sequential(
            _build_dense(input_shape, embedding_size, num_layers=3,
                         normalization_str=normalization_str, **kwargs),
            View([1, -1, embedding_size])
        )

        self.attn = nn.MultiheadAttention(embedding_size, num_heads=num_heads)

    def forward(self, x):
        """ Takes an input x, projects to key-dim, query-dim and value-dim and runs attn.

        :param x: input tensor [B, input_size]
        :returns: output tensor [B, embedding_size]
        :rtype: torch.Tensor

        """
        key = self.key_net(x)
        query = self.query_net(x)
        value = self.value_net(x)

        out = self.attn(query, key, value, need_weights=False)[0]
        return out.squeeze()


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_fn, resample=None,
                 normalization_str="batchnorm", init_norm=True,
                 activation_str="relu", **kwargs):
        super(ResnetBlock, self).__init__()
        gn_groups = {"num_groups": _compute_group_norm_planes(normalization_str, out_channels)}
        norm_fn = functools.partial(
            add_normalization, normalization_str=normalization_str,
            ndims=2, nfeatures=out_channels, **gn_groups)

        # The actual underlying model
        self.resample = resample
        self.init_norm = None
        if normalization_str not in ['weightnorm', 'spectralnorm'] and init_norm:  # handle special case of weight hooks
            self.init_norm = norm_fn(Identity(), nfeatures=in_channels)

        self.conv1 = norm_fn(layer_fn(in_channels, out_channels))
        self.act = str_to_activ(activation_str)
        self.conv2 = layer_fn(out_channels, out_channels)

        # Learnable skip-connection
        self.skip_connection = None
        if in_channels != out_channels or resample is not None:
            self.skip_connection = layer_fn(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = x
        if self.init_norm is not None:
            out = self.act(self.init_norm(x))

        if self.resample:
            out = self.resample(out)
            x = self.resample(x)

        out = self.act(self.conv1(x))
        out = self.conv2(out)

        if self.skip_connection is not None:
            x = self.skip_connection(x)

        return out + x


def str_to_activ_module(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    activ_map = {
        'identity': Identity,
        'elu': nn.ELU,
        'sigmoid': nn.Sigmoid,
        'log_sigmoid': nn.LogSigmoid,
        'tanh': nn.Tanh,
        'oneplus': OnePlus,
        'swish': Swish,
        'softmax': nn.Softmax,
        'log_softmax': nn.LogSoftmax,
        'selu': nn.SELU,
        'relu': nn.ReLU,
        'softplus': nn.Softplus,
        'hardtanh': nn.Hardtanh,
        'leaky_relu': nn.LeakyReLU,
        'softsign': nn.Softsign
    }

    assert str_activ in activ_map, "unknown activation requested"
    return activ_map[str_activ]


def str_to_activ(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    activ_map = {
        'identity': lambda x: x,
        'elu': F.elu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'oneplus': lambda x: F.softplus(x, beta=1),
        'swish': lambda x: x * torch.sigmoid(x),
        'softmax': F.softmax,
        'log_softmax': F.log_softmax,
        'selu': F.selu,
        'relu': F.relu,
        'softplus': F.softplus,
        'hardtanh': F.hardtanh,
        'leaky_relu': F.leaky_relu,
        'softsign': F.softsign
    }

    assert str_activ in activ_map, "unknown activation requested"
    return activ_map[str_activ]


def build_image_downsampler(img_shp, new_shp,
                            stride=[3, 3],
                            padding=[0, 0]):
    '''Takes a tensor and returns a downsampling operator'''
    equality_test = np.asarray(img_shp) == np.asarray(new_shp)
    if equality_test.all():
        return Identity()

    height = img_shp[0]
    width = img_shp[1]
    new_height = new_shp[0]
    new_width = new_shp[1]

    # calculate the width and height by inverting the equations from:
    # http://pytorch.org/docs/master/nn.html?highlight=avgpool2d#torch.nn.AvgPool2d
    kernel_width = -1 * ((new_width - 1) * stride[1] - width - 2 * padding[1])
    kernel_height = -1 * ((new_height - 1) * stride[0] - height - 2 * padding[0])
    print('kernel = ', kernel_height, 'x', kernel_width)
    assert kernel_height > 0
    assert kernel_width > 0

    return nn.AvgPool2d(kernel_size=(kernel_height, kernel_width),
                        stride=stride, padding=padding)


def build_pixelcnn_decoder(input_size, output_shape, filter_depth=64,
                           activation_fn=nn.ReLU, normalization_str="none",
                           nr_logistic_mix=10, layer_type=MaskedConv2d, **kwargs):
    ''' modified from jmtomczak's github, do not use, use submodule pixelcnn '''
    # warnings.warn("use pixelcnn from helpers submodule instead, this is not tested")
    chans = output_shape[0]
    # num_mix = 3 if chans == 1 else 10

    return nn.Sequential(
        add_normalization(layer_type('A', input_size, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        add_normalization(layer_type('B', filter_depth, filter_depth, 3, 1, 1, bias=False),
                          normalization_str, 2, filter_depth, num_groups=32), activation_fn(),
        # nn.Conv2d(filter_depth, num_mix * nr_logistic_mix, 1, 1, 0, dilation=1, bias=True)
        # nn.Conv2d(filter_depth, num_mix * nr_logistic_mix, 1)
        nn.Conv2d(filter_depth, chans, 1, bias=False)
    )


def add_normalization(module, normalization_str, ndims, nfeatures, **kwargs):
    norm_map = {
        'sync_batchnorm': {
            1: lambda nfeatures, **kwargs: nn.SyncBatchNorm(nfeatures),
            2: lambda nfeatures, **kwargs: nn.SyncBatchNorm(nfeatures),
            3: lambda nfeatures, **kwargs: nn.SyncBatchNorm(nfeatures),
        },
        'batchnorm': {
            1: lambda nfeatures, **kwargs: nn.BatchNorm1d(nfeatures),
            2: lambda nfeatures, **kwargs: nn.BatchNorm2d(nfeatures),
            3: lambda nfeatures, **kwargs: nn.BatchNorm3d(nfeatures)
        },
        'groupnorm': {
            2: lambda nfeatures, **kwargs: nn.GroupNorm(kwargs['num_groups'], nfeatures)
        },
        'batch_groupnorm': {
            2: lambda nfeatures, **kwargs: BatchGroupNorm(kwargs['num_groups'], nfeatures)
        },
        'instancenorm': {
            1: lambda nfeatures, **kwargs: nn.Sequential(
                View([-1, 1, nfeatures]),
                nn.InstanceNorm1d(nfeatures),
                View([-1, nfeatures])
            ),
            2: lambda nfeatures, **kwargs: nn.InstanceNorm2d(nfeatures),
            3: lambda nfeatures, **kwargs: nn.InstanceNorm3d(nfeatures)
        },
        'weightnorm': {
            1: lambda nfeatures, **kwargs: nn.utils.weight_norm(module),
            2: lambda nfeatures, **kwargs: nn.utils.weight_norm(module),
            3: lambda nfeatures, **kwargs: nn.utils.weight_norm(module)
        },
        'spectralnorm': {
            1: lambda nfeatures, **kwargs: nn.utils.spectral_norm(module),
            2: lambda nfeatures, **kwargs: nn.utils.spectral_norm(module),
            3: lambda nfeatures, **kwargs: nn.utils.spectral_norm(module)
        },
        'none': {
            1: lambda nfeatures, **kwargs: Identity(),
            2: lambda nfeatures, **kwargs: Identity(),
            3: lambda nfeatures, **kwargs: Identity()
        }
    }

    if normalization_str == 'groupnorm':
        assert 'num_groups' in kwargs, "need to specify groups for GN"
        assert ndims > 1, "group norm needs channels to operate"

    if normalization_str == 'weightnorm' or normalization_str == 'spectralnorm':
        return norm_map[normalization_str][ndims](nfeatures, **kwargs)

    return nn.Sequential(module, norm_map[normalization_str][ndims](nfeatures, **kwargs))


def _build_resnet_stack(input_chans, output_chans,
                        layer_fn,
                        base_channels,
                        channel_multiplier,
                        kernels,
                        strides,
                        resample,
                        resample_fn,
                        attentions=None,
                        activation_str="relu",
                        normalization_str="none",
                        norm_first_layer=False,
                        norm_last_layer=False,
                        **kwargs):
    """ Helper to build an arbitrary convolutional decoder.

    :param input_chans: number of input channels
    :param output_chans: output channel dimension
    :param layer_fn: what layer function to use?
    :param base_channels: base feature maps
    :param channel_multiplier: expand by this per layer, usually < 1
    :param kernels: list of kernels per layer
    :param strides: list of strides for each layer
    :param resample: a list of boolean flags for each layer
    :param resample_fn: the actual function to use for resampling
    :param attentions: list of attention boolean flags or None
    :param activation_str: what activation function to use
    :param normalization_str: layer normalization type, eg: batchnorm
    :param norm_first_layer: apply normalization to the input layer?
    :param norm_last_layer: apply normalization to the final layer?
    :returns: a model with a bunch of conv layers.
    :rtype: nn.Sequential

    """
    if attentions is None:  # Make attentions the same size as the rest
        attentions = [None] * len(kernels)

    assert len(resample) == len(kernels) == len(strides) == len(attentions)

    # Normalization for pre and post model.
    norm_fn = functools.partial(
        add_normalization, module=Identity(), normalization_str=normalization_str, ndims=2)
    layers = []

    if norm_first_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
        init_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, input_chans)}
        layers.append(norm_fn(nfeatures=input_chans, **init_gn_groups))
        # layers.append(activation_fn())  # TODO(jramapuram): consider this.

    # build the channel map.
    channels = [input_chans, int(base_channels)]
    for i in range(len(kernels) - 2):  # -1 because last one is output_shape[1]
        channels.append(int(channel_multiplier * channels[-1]))

    channels.append(output_chans)

    # build the rest of the layers, from 0 --> end -1
    for idx, (k, s, r, a, chan_in, chan_out) in enumerate(zip(kernels, strides, resample, attentions,
                                                              channels[0:-1], channels[1:])):
        # Build the layer definition
        padding_i = 1 if k > 1 else 0  # 1x1 doesn't need padding.
        layer_fn_i = functools.partial(layer_fn, kernel_size=k, stride=s, padding=padding_i)
        resample_fn_i = resample_fn if r else lambda x: x
        init_norm_i = norm_first_layer if idx == 0 else True

        # Construct the actual underlying layer
        layer_i = ResnetBlock(chan_in, chan_out,
                              resample=resample_fn_i,
                              layer_fn=layer_fn_i,
                              normalization_str=normalization_str,
                              init_norm=init_norm_i,
                              activation_str=activation_str)
        layers.append(layer_i)

        # Add attention block
        if chan_out >= 4 and a:
            layers.append(Attention(chan_out, layer_fn))

    # Add normalization to the final layer if requested
    if norm_last_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
        final_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, output_chans)}
        layers.append(norm_fn(nfeatures=channels[-1], **final_gn_groups))

    return nn.Sequential(*layers)


def _build_conv_stack(input_chans, output_chans,
                      layer_fn,
                      base_channels,
                      channel_multiplier,
                      kernels,
                      strides,
                      attentions=None,
                      activation_str="relu",
                      normalization_str="none",
                      norm_first_layer=False,
                      norm_last_layer=False,
                      **kwargs):
    """ Helper to build an arbitrary convolutional decoder.

    :param input_chans: number of input channels
    :param output_chans: output channel dimension
    :param layer_fn: what layer function to use?
    :param base_channels: base feature maps
    :param channel_multiplier: expand by this per layer, usually < 1
    :param kernels: list of kernels per layer
    :param strides: list of strides for each layer
    :param attentions: list of bools for each layer or None
    :param activation_str: what activation function to use
    :param normalization_str: layer normalization type, eg: batchnorm
    :param norm_first_layer: apply normalization to the input layer?
    :param norm_last_layer: apply normalization to the final layer?
    :returns: a model with a bunch of conv layers.
    :rtype: nn.Sequential

    """
    if attentions is None:  # Make attentions the same size as the rest
        attentions = [None] * len(kernels)

    assert len(kernels) == len(strides) == len(attentions)

    # Normalization and activation helpers
    norm_fn = functools.partial(
        add_normalization, normalization_str=normalization_str, ndims=2)
    activation_fn = str_to_activ_module(activation_str)
    layers = []

    if norm_first_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
        init_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, input_chans)}
        layers.append(norm_fn(Identity(), nfeatures=input_chans, **init_gn_groups))
        # layers.append(activation_fn())  # TODO(jramapuram): consider this.

    # build the channel map.
    channels = [input_chans, int(base_channels)]
    for i in range(len(kernels) - 2):  # -2 because last one is output_shape[1] and first is input_chans
        channels.append(int(channel_multiplier*channels[-1]))

    channels.append(output_chans)

    # build each individual layer
    for idx, (k, s, a, chan_in, chan_out) in enumerate(zip(kernels, strides, attentions, channels[0:-1], channels[1:])):
        is_last_layer = (idx == len(kernels) - 1)
        if is_last_layer:
            normalization_str = 'none' if norm_last_layer is False else normalization_str
            norm_fn = functools.partial(norm_fn, normalization_str=normalization_str)

        li_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, chan_out)}
        layer_i = norm_fn(layer_fn(chan_in, chan_out, kernel_size=k, stride=s),
                          nfeatures=chan_out, **li_gn_groups)
        layers.append(layer_i)

        if not is_last_layer:
            if chan_out >= 4 and a:
                layers.append(Attention(chan_out, layer_fn))

            layers.append(activation_fn())

    return nn.Sequential(*layers)


class Attention(nn.Module):
    def __init__(self, ch, conv_fn):
        """Attention from SAGAN with modification from BigGAN.
           NOTE: very unstable without spectral-normed conv2d

           From https://github.com/ajbrock/BigGAN-PyTorch/

        :param ch: inputs channels
        :param conv_fn: what type of convolution to use
        :returns: Attention module
        :rtype: nn.Module

        """

        super(Attention, self).__init__()

        # Channel multiplier
        self.ch = ch
        self.which_conv = conv_fn
        self.theta = nn.utils.spectral_norm(self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.phi = nn.utils.spectral_norm(self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.g = nn.utils.spectral_norm(self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False))
        self.o = nn.utils.spectral_norm(self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False))

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])

        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)

        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class Conv32UpsampleDecoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=UpsampleConvLayer):
        super(Conv32UpsampleDecoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder normalization
        norm_layer = Identity()
        if norm_first_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_size)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_size]),
            norm_layer,
            add_normalization(nn.Linear(input_size, input_size*4*4),
                              normalization_str, ndims=1, nfeatures=input_size*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_size, 4, 4]),
        )

        # The main model
        self.model = _build_conv_stack(input_chans=input_size,
                                       output_chans=output_chans,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[3, 3, 3],
                                       strides=[1, 1, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=False,  # Handled already
                                       norm_last_layer=norm_last_layer)

    def forward(self, images, upsample_last=False):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        outputs = self.mlp_proj(images)
        outputs = self.model(outputs)

        if upsample_last:
            return F.upsample(outputs, size=outputs.shape[-2:],
                              mode='bilinear',
                              align_corners=True)

        return outputs


class VolumePreservingResnet(nn.Module):
    def __init__(self, input_chans, base_channels=256, num_layers=3,
                 activation_str="relu", normalization_str="none",
                 norm_first_layer=False, norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(VolumePreservingResnet, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_chans as float or int."

        # The encoding part of the model
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=input_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=1.0,
                                         kernels=[1] * num_layers,
                                         strides=[1] * num_layers,
                                         resample=[False] * num_layers,
                                         attentions=[False] * num_layers,
                                         resample_fn=Identity(),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=norm_first_layer,
                                         norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        return self.model(images)


class Resnet32Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet32Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder and post-decoder normalization
        def build_norm_layer(use_norm, feature_size, ndims=1):
            norm_layer = Identity()
            if use_norm and normalization_str not in ['weightnorm', 'spectralnorm']:
                norm_layer = add_normalization(norm_layer, normalization_str,
                                               ndims=ndims, nfeatures=feature_size)
            return norm_layer

        init_norm = build_norm_layer(norm_first_layer, input_size)
        final_norm = build_norm_layer(norm_last_layer, output_chans, ndims=2)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_size]),
            init_norm,
            add_normalization(nn.Linear(input_size, input_size*4*4),
                              normalization_str, ndims=1, nfeatures=input_size*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_size, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_size,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3],
                                         strides=[1, 1, 1],
                                         resample=[True, True, True],
                                         attentions=[True, True, True],
                                         resample_fn=functools.partial(F.interpolate, scale_factor=2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # Handled already
                                         norm_last_layer=True)    # We have a final conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_chans, output_chans, kernel_size=1, stride=1),
            final_norm
        )

    def forward(self, images, upsample_last=False):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        outputs = self.mlp_proj(images)
        outputs = self.model(outputs)
        outputs = self.final_conv(outputs)

        if upsample_last:
            return F.upsample(outputs, size=outputs.shape[-2:],
                              mode='bilinear',
                              align_corners=True)

        return outputs


class Resnet64Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet64Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder and post-decoder normalization
        def build_norm_layer(use_norm, feature_size, ndims=1):
            norm_layer = Identity()
            if use_norm and normalization_str not in ['weightnorm', 'spectralnorm']:
                norm_layer = add_normalization(norm_layer, normalization_str,
                                               ndims=ndims, nfeatures=feature_size)
            return norm_layer

        init_norm = build_norm_layer(norm_first_layer, input_size)
        final_norm = build_norm_layer(norm_last_layer, output_chans, ndims=2)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_size]),
            init_norm,
            add_normalization(nn.Linear(input_size, input_size*4*4),
                              normalization_str, ndims=1, nfeatures=input_size*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_size, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_size,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3, 3],
                                         strides=[1, 1, 1, 1],
                                         resample=[True, True, True, True],
                                         # attentions=[False, False, False, False],
                                         attentions=[True, True, True, True],
                                         resample_fn=functools.partial(F.interpolate, scale_factor=2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # Handled already
                                         norm_last_layer=norm_last_layer)
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_chans, output_chans, kernel_size=1, stride=1),
            final_norm
        )

    def forward(self, images, upsample_last=False):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        outputs = self.mlp_proj(images)
        outputs = self.model(outputs)
        outputs = self.final_conv(outputs)

        if upsample_last:
            return F.upsample(outputs, size=outputs.shape[-2:],
                              mode='bilinear',
                              align_corners=True)

        return outputs


class Resnet128Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet128Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder and post-decoder normalization
        def build_norm_layer(use_norm, feature_size, ndims=1):
            norm_layer = Identity()
            if use_norm and normalization_str not in ['weightnorm', 'spectralnorm']:
                norm_layer = add_normalization(norm_layer, normalization_str,
                                               ndims=ndims, nfeatures=feature_size)
            return norm_layer

        init_norm = build_norm_layer(norm_first_layer, input_size)
        final_norm = build_norm_layer(norm_last_layer, output_chans, ndims=2)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_size]),
            init_norm,
            add_normalization(nn.Linear(input_size, input_size*4*4),
                              normalization_str, ndims=1, nfeatures=input_size*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_size, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_size,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3, 3, 3],
                                         strides=[1, 1, 1, 1, 1],
                                         resample=[True, True, True, True, True],
                                         attentions=[True, True, True, True, True],
                                         resample_fn=functools.partial(F.interpolate, scale_factor=2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # Handled already
                                         norm_last_layer=norm_last_layer)
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_chans, output_chans, kernel_size=1, stride=1),
            final_norm
        )

    def forward(self, images, upsample_last=False):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        outputs = self.mlp_proj(images)
        outputs = self.model(outputs)
        outputs = self.final_conv(outputs)

        if upsample_last:
            return F.upsample(outputs, size=outputs.shape[-2:],
                              mode='bilinear',
                              align_corners=True)

        return outputs


class Conv32Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=False,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv32Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_size,
                                       output_chans=output_chans,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[5, 4, 4, 4],  # NEW ver
                                       strides=[1, 2, 1, 2],  # NEW ver
                                       # kernels=[5, 1, 4, 1, 4, 1, 4, 1],  # NEW small ver
                                       # strides=[1, 1, 2, 1, 1, 1, 2, 1],  # NEW small ver
                                       # kernels=[5, 4, 4, 4, 4, 1, 1],  # OLD ver
                                       # strides=[1, 2, 1, 2, 1, 1, 1],  # OLD ver
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=norm_first_layer,
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        return self.model(images)


class Conv28Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=False,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv28Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_size,
                                       output_chans=output_chans,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[4, 4, 4, 4, 1],
                                       strides=[1, 2, 1, 2, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=norm_first_layer,
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        return self.model(images)


class Conv64Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv64Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_size,
                                       output_chans=output_chans,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[7, 5, 5, 5, 4, 4, 2],
                                       strides=[2, 1, 2, 1, 2, 1, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=norm_first_layer,
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        return self.model(images)


class Conv128Decoder(nn.Module):
    def __init__(self, input_size, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv128Decoder, self).__init__()
        assert isinstance(input_size, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_size,
                                       output_chans=output_chans,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[7, 7, 7, 7, 7, 5, 4],
                                       strides=[2, 2, 1, 2, 1, 2, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=norm_first_layer,
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        if len(images.shape) == 2:
            images = images.view(*images.shape, 1, 1)

        return self.model(images)


class Resnet128Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Resnet128Encoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."
        self.act = str_to_activ(activation_str)

        # The main model
        kernels = [3, 3, 3, 3, 3, 3]
        strides = [1, 1, 1, 1, 1, 1]
        resample = [True, True, True, True, True, False]
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_size,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=kernels,
                                         strides=strides,
                                         resample=resample,
                                         resample_fn=nn.AvgPool2d(2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # raw data
                                         norm_last_layer=True)   # input to 1x1

        # Handle norm_last_layer if requested
        norm_layer = Identity()
        if norm_last_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Do a final linear projection on the pooled representation
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_size, output_size, kernel_size=1),
            norm_layer
        )

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        outputs = self.model(images)
        outputs = torch.mean(self.act(outputs), [-2, -1])     # pool over x and y
        outputs = outputs.view(list(outputs.shape) + [1, 1])  # un-flatten and do 1x1
        outputs = self.final_conv(outputs)                    # 1x1 conv
        return outputs


class Resnet64Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Resnet64Encoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."
        self.act = str_to_activ(activation_str)

        # The main model
        kernels = [3, 3, 3, 3, 3]
        strides = [1, 1, 1, 1, 1]
        resample = [True, True, True, True, False]
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_size,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=kernels,
                                         strides=strides,
                                         resample=resample,
                                         resample_fn=nn.AvgPool2d(2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # raw data
                                         norm_last_layer=True)    # input to 1x1

        # Handle norm_last_layer if requested
        norm_layer = Identity()
        if norm_last_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Do a final linear projection on the pooled representation
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_size, output_size, kernel_size=1),
            norm_layer
        )

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        outputs = self.model(images)
        outputs = torch.mean(self.act(outputs), [-2, -1])     # pool over x and y
        outputs = outputs.view(list(outputs.shape) + [1, 1])  # un-flatten and do 1x1
        outputs = self.final_conv(outputs)                    # 1x1 conv
        return outputs


class Resnet32Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Resnet32Encoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."
        self.act = str_to_activ(activation_str)

        # The main model
        kernels = [3, 3, 3, 3]
        strides = [1, 1, 1, 1]
        resample = [True, True, True, False]
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_size,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=kernels,
                                         strides=strides,
                                         resample=resample,
                                         resample_fn=nn.AvgPool2d(2),
                                         activation_str=activation_str,
                                         normalization_str=normalization_str,
                                         norm_first_layer=False,  # raw data
                                         norm_last_layer=True)    # input to 1x1

        # Handle norm_last_layer if requested
        norm_layer = Identity()
        if norm_last_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=2, nfeatures=input_chans)

        # Do a final linear projection on the pooled representation
        self.final_conv = nn.Sequential(
            nn.Conv2d(output_size, output_size, kernel_size=1),
            norm_layer
        )

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        outputs = self.model(images)
        outputs = torch.mean(self.act(outputs), [-2, -1])     # pool over x and y
        outputs = outputs.view(list(outputs.shape) + [1, 1])  # un-flatten and do 1x1
        outputs = self.final_conv(outputs)                    # 1x1 conv
        return outputs


class TorchvisionEncoder(nn.Module):
    """Wraps torchvision models such as Resnet50, etc."""
    def __init__(self, pretrained_output_size, output_size, latent_size=512, activation_str="relu",
                 normalization_str="none", norm_first_layer=False, norm_last_layer=False,
                 layer_fn=models.resnet50, pretrained=False, freeze_base=False,
                 **unused_args):
        """ Wrap a torchvision encoder such as resnet50 and adds an FC.

        :param pretrained_output_size: output size of the pretrained model
        :param output_size: output size for FC projection
        :param latent_size: latent size for FC projection
        :param activation_str: activation for FC layers
        :param normalization_str: normalization for FC
        :param norm_first_layer: norm input to FC (output of base model)
        :param norm_last_layer: norm output of FC
        :param layer_fn: layer fn for FC
        :param pretrained: pull pretrained classifier weights for torchvision model
        :param freeze_base: prevent updating of gradients of base model
        :returns: module with FC attached
        :rtype: nn.Module

        """
        super(TorchvisionEncoder, self).__init__()
        self.output_size = output_size
        self.latent_size = latent_size
        self.norm_first_layer = norm_first_layer
        self.norm_last_layer = norm_last_layer
        self.normalization_str = normalization_str
        self.activation_str = activation_str

        # Compute the input image size; TODO(jramapuram): do we need this for non-pretrained?
        # Ideally the pooling layer should auto-magically handle this for us.
        model_input_size = (299, 299) if layer_fn == models.inception_v3 else (224, 224)
        self.required_input_shape = model_input_size if pretrained else None

        # Build the torchvision model and (optionally) load the pretained weights.
        self.model = nn.Sequential(
            *list(layer_fn(pretrained=pretrained).children())[:-1]
        )
        if normalization_str == 'sync_batchnorm':
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Freeze base model if requested
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False

        # Lazy initialize this.
        self.fc = _build_dense(input_size=pretrained_output_size,
                               output_size=self.output_size,
                               latent_size=self.latent_size,
                               num_layers=3,  # XXX(jramapuram): hardcoded for now
                               layer_fn=nn.Linear,
                               activation_str=self.activation_str,
                               normalization_str=self.normalization_str,
                               norm_first_layer=self.norm_first_layer,
                               norm_last_layer=self.norm_last_layer)

    def forward(self, images):
        """Infers using the given torchvision model and projects with the FC.

        :param images: image tensor
        :param required_input_shape: None or tuple of (w, h)
        :returns: fc output logits
        :rtype: torch.tensor

        """
        if self.required_input_shape is not None and (images.shape[-2] != self.required_input_shape[-2]
                                                      and images.shape[-1] != self.required_input_shape[-1]):
            images = F.interpolate(images, size=self.required_input_shape,
                                   mode='bilinear', align_corners=True)

        outputs = self.model(images)
        return self.fc(outputs.squeeze())


class Conv32Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Conv32Encoder, self).__init__()
        assert isinstance(output_size, (float, int)), "Expect output_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
                                       output_chans=output_size,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       # TODO(jramapuram): consider this
                                       # kernels=[3, 4, 4, 3, 3, 3],
                                       # strides=[1, 2, 1, 2, 1, 1],
                                       kernels=[4, 4, 3, 3, 3],
                                       strides=[2, 1, 2, 1, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=False,  # dont norm inputs
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        return self.model(images)


class Conv28Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Conv28Encoder, self).__init__()
        assert isinstance(output_size, (float, int)), "Expect output_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
                                       output_chans=output_size,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[3, 4, 4, 4, 1],
                                       strides=[1, 2, 2, 2, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=False,  # dont norm inputs
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        return self.model(images)


class Conv64Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Conv64Encoder, self).__init__()
        assert isinstance(output_size, (float, int)), "Expect output_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
                                       output_chans=output_size,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[5, 4, 4, 3, 3, 3, 2],
                                       strides=[2, 1, 2, 1, 2, 1, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=False,  # dont norm inputs
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        return self.model(images)


class Conv128Encoder(nn.Module):
    def __init__(self, input_chans, output_size, base_channels=32, channel_multiplier=2,
                 activation_str="relu", normalization_str="none", norm_last_layer=False,
                 layer_fn=nn.Conv2d):
        super(Conv128Encoder, self).__init__()
        assert isinstance(output_size, (float, int)), "Expect output_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
                                       output_chans=output_size,
                                       layer_fn=layer_fn,
                                       base_channels=base_channels,
                                       channel_multiplier=channel_multiplier,
                                       kernels=[7, 5, 4, 4, 3, 3, 3, 2],
                                       strides=[2, 2, 1, 2, 1, 2, 1, 1],
                                       activation_str=activation_str,
                                       normalization_str=normalization_str,
                                       norm_first_layer=False,  # dont norm inputs
                                       norm_last_layer=norm_last_layer)

    def forward(self, images):
        """Iterate over each of the layers to produce an output."""
        assert len(images.shape) == 4, "Require [B, C, H, W] inputs."
        return self.model(images)


def _build_dense(input_size,
                 output_size,
                 layer_fn=nn.Linear,
                 latent_size=512, num_layers=3,
                 activation_str="elu", normalization_str="none",
                 norm_first_layer=False, norm_last_layer=False):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
    assert normalization_str != "groupnorm", "Groupnorm not supported for dense models."

    # Activation and normalization functions
    activation_fn = str_to_activ_module(activation_str)
    norm_fn = functools.partial(
        add_normalization, normalization_str=normalization_str, ndims=1)

    # add initial norm if requested
    init_norm_layer = Identity()
    if norm_first_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
        init_norm_layer = norm_fn(init_norm_layer, nfeatures=input_size)

    # add final norm if requested
    final_norm_layer = Identity()
    if norm_last_layer and normalization_str not in ['weightnorm', 'spectralnorm']:
        final_norm_layer = norm_fn(final_norm_layer, nfeatures=output_size)

    layers = [('init_norm', init_norm_layer),
              ('l0', norm_fn(layer_fn(input_size, latent_size), nfeatures=latent_size)),
              ('act0', activation_fn())]

    for i in range(num_layers - 2):  # 2 for init layer[above] + final layer[below]
        layers.append(
            ('l{}'.format(i+1), norm_fn(layer_fn(latent_size, latent_size), nfeatures=latent_size)),
        )
        layers.append(('act{}'.format(i+1), activation_fn()))

    layers.append(('output', layer_fn(latent_size, output_size)))
    layers.append(('final_norm', final_norm_layer))
    return nn.Sequential(OrderedDict(layers))


class Dense(nn.Module):
    def __init__(self, input_shape, output_shape,
                 latent_size=512, num_layers=3,
                 activation_str="relu", normalization_str="none",
                 norm_first_layer=False, norm_last_layer=False,
                 layer_fn=nn.Linear):
        super(Dense, self).__init__()
        input_shape = [input_shape] if not isinstance(input_shape, (list, tuple)) else input_shape
        output_shape = [output_shape] if not isinstance(output_shape, (list, tuple)) else output_shape
        input_size = int(np.prod(input_shape))
        output_size = int(np.prod(output_shape))

        # the views and model
        self.input_view = View([-1, input_size])
        self.model = _build_dense(input_size=input_size,
                                  output_size=output_size,
                                  latent_size=latent_size,
                                  num_layers=num_layers,
                                  layer_fn=layer_fn,
                                  activation_str=activation_str,
                                  normalization_str=normalization_str,
                                  norm_first_layer=norm_first_layer,
                                  norm_last_layer=norm_last_layer)
        self.output_view = View([-1, *output_shape])

    def forward(self, inputs):
        h = self.input_view(inputs)
        h = self.model(h)
        return self.output_view(h)


class DenseEncoder(Dense):
    def __init__(self, input_shape, output_size,
                 latent_size=512, num_layers=3,
                 activation_str="relu", normalization_str="none",
                 norm_first_layer=False, norm_last_layer=False,
                 layer_fn=nn.Linear):
        super(DenseEncoder, self).__init__(input_shape=input_shape,
                                           output_shape=[output_size],
                                           latent_size=latent_size,
                                           num_layers=num_layers,
                                           activation_str=activation_str,
                                           normalization_str=normalization_str,
                                           norm_first_layer=norm_first_layer,
                                           norm_last_layer=norm_last_layer,
                                           layer_fn=layer_fn)


class DenseDecoder(Dense):
    def __init__(self, input_size, output_shape,
                 latent_size=512, num_layers=3,
                 activation_str="relu", normalization_str="none",
                 norm_first_layer=False, norm_last_layer=False,
                 layer_fn=nn.Linear):
        super(DenseDecoder, self).__init__(input_shape=[input_size],
                                           output_shape=output_shape,
                                           latent_size=latent_size,
                                           num_layers=num_layers,
                                           activation_str=activation_str,
                                           normalization_str=normalization_str,
                                           norm_first_layer=norm_first_layer,
                                           norm_last_layer=norm_last_layer,
                                           layer_fn=layer_fn)


def get_conv_encoder(input_shape: Tuple[int, int, int],  # [C, H, W]
                     encoder_layer_type: str = 'conv',
                     encoder_base_channels: int = 32,  # For conv models
                     encoder_channel_multiplier: int = 2,
                     latent_size: int = 512,   # For dense models
                     dense_normalization: str = 'none',
                     conv_normalization: str = 'none',
                     disable_gated: bool = True,
                     norm_first_layer: bool = False,
                     norm_last_layer: bool = False,
                     activation: str = 'relu',
                     pretrained: bool = False,
                     name: str = 'encoder',
                     **unused_kwargs):
    '''Helper to return the correct encoder function.'''
    conv_size_dict = {
        128: Conv128Encoder,
        64: Conv64Encoder,
        32: Conv32Encoder,
        28: Conv28Encoder,
    }
    resnet_size_dict = {
        128: Resnet128Encoder,
        64: Resnet64Encoder,
        32: Resnet32Encoder,
        28: lambda **kwargs: None  # XXX: fix
    }
    chans, image_size = input_shape[0], input_shape[-1]

    # Mega-dict that curried the appropriate encoder.
    # The returned encoder still needs the CTOR, eg: enc(input_shape)
    net_map = {
        'resnet50': {
            False: functools.partial(TorchvisionEncoder,
                                     pretrained_output_size=2048,  # r50 avg-pool size
                                     latent_size=latent_size,
                                     activation_str=activation,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     pretrained=pretrained,
                                     freeze_base=False,  # TODO(jramapuram): parameterize
                                     layer_fn=models.resnet50),
        },
        'resnet': {
            # True for gated, False for non-gated
            True: functools.partial(resnet_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=GatedConv2d),
            False: functools.partial(resnet_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=nn.Conv2d),
        },
        'conv': {
            True: functools.partial(conv_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=GatedConv2d),
            False: functools.partial(conv_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=nn.Conv2d),
        },
        'batch_conv': {
            True: functools.partial(conv_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=BatchConv2D)),
            False: functools.partial(conv_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=BatchConv2D),
        },
        'coordconv': {
            True: functools.partial(conv_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=CoordConv)),
            False: functools.partial(conv_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=CoordConv),
        },
    }

    fn = net_map[encoder_layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        encoder_layer_type,
        name
    ))
    return fn


def get_encoder(input_shape: Union[int, Tuple[int, int, int]],  # [C, H, W]
                encoder_layer_type: str = 'conv',
                encoder_base_channels: int = 32,  # For conv models
                encoder_channel_multiplier: int = 2,
                latent_size: int = 512,   # For dense models
                num_layers: int = 3,      # For dense models
                dense_normalization: str = 'none',
                conv_normalization: str = 'none',
                disable_gated: bool = True,
                norm_first_layer: bool = False,
                norm_last_layer: bool = False,
                activation: str = 'relu',
                pretrained: bool = False,
                name: str = 'encoder',
                **unused_kwargs):
    '''Helper to return the correct encoder function.'''
    if encoder_layer_type != 'dense':
        return get_conv_encoder(input_shape=input_shape,
                                encoder_layer_type=encoder_layer_type,
                                encoder_base_channels=encoder_base_channels,
                                encoder_channel_multiplier=encoder_channel_multiplier,
                                latent_size=latent_size,
                                dense_normalization=dense_normalization,
                                conv_normalization=conv_normalization,
                                disable_gated=disable_gated,
                                norm_first_layer=norm_first_layer,
                                norm_last_layer=norm_last_layer,
                                activation=activation,
                                pretrained=pretrained,
                                name=name, **unused_kwargs)

    # Handle dense model building separately since there are no size restrictions.
    # The returned encoder still needs the CTOR, eg: enc(input_shape)
    net_map = {
        'dense': {
            # True for gated, False for non-gated
            True: functools.partial(DenseEncoder,
                                    input_shape=input_shape,
                                    latent_size=latent_size,
                                    num_layers=num_layers,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    normalization_str=dense_normalization,
                                    layer_fn=GatedDense),
            False: functools.partial(DenseEncoder,
                                     input_shape=input_shape,
                                     latent_size=latent_size,
                                     num_layers=num_layers,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     normalization_str=dense_normalization,
                                     layer_fn=nn.Linear)
        }
    }

    fn = net_map[encoder_layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        encoder_layer_type,
        name
    ))
    return fn


def get_conv_decoder(output_shape: Tuple[int, int, int],  # output image shape [B, H, W]
                     decoder_layer_type: str = 'conv',
                     decoder_base_channels: int = 1024,      # For conv models
                     decoder_channel_multiplier: int = 0.5,  # Decoding shrinks channels
                     latent_size: int = 512,         # For dense models
                     dense_normalization: str = 'none',
                     conv_normalization: str = 'none',
                     disable_gated: bool = True,
                     norm_first_layer: bool = True,
                     norm_last_layer: bool = False,
                     activation: str = 'relu',
                     name: str = 'decoder',
                     **unused_kwargs):
    '''Helper to return the correct decoder function.'''
    conv_size_dict = {
        128: Conv128Decoder,
        64: Conv64Decoder,
        32: Conv32Decoder,
        # 32: Conv32UpsampleDecoder,
        28: Conv28Decoder,
    }
    resnet_size_dict = {
        128: Resnet128Decoder,
        64: Resnet64Decoder,
        32: Resnet32Decoder,
        28: lambda **kwargs: None  # XXX: fix
    }
    image_size = output_shape[-1]

    # Mega-dict that curried the appropriate decoder.
    # The returned decoder still needs the CTOR, eg: dec(input_size)
    net_map = {
        'resnet': {
            # True for gated, False for non-gated
            True: functools.partial(resnet_size_dict[image_size],
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=nn.Conv2d)),
            False: functools.partial(resnet_size_dict[image_size],
                                     output_chans=output_shape[0],
                                     base_channels=decoder_base_channels,
                                     channel_multiplier=decoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=nn.Conv2d)
        },
        'conv': {
            True: functools.partial(conv_size_dict[image_size],
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=nn.ConvTranspose2d)),
            False: functools.partial(conv_size_dict[image_size],
                                     output_chans=output_shape[0],
                                     base_channels=decoder_base_channels,
                                     channel_multiplier=decoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=nn.ConvTranspose2d)
        },
        'batch_conv': {
            True: functools.partial(conv_size_dict[image_size],
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=BatchConvTranspose2D)),
            False: functools.partial(conv_size_dict[image_size],
                                     output_chans=output_shape[0],
                                     base_channels=decoder_base_channels,
                                     channel_multiplier=decoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=BatchConvTranspose2D),
        },
        'coordconv': {
            True: functools.partial(conv_size_dict[image_size],
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=CoordConvTranspose)),
            False: functools.partial(conv_size_dict[image_size],
                                     output_chans=output_shape[0],
                                     base_channels=decoder_base_channels,
                                     channel_multiplier=decoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=CoordConvTranspose),
        },
    }

    fn = net_map[decoder_layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        decoder_layer_type,
        name
    ))
    return fn


def get_decoder(output_shape: Union[int, Tuple[int, int, int]],  # output image shape [B, H, W]
                decoder_layer_type: str = 'conv',
                decoder_base_channels: int = 1024,      # For conv models
                decoder_channel_multiplier: int = 0.5,  # Decoding shrinks channels
                latent_size: int = 512,   # For dense models
                num_layers: int = 3,      # For dense models
                dense_normalization: str = 'none',
                conv_normalization: str = 'none',
                disable_gated: bool = True,
                norm_first_layer: bool = True,
                norm_last_layer: bool = True,
                activation: str = 'relu',
                name: str = 'decoder',
                **unused_kwargs):
    '''Helper to return the correct decoder function.'''
    if decoder_layer_type != 'dense':
        return get_conv_decoder(output_shape=output_shape,
                                decoder_layer_type=decoder_layer_type,
                                decoder_base_channels=decoder_base_channels,
                                decoder_channel_multiplier=decoder_channel_multiplier,
                                latent_size=latent_size,
                                dense_normalization=dense_normalization,
                                conv_normalization=conv_normalization,
                                disable_gated=disable_gated,
                                norm_first_layer=norm_first_layer,
                                norm_last_layer=norm_last_layer,
                                activation=activation,
                                name=name, **unused_kwargs)

    # Handle dense model building separately since there are no size restrictions.
    # The returned decoder still needs the CTOR, eg: dec(input_size)
    net_map = {
        'dense': {
            # True for gated, False for non-gated
            True: functools.partial(DenseDecoder,
                                    output_shape=output_shape,
                                    latent_size=latent_size,
                                    num_layers=num_layers,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    normalization_str=dense_normalization,
                                    layer_fn=GatedDense),
            False: functools.partial(DenseDecoder,
                                     output_shape=output_shape,
                                     latent_size=latent_size,
                                     num_layers=num_layers,
                                     norm_first_layer=True,  # input data
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     normalization_str=dense_normalization,
                                     layer_fn=nn.Linear)
        },
    }

    fn = net_map[decoder_layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        decoder_layer_type,
        name
    ))
    return fn


def get_polyak_prediction(model, pred_fn):
    """Backs up the model, sets EMA mean parameters, runs the prediction and returns."""
    ema_mean = model.polyak_ema.ema_val
    original_params = nn.utils.parameters_to_vector(model.parameters())
    nn.utils.vector_to_parameters(ema_mean, model.parameters())
    preds = pred_fn()
    nn.utils.vector_to_parameters(original_params, model.parameters())
    return preds


def polyak_ema_parameters(model, decay=0.9999):
    """Apply Polyak averaging to the provided model with given decay."""
    assert hasattr(model, 'polyak_ema'), "model needs to create EMA op as a member and initialize it."
    model.polyak_ema(nn.utils.parameters_to_vector(model.parameters()))


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """Splits param group into weight_decay / non-weight decay.
       Tweaked from https://bit.ly/3dzyqod

    :param model: the torch.nn model
    :param weight_decay: weight decay term
    :param skip_list: extra modules (besides BN/bias) to skip
    :returns: split param group into weight_decay/not-weight decay
    :rtype: list(dict)

    """
    # if weight_decay == 0:
    #     return model.parameters()

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay':  0, 'ignore': True},
        {'params': decay, 'weight_decay': weight_decay, 'ignore': False}]


def append_save_and_load_fns(model, optimizer, scheduler, grapher, args):
    """ Hax to add save and load functionality to use with early-stopping module.

    :param model: any torch module
    :param optimizer: the optimizer to save
    :param grapher: the visdom or tensorboard object
    :param args: argparse
    :returns: the same module with the added methods
    :rtype: nn.Module

    """
    from .utils import get_name

    def load(model, optimizer, scheduler, grapher, **kwargs):
        """ load the model if it exists, returns the cached dictionary

        :param model: the nn.Module
        :param optimizer: nn.Optim
        :returns: dictionary of losses and predictions
        :rtype: dict

        """
        save_dict = {'epoch': 1}

        prefix = kwargs.get('prefix', '')
        if os.path.isdir(args.model_dir):
            model_filename = os.path.join(args.model_dir, prefix + get_name(args) + ".th")
            if os.path.isfile(model_filename):
                # set the map location to the gpu0 or None ; devices are set via CUDA_VISIBLE_DEVICES.
                map_location = 'cuda:0' if args.cuda else None
                print("loading existing model: {} to map-loc {}".format(model_filename, map_location))

                # load the full dictionary and set the model and optimizer params
                save_dict = torch.load(model_filename, map_location=map_location)
                model.load_state_dict(save_dict['model'])
                optimizer.load_state_dict(save_dict['optimizer'])
                scheduler.load_state_dict(save_dict['scheduler'])
                if grapher is not None:
                    grapher.state_dict = save_dict['grapher']

                # remove the keys that we used to load the models
                del save_dict['model']
                del save_dict['optimizer']
                del save_dict['scheduler']
            else:
                print("{} does not exist...".format(model_filename))

        # restore the visdom grapher
        if grapher is not None and 'grapher' in save_dict and save_dict['grapher'] \
           and 'scalars' in save_dict['grapher'] and save_dict['grapher']['scalars']:
            grapher.set_data(save_dict['grapher']['scalars'], save_dict['grapher']['windows'])
            del save_dict['grapher']

        return save_dict

    def save(model, optimizer, scheduler, grapher, **kwargs):
        """ Saves a model and optimizer (w/scheduler) to a file.

            Optional params:
                  -  'overwrite' : force over-writes a savefile
                  -  'prefix': prefix the save file with this st
                  -  'epoch': the epoch that were at

        :param model: nn.Module
        :param optimizer: nn.Optim
        :param scheduler: optim.lr_scheduler
        :returns: None
        :rtype: None

        """
        kwargs.setdefault('overwrite', True)
        kwargs.setdefault('prefix', '')
        kwargs.setdefault('epoch', 1)

        check_or_create_dir(args.model_dir)
        model_filename = os.path.join(args.model_dir, kwargs['prefix'] + get_name(args) + ".th")
        if not os.path.isfile(model_filename) or kwargs['overwrite']:
            print("saving existing model to {}".format(model_filename))

            # HAX: write the scalars to a temp file and re-read them
            scalar_dict, window_dict = {}, {}
            if grapher is not None:
                with tempfile.NamedTemporaryFile() as scalar, tempfile.NamedTemporaryFile() as window:
                    grapher.pickle_data(scalar.name, window.name)
                    scalar_dict = pickle.load(scalar.file)
                    window_dict = pickle.load(window.file)

            # save the entire state
            torch.save(
                {**{
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args,
                    'grapher': {'scalars': scalar_dict, 'windows': window_dict}
                }, **kwargs},
                model_filename
            )

    grapher = None if args.visdom_url is None else grapher  # nothing to save for tensorboard.
    model.load = functools.partial(load, model=model, grapher=grapher, optimizer=optimizer, scheduler=scheduler)
    model.save = functools.partial(save, model=model, grapher=grapher, optimizer=optimizer, scheduler=scheduler)
    return model
