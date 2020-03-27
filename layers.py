import os
import pickle
import tempfile
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
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
            build_conv_encoder(input_shape, latent_size,
                               normalization_str=conv_normalization_str),
            nn.SELU())
        self.rnn = self._build_rnn(latent_size, bias=bias, dropout=dropout)
        self.output_projector = build_dense_encoder(latent_size, output_size,
                                                    normalization_str=dense_normalization_str, nlayers=2)
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
        outputs = self.forward_rnn(x, reset_state, return_outputs=False)
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
    def __init__(self, args, model, burn_in_interval=20, larger_is_better=False, **kwargs):
        """ Creates earlystopping or simple best-model storer.

        :param args: argparse object
        :param model: nn.Module with save and load fns
        :param burn_in_interval: dont save for at least this many epochs.
        :param larger_is_better: are we maximizing or minimizing?
        :returns: ModelSaver Object
        :rtype: object

        """
        self.epoch = 1
        self.model = model
        self.burn_in_interval = burn_in_interval
        self.best_loss = -np.inf if larger_is_better else np.inf
        self.larger_is_better = larger_is_better
        self.saver = EarlyStopping(**kwargs) if args.early_stop else BestModelSaver(**kwargs)

    def save(self, **kwargs):
        kwargs.setdefault('epoch', self.epoch)
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


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("initializing ", m, " with xavier init")
            nn.init.xavier_uniform(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod)

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
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 normalization_str="groupnorm",
                 activation_fn=nn.ReLU, **kwargs):
        super(ResnetBlock, self).__init__()
        layer_type = kwargs['layer_type'] if 'layer_type' in kwargs else ResnetBlock.conv3x3
        self.gn_planes = _compute_group_norm_planes(normalization_str, planes)
        self.conv1 = add_normalization(layer_type(inplanes, planes, stride),
                                       normalization_str, 2, planes, num_groups=self.gn_planes)
        # self.act = str_to_activ(activation_str)
        self.act = activation_fn()
        self.conv2 = add_normalization(layer_type(planes, planes),
                                       normalization_str, 2, planes, num_groups=self.gn_planes)
        self.stride = stride
        self.downsample = self.downsampler(inplanes, planes, normalization_str) \
            if downsample is not None else None

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    @staticmethod
    def gated_conv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return GatedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)

    def downsampler(self, inplanes, planes, normalization_str):
        return add_normalization(nn.Conv2d(inplanes, planes, kernel_size=1,
                                           stride=self.stride, bias=False),
                                 normalization_str, 2, planes, num_groups=self.gn_planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ResnetDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_fn, upsample=None,
                 normalization_str="batchnorm", activation_str="relu", **kwargs):
        super(ResnetDeconvBlock, self).__init__()
        gn_groups = {"num_groups": _compute_group_norm_planes(normalization_str, out_channels)}
        norm_fn = functools.partial(
            add_normalization, normalization_str=normalization_str,
            ndims=2, nfeatures=out_channels, **gn_groups)

        # The actual underlying model
        self.upsample = upsample
        self.norm1 = norm_fn(Identity(), nfeatures=in_channels)  # TODO(jramapuram): doesn't handle weightnorm
        self.conv1 = norm_fn(layer_fn(in_channels, out_channels))
        self.act = str_to_activ(activation_str)
        self.conv2 = layer_fn(out_channels, out_channels)
        # self.conv2 = norm_fn(layer_fn(out_channels, out_channels))

        # Learnable skip-connection
        self.skip_connection = None
        if in_channels != out_channels or upsample is not None:
            self.skip_connection = layer_fn(in_channels, out_channels,
                                            kernel_size=1, padding=0)

    def forward(self, x):
        out = self.act(self.norm1(x))
        if self.upsample:
            out = self.upsample(out)
            x = self.upsample(x)
            print("[upsample] out = {} | res = {}".format(out.shape, x.shape))

        out = self.act(self.conv1(x))
        print("out1 = ", out.shape)
        out = self.conv2(out)
        print("out2 = ", out.shape)

        if self.skip_connection is not None:
            x = self.skip_connection(x)

        print("[final] out = ", out.shape, " | res = ", x.shape)
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


def build_volume_preserving_resnet(input_shape, filter_depth=32,
                                   activation_fn=nn.SELU, num_layers=4,
                                   normalization_str="none", **kwargs):
    assert num_layers % 2 == 0, "need even layers for upscale --> downscale"
    chans = input_shape[0]

    def _make_layer(inchan, outchan, stride):
        return ResnetBlock(inchan, outchan, stride=1,
                           normalization_str=normalization_str,
                           activation_fn=activation_fn,
                           downsample=True,
                           **kwargs)

    normalization_override = 'none' if normalization_str == 'groupnorm' else normalization_str
    return nn.Sequential(
        ResnetBlock(chans, filter_depth, stride=1, normalization_str=normalization_override, downsample=True, activation_fn=activation_fn, **kwargs),
        *[_make_layer(filter_depth*(2**i), filter_depth*(2**(i+1)), stride=1) for i in range(num_layers//2)],
        *[_make_layer(filter_depth*(2**(i+1)), filter_depth*(2**i), stride=1) for i in range(num_layers//2 - 1, -1, -1)],
        ResnetBlock(filter_depth, chans, stride=1, normalization_str=normalization_override, downsample=True, activation_fn=activation_fn, **kwargs),
        # nn.Conv2d(chans, chans, kernel_size=1, stride=1),
        Squeeze()
    )


def _build_resnet_stack(input_chans, output_chans,
                        layer_fn,
                        base_channels,
                        channel_multiplier,
                        kernels,
                        strides,
                        upsample,  # variadic bool list
                        activation_str="relu",
                        normalization_str="none",
                        norm_first_layer=False,
                        norm_last_layer=False,
                        **kwargs):
    """ Helper to build an arbitrary convolutional decoder.

    :param input_chans: number of input channels
    :param output_shape: [C, H, W] output shape
    :param layer_fn: what layer function to use?
    :param base_channels: base feature maps
    :param channel_multiplier: expand by this per layer, usually < 1
    :param kernels: list of kernels per layer
    :param strides: list of strides for each layer
    :param activation_str: what activation function to use
    :param normalization_str: layer normalization type, eg: batchnorm
    :param norm_first_layer: apply normalization to the input layer?
    :param norm_last_layer: apply normalization to the final layer?
    :returns: a model with a bunch of conv layers.
    :rtype: nn.Sequential

    """
    assert len(upsample) == len(kernels) == len(strides)

    # Normalization for pre and post model.
    norm_fn = functools.partial(
        add_normalization, module=Identity(), normalization_str=normalization_str, ndims=2)
    layers = []

    if norm_first_layer:
        init_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, input_chans)}
        layers.append(norm_fn(nfeatures=input_chans, **init_gn_groups))
        # layers.append(activation_fn())  # TODO(jramapuram): consider this.

    # build the channel map.
    channels = [input_chans, int(base_channels)]
    for i in range(len(kernels) - 2):  # -1 because last one is output_shape[1]
        channels.append(int(channel_multiplier*channels[-1]))

    channels.append(output_chans)

    # build the rest of the layers, from 0 --> end -1
    # for k, s, u, chan_in, chan_out in zip(kernels[0:-1], strides[0:-1], upsample, channels[0:-1], channels[1:]):
    for k, s, u, chan_in, chan_out in zip(kernels, strides, upsample, channels[0:-1], channels[1:]):
        # Build the layer definition
        layer_fn_i = functools.partial(layer_fn, kernel_size=k, stride=s, padding=1)
        upsample_i = functools.partial(F.interpolate, scale_factor=2) if u else None

        # Construct the actual underlying layer
        layer_i = ResnetDeconvBlock(chan_in, chan_out,
                                    upsample=upsample_i,
                                    layer_fn=layer_fn_i,
                                    normalization_str=normalization_str,
                                    activation_str=activation_str)
        layers.append(layer_i)
        # TODO(jramapuram): consider adding attention
        # layers.append(Attention(chan_out, SNConv2d))

    # build the final layer
    # layer_fn_i = functools.partial(layer_fn, kernel_size=kernels[-1], stride=strides[-1], padding=1)
    # upsample_i = functools.partial(F.interpolate, scale_factor=2) if upsample[-1] else None
    # layers.append(ResnetDeconvBlock(channels[-1], output_chans,
    #                                 upsample=upsample_i,
    #                                 layer_fn=layer_fn_i,
    #                                 normalization_str=normalization_str,
    #                                 activation_str=activation_str))

    # Add normalization to the final layer if requested
    if norm_last_layer:
        final_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, output_chans)}
        layers.append(norm_fn(nfeatures=input_chans, **final_gn_groups))

    return nn.Sequential(*layers)


def _build_conv_stack(input_chans, output_chans,
                      layer_fn,
                      base_channels,
                      channel_multiplier,
                      kernels,
                      strides,
                      activation_str="relu",
                      normalization_str="none",
                      norm_first_layer=False,
                      norm_last_layer=False,
                      **kwargs):
    """ Helper to build an arbitrary convolutional decoder.

    :param input_chans: number of input channels
    :param output_shape: [C, H, W] output shape
    :param layer_fn: what layer function to use?
    :param base_channels: base feature maps
    :param channel_multiplier: expand by this per layer, usually < 1
    :param kernels: list of kernels per layer
    :param strides: list of strides for each layer
    :param activation_str: what activation function to use
    :param normalization_str: layer normalization type, eg: batchnorm
    :param norm_first_layer: apply normalization to the input layer?
    :param norm_last_layer: apply normalization to the final layer?
    :returns: a model with a bunch of conv layers.
    :rtype: nn.Sequential

    """
    assert len(kernels) == len(strides)

    # Normalization and activation helpers
    norm_fn = functools.partial(
        add_normalization, normalization_str=normalization_str, ndims=2)
    activation_fn = str_to_activ_module(activation_str)
    layers = []

    if norm_first_layer:
        init_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, input_chans)}
        layers.append(norm_fn(Identity(), nfeatures=input_chans, **init_gn_groups))
        # layers.append(activation_fn())  # TODO(jramapuram): consider this.

    # build the channel map.
    channels = [input_chans, int(base_channels)]
    for i in range(len(kernels) - 2):  # -2 because last one is output_shape[1] and first is input_chans
        channels.append(int(channel_multiplier*channels[-1]))

    channels.append(output_chans)

    print('channels = ', channels)
    print('channels = {} | kernels = {} | strides = {}'.format(len(channels), len(kernels), len(strides)))
    print(len(channels[1:]), len(channels[0:-1]))

    # build each individual layer
    # for k, s, chan_in, chan_out in zip(kernels[0:-1], strides[0:-1], channels[0:-1], channels[1:]):
    for idx, (k, s, chan_in, chan_out) in enumerate(zip(kernels, strides, channels[0:-1], channels[1:])):
        is_last_layer = (idx == len(kernels) - 1)
        if is_last_layer:
            normalization_str = 'none' if norm_last_layer is False else normalization_str

        li_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, chan_out)}
        layer_i = norm_fn(layer_fn(chan_in, chan_out, kernel_size=k, stride=s),
                          nfeatures=chan_out, **li_gn_groups)
        layers.append(layer_i)

        if not is_last_layer:
            # TODO(jramapuram): consider adding attention; works only with SNConv2d though
            # layers.append(Attention(chan_out, SNConv2d))
            layers.append(activation_fn())

    # build the final layer
    # normalization_str = 'none' if norm_last_layer is False else normalization_str
    # final_gn_groups = {'num_groups': _compute_group_norm_planes(normalization_str, output_chans)}
    # # layers.append(add_normalization(layer_fn(channels[-2], output_chans, kernel_size=kernels[-1], stride=strides[-1]),
    # layers.append(add_normalization(layer_fn(channels[-1], output_chans, kernel_size=kernels[-1], stride=strides[-1]),
    #                                 normalization_str=normalization_str, ndims=2,
    #                                 nfeatures=output_chans, **final_gn_groups))
    return nn.Sequential(*layers)


def proj(x, y):
    """Projection of x onto y.

       From https://github.com/ajbrock/BigGAN-PyTorch/
    """
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    """Orthogonalize x wrt list of vectors ys.

       From https://github.com/ajbrock/BigGAN-PyTorch/
    """
    for y in ys:
        x = x - proj(x, y)

    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    """Apply num_itrs steps of the power method to estimate top N singular values."""
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)

            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)

            # Add to the list
            vs += [v]

            # Update the other singular vector
            u = torch.matmul(v, W.t())

            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)

            # Add to the list
            us += [u]

            if update:
                u_[i][:] = u

        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]

    return svs, us, vs


class SN(object):
    """Spectral normalization base class.

       From https://github.com/ajbrock/BigGAN-PyTorch/
    """

    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs

        # Number of singular values
        self.num_svs = num_svs

        # Transposed?
        self.transpose = transpose

        # Epsilon value for avoiding divide-by-0
        self.eps = eps

        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        """Singular vectors (u side)."""
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        """Singular values;note that these buffers are just for logging and are not used in training."""
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        """Compute the spectrally-normalized weight."""
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()

        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)

        # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv

        return self.weight / svs[0]


class SNConv2d(nn.Conv2d, SN):
    """2D Conv layer with spectral norm,


       From https://github.com/ajbrock/BigGAN-PyTorch/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Attention(nn.Module):
    def __init__(self, ch, conv_fn=SNConv2d):
        """Attention from SAGAN with modification from BigGAN.

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
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)

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
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=UpsampleConvLayer):
        super(Conv32UpsampleDecoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder normalization
        norm_layer = Identity()
        if norm_first_layer:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_chans]),
            norm_layer,
            add_normalization(nn.Linear(input_chans, input_chans*4*4),
                              normalization_str, ndims=1, nfeatures=input_chans*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_chans, 4, 4]),
        )

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
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


class Resnet32Decoder(nn.Module):
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet32Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder normalization
        norm_layer = Identity()
        if norm_first_layer:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_chans]),
            norm_layer,
            add_normalization(nn.Linear(input_chans, input_chans*4*4),
                              normalization_str, ndims=1, nfeatures=input_chans*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_chans, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3],
                                         strides=[1, 1, 1],
                                         upsample=[True, True, True],
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


class Resnet64Decoder(nn.Module):
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet64Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder normalization
        norm_layer = Identity()
        if norm_first_layer:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_chans]),
            norm_layer,
            add_normalization(nn.Linear(input_chans, input_chans*4*4),
                              normalization_str, ndims=1, nfeatures=input_chans*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_chans, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3, 3],
                                         strides=[1, 1, 1, 1],
                                         upsample=[True, True, True, True],
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


class Resnet128Decoder(nn.Module):
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.Conv2d):
        super(Resnet128Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # Handle pre-decoder normalization
        norm_layer = Identity()
        if norm_first_layer:
            norm_layer = add_normalization(norm_layer, normalization_str,
                                           ndims=1, nfeatures=input_chans)

        # Project to 4x4 first
        self.mlp_proj = nn.Sequential(
            View([-1, input_chans]),
            norm_layer,
            add_normalization(nn.Linear(input_chans, input_chans*4*4),
                              normalization_str, ndims=1, nfeatures=input_chans*4*4),
            str_to_activ_module(activation_str)(),
            View([-1, input_chans, 4, 4]),
        )

        # The main model
        self.model = _build_resnet_stack(input_chans=input_chans,
                                         output_chans=output_chans,
                                         layer_fn=layer_fn,
                                         base_channels=base_channels,
                                         channel_multiplier=channel_multiplier,
                                         kernels=[3, 3, 3, 3, 3],
                                         strides=[1, 1, 1, 1, 1],
                                         upsample=[True, True, True, True, True],
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


class Conv32Decoder(nn.Module):
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=False,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv32Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
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


class Conv64Decoder(nn.Module):
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv64Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
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
    def __init__(self, input_chans, output_chans, base_channels=1024, channel_multiplier=0.5,
                 activation_str="relu", normalization_str="none", norm_first_layer=True,
                 norm_last_layer=False, layer_fn=nn.ConvTranspose2d):
        super(Conv128Decoder, self).__init__()
        assert isinstance(input_chans, (float, int)), "Expect input_size as float or int."

        # The main model
        self.model = _build_conv_stack(input_chans=input_chans,
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


def _build_dense(input_shape, output_shape, latent_size=512, num_layers=2,
                 activation_fn=nn.SELU, normalization_str="none",
                 layer=nn.Linear, **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_shape))
    output_shape = [output_shape] if not isinstance(output_shape, list) else output_shape

    # don't flatten if we passed 'disable_flatten' in kwargs
    if kwargs.get('disable_flatten', False) and input_flat == input_shape:
        view_init_layer = Identity()
        view_final_layer = Identity()
    else:
        view_init_layer = View([-1, input_flat])
        view_final_layer = View([-1] + output_shape)

    layers = [('view0', view_init_layer),
              ('l0', add_normalization(layer(input_flat, latent_size),
                                       normalization_str, 1, latent_size, num_groups=32)),
              ('act0', activation_fn())]

    for i in range(num_layers - 2):  # 2 for init layer[above] + final layer[below]
        layers.append(
            ('l{}'.format(i+1), add_normalization(layer(latent_size, latent_size),
                                                  normalization_str, 1, latent_size, num_groups=32))
        )
        layers.append(('act{}'.format(i+1), activation_fn()))

    layers.append(('output', layer(latent_size, output_flat)))
    layers.append(('viewout', view_final_layer))

    return nn.Sequential(OrderedDict(layers))


def build_dense_encoder(input_shape, output_size, latent_size=512, num_layers=2,
                        activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
    return _build_dense(input_shape, output_size, latent_size, num_layers,
                        activation_fn, normalization_str, layer=nn.Linear, **kwargs)


def build_gated_dense_encoder(input_shape, output_size, latent_size=512, num_layers=2,
                              activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view '''
    return _build_dense(input_shape, output_size, latent_size, num_layers,
                        activation_fn, normalization_str, layer=GatedDense, **kwargs)


def build_gated_dense_decoder(input_size, output_shape, latent_size=512, num_layers=2,
                              activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' accecpts a flattened vector (input_size) and outputs output_shape '''
    return _build_dense(input_size, output_shape, latent_size, num_layers,
                        activation_fn, normalization_str, layer=GatedDense, **kwargs)


def build_dense_decoder(input_size, output_shape, latent_size=512, num_layers=3,
                        activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' accepts a flattened vector (input_size) and outputs output_shape '''
    return _build_dense(input_size, output_shape, latent_size, num_layers,
                        activation_fn, normalization_str, layer=nn.Linear, **kwargs)


def get_encoder(input_shape: Tuple[int, int, int],  # [C, H, W]
                encoder_layer_type: str = 'conv',
                encoder_base_channels: int = 32,  # For conv models
                encoder_channel_multiplier: int = 2,
                latent_size: int = 512,   # For dense models
                dense_normalization: str = 'none',
                conv_normalization: str = 'none',
                disable_gated: bool = True,
                norm_last_layer: bool = False,
                activation: str = 'relu',
                name: str = 'encoder',
                **unused_kwargs):
    '''Helper to return the correct encoder function.'''
    conv_encoder_size_dict = {
        128: Conv128Encoder,
        64: Conv64Encoder,
        32: Conv32Encoder,
        # TODO(jramapuram): add other sizing here.
    }
    chans, image_size = input_shape[0], input_shape[-1]

    net_map = {
        # 'resnet': {
        #     # True for gated, False for non-gated
        #     True: functools.partial(_build_resnet_encoder,
        #                             layer_type=ResnetBlock.gated_conv3x3,
        #                             filter_depth=config['filter_depth'],
        #                             num_layers=num_layers,
        #                             bilinear_size=(determined_size, determined_size),
        #                             normalization_str=config['conv_normalization']),
        #     False: functools.partial(_build_resnet_encoder,
        #                              layer_type=ResnetBlock.conv3x3,
        #                              filter_depth=config['filter_depth'],
        #                              num_layers=num_layers,
        #                              bilinear_size=(determined_size, determined_size),
        #                              normalization_str=config['conv_normalization'])
        # },
        'conv': {
            True: functools.partial(conv_encoder_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=GatedConv2d),
            False: functools.partial(conv_encoder_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=nn.Conv2d),
        },
        'batch_conv': {
            True: functools.partial(conv_encoder_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=BatchConv2D)),
            False: functools.partial(conv_encoder_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=BatchConv2D),
        },
        'coordconv': {
            True: functools.partial(conv_encoder_size_dict[image_size],
                                    input_chans=chans,
                                    base_channels=encoder_base_channels,
                                    channel_multiplier=encoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=CoordConv)),
            False: functools.partial(conv_encoder_size_dict[image_size],
                                     input_chans=chans,
                                     base_channels=encoder_base_channels,
                                     channel_multiplier=encoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation,
                                     layer_fn=CoordConv),
        },
        # 'dense': {
        #     # True for gated, False for non-gated
        #     True: functools.partial(build_gated_dense_encoder,
        #                             latent_size=latent_size,
        #                             num_layers=3,  # XXX(jramapuram): hardcoded
        #                             bilinear_size=(determined_size, determined_size),
        #                             normalization_str=dense_normalization),
        #     False: functools.partial(build_dense_encoder,
        #                              latent_size=latent_size,
        #                              num_layers=3,
        #                              bilinear_size=(determined_size, determined_size),
        #                              normalization_str=dense_normalization)
        # }
    }

    fn = net_map[encoder_layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        encoder_layer_type,
        name
    ))
    return fn


def get_decoder(input_chans: int,                    # input size to decoder
                output_shape: Tuple[int, int, int],  # output image shape [B, H, W]
                decoder_layer_type: str = 'conv',
                decoder_base_channels: int = 1024,      # For conv models
                decoder_channel_multiplier: int = 0.5,  # Decoding shrinks channels
                latent_size: int = 512,         # For dense models
                dense_normalization: str = 'none',
                conv_normalization: str = 'none',
                disable_gated: bool = True,
                norm_first_layer: bool = False,
                norm_last_layer: bool = False,
                activation: str = 'relu',
                name: str = 'decoder',
                **unused_kwargs):
    '''Helper to return the correct decoder function.'''
    conv_decoder_size_dict = {
        128: Conv128Decoder,
        64: Conv64Decoder,
        32: Conv32Decoder,
        # 32: Conv32UpsampleDecoder,
        # TODO(jramapuram): add other sizing here.
    }
    resnet_size_dict = {
        128: Resnet128Decoder,
        64: Resnet64Decoder,
        32: Resnet32Decoder,
    }

    image_size = output_shape[-1]

    net_map = {
        # 'resnet': {
        #     # True for gated, False for non-gated
        #     True: functools.partial(_build_resnet_decoder,
        #                             layer_type=ResnetDeconvBlock.gated_deconv3x3,
        #                             filter_depth=config['filter_depth'],
        #                             num_layers=num_layers,
        #                             normalization_str=config['conv_normalization']),
        #     False: functools.partial(_build_resnet_decoder,
        #                              layer_type=ResnetDeconvBlock.deconv3x3,
        #                              filter_depth=config['filter_depth'],
        #                              num_layers=num_layers,
        #                              normalization_str=config['conv_normalization'])
        # },
        # 'dense': {
        #     # True for gated, False for non-gated
        #     True: functools.partial(build_gated_dense_decoder,
        #                             latent_size=config['latent_size'],
        #                             num_layers=num_layers,
        #                             normalization_str=config['dense_normalization']),
        #     False: functools.partial(build_dense_decoder,
        #                              latent_size=config['latent_size'],
        #                              num_layers=num_layers,
        #                              normalization_str=config['dense_normalization'])
        # }
        'resnet': {
            # True for gated, False for non-gated
            True: functools.partial(resnet_size_dict[image_size],
                                    input_chans=input_chans,
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=nn.ConvTranspose2d)),
            False: functools.partial(resnet_size_dict[image_size],
                                     input_chans=input_chans,
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
            True: functools.partial(conv_decoder_size_dict[image_size],
                                    input_chans=input_chans,
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=nn.ConvTranspose2d)),
            False: functools.partial(conv_decoder_size_dict[image_size],
                                     input_chans=input_chans,
                                     output_chans=output_shape[0],
                                     base_channels=decoder_base_channels,
                                     channel_multiplier=decoder_channel_multiplier,
                                     normalization_str=conv_normalization,
                                     norm_first_layer=norm_first_layer,
                                     norm_last_layer=norm_last_layer,
                                     activation_str=activation)
                                     # layer_fn=nn.ConvTranspose2d),
        },
        'batch_conv': {
            True: functools.partial(conv_decoder_size_dict[image_size],
                                    input_chans=input_chans,
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=BatchConvTranspose2D)),
            False: functools.partial(conv_decoder_size_dict[image_size],
                                     input_chans=input_chans,
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
            True: functools.partial(conv_decoder_size_dict[image_size],
                                    input_chans=input_chans,
                                    output_chans=output_shape[0],
                                    base_channels=decoder_base_channels,
                                    channel_multiplier=decoder_channel_multiplier,
                                    normalization_str=conv_normalization,
                                    norm_first_layer=norm_first_layer,
                                    norm_last_layer=norm_last_layer,
                                    activation_str=activation,
                                    layer_fn=functools.partial(GatedConv2d, layer_type=CoordConvTranspose)),
            False: functools.partial(conv_decoder_size_dict[image_size],
                                     input_chans=input_chans,
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

    # NOTE: pixelcnn is added later, override here
    layer_type = "conv" if decoder_layer_type == "pixelcnn" else decoder_layer_type
    fn = net_map[layer_type][not disable_gated]
    print("using {} {} for {}".format(
        "gated" if not disable_gated else "standard",
        decoder_layer_type,
        name
    ))
    return fn


def append_save_and_load_fns(model, optimizer, grapher, args):
    """ Hax to add save and load functionality to use with early-stopping module.

    :param model: any torch module
    :param optimizer: the optimizer to save
    :param grapher: the visdom or tensorboard object
    :param args: argparse
    :returns: the same module with the added methods
    :rtype: nn.Module

    """
    from .utils import get_name

    def load(model, optimizer, grapher, **kwargs):
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
                print("loading existing model: {}".format(model_filename))

                # load the full dictionary and set the model and optimizer params
                save_dict = torch.load(model_filename)
                model.load_state_dict(save_dict['model'])
                optimizer.load_state_dict(save_dict['optimizer'])
                grapher.state_dict = save_dict['grapher']

                # remove the keys that we used to load the models
                del save_dict['model']
                del save_dict['optimizer']
            else:
                print("{} does not exist...".format(model_filename))

        # restore the visdom grapher
        if 'grapher' in save_dict and save_dict['grapher'] \
           and 'scalars' in save_dict['grapher'] and save_dict['grapher']['scalars']:
            grapher.set_data(save_dict['grapher']['scalars'], save_dict['grapher']['windows'])
            del save_dict['grapher']

        return save_dict

    def save(model, optimizer, grapher, **kwargs):
        """ Saves a model and optimizer to a file.

            Optional params:
                  -  'overwrite' : force over-writes a savefile
                  -  'prefix': prefix the save file with this st
                  -  'epoch': the epoch that were at

        :param model: nn.Module
        :param optimizer: nn.Optim
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
            with tempfile.NamedTemporaryFile() as scalar, tempfile.NamedTemporaryFile() as window:
                grapher.pickle_data(scalar.name, window.name)
                scalar_dict = pickle.load(scalar.file)
                window_dict = pickle.load(window.file)

            # save the entire state
            torch.save(
                {**{
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    'grapher': {'scalars': scalar_dict, 'windows': window_dict}
                }, **kwargs},
                model_filename
            )

    model.load = functools.partial(load, model=model, grapher=grapher, optimizer=optimizer)
    model.save = functools.partial(save, model=model, grapher=grapher, optimizer=optimizer)
    return model
