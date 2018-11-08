import torch
import warnings
import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from .utils import check_or_create_dir, same_type


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


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


class BWtoRGB(nn.Module):
    def __init__(self):
        super(BWtoRGB, self).__init__()

    def forward(self, x):
        assert len(list(x.size())) == 4
        chans = x.size(1)
        if chans < 3:
            return torch.cat([x, x, x], 1)
        else:
            return x


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
                 stride, padding=0, dilation=1, activation=None, bias=True):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding,
                           dilation=dilation, bias=bias)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding,
                           dilation=dilation, bias=bias)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

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
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g


class GatedConvTranspose2d(nn.Module):
    ''' from jmtomczak's github'''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, activation=None, bias=True):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=bias)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

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
        return ( # LSTM state is (h, c)
            _init_state(),
            _init_state()
        )

    return _init_state()


class EarlyStopping(object):
    def __init__(self, model, max_steps=10, burn_in_interval=None, save_best=True):
        self.max_steps = max_steps
        self.model = model
        self.save_best = save_best
        self.burn_in_interval = burn_in_interval

        self.loss = 0.0
        self.iteration = 0
        self.stopping_step = 0
        self.best_loss = np.inf

    def restore(self):
        self.model.load()

    def __call__(self, loss):
        if self.burn_in_interval is not None and self.iteration < self.burn_in_interval:
            ''' don't save the model until the burn-in-interval has been exceeded'''
            self.iteration += 1
            return False

        # detach if not already detached
        if not isinstance(loss, (float, np.float32, np.float64)):
            loss = loss.clone().detach()

        if (loss < self.best_loss):
            self.stopping_step = 0
            self.best_loss = loss
            if self.save_best:
                self.model.save(overwrite=True)
        else:
            self.stopping_step += 1

        is_early_stop = False
        if self.stopping_step >= self.max_steps:
            print("Early stopping is triggered;  current_loss:{} --> best_loss:{} | iter: {}".format(
                loss, self.best_loss, self.iteration))
            is_early_stop = True

        self.iteration += 1
        return is_early_stop

def flatten_layers(model, base_index=0):
    layers = []
    for l in model.children():
        if isinstance(l, nn.Sequential):
            sub_layers, base_index = flatten_layers(l, base_index)
            layers.extend(sub_layers)
        else:
            layers.append(('layer_%d'%base_index, l))
            base_index += 1

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


class UpsampleConvLayer(torch.nn.Module):
    '''Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/ '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,
                                                    mode='bilinear',
                                                    align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 normalization_str="groupnorm",
                 activation_fn=nn.ReLU, **kwargs):
        super(ResnetBlock, self).__init__()
        layer_type = kwargs['layer_type'] if 'layer_type' in kwargs else ResnetBlock.conv3x3
        self.gn_planes = int(min(np.ceil(planes / 2), 32))
        self.conv1 = add_normalization(layer_type(inplanes, planes, stride),
                                       normalization_str, 2, planes, num_groups=self.gn_planes)
        #self.act = str_to_activ(activation_str)
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
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 normalization_str="groupnorm", activation_fn=nn.ReLU):
        super(ResnetDeconvBlock, self).__init__()
        self.gn_planes = int(min(np.ceil(planes / 2), 32))
        self.conv1 = add_normalization(self.deconv3x3(inplanes, planes, stride),
                                       normalization_str, 2, planes, num_groups=self.gn_planes)
        #self.act = str_to_activ(activation_str)
        self.act = activation_fn()
        self.conv2 = add_normalization(self.deconv3x3(planes, planes),
                                       normalization_str, 2, planes, num_groups=self.gn_planes)
        self.stride = stride
        self.downsample = self.downsampler(inplanes, planes, normalization_str) \
            if downsample is not None else None

    @staticmethod
    def deconv3x3(in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        # nn.ConvTranspose2d(input_size, filter_depth*8, 4, stride=1, bias=True),
        return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=0, bias=False)

    def downsampler(self, inplanes, planes, normalization_str):
        return add_normalization(nn.ConvTranspose2d(inplanes, planes, kernel_size=1,
                                                    stride=self.stride, bias=False),
                                 normalization_str, 2, planes, num_groups=self.gn_planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        print("out = ", out.shape, " | res = ", residual.shape, " | x = ", x.shape)
        out += residual
        out = self.act(out)

        return out


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

    return  nn.AvgPool2d(kernel_size=(kernel_height, kernel_width),
                         stride=stride, padding=padding)


def build_relational_conv_encoder(input_shape, filter_depth=32,
                                  activation_fn=nn.ELU, **kwargs):
    bilinear_size = (32, 32) if 'bilinear_size' not in kwargs else kwargs['bilinear_size']
    upsampler = nn.Upsample(size=bilinear_size, mode='bilinear', align_corners=True)
    chans = input_shape[0]
    return nn.Sequential(
        upsampler if input_shape[1:] != bilinear_size else Identity(),
        # input dim: num_channels x 32 x 32
        nn.Conv2d(chans, filter_depth, 5, stride=1, bias=True),
        nn.BatchNorm2d(filter_depth),
        activation_fn(),
        # state dim: 32 x 28 x 28
        nn.Conv2d(filter_depth, filter_depth*2, 4, stride=2, bias=True),
        nn.BatchNorm2d(filter_depth*2),
        activation_fn(),
        # state dim: 64 x 13 x 13
        nn.Conv2d(filter_depth*2, filter_depth*4, 4, stride=1, bias=True),
        nn.BatchNorm2d(filter_depth*4),
        activation_fn(),
        # state dim: 128 x 10 x 10
        nn.Conv2d(filter_depth*4, filter_depth*8, 4, stride=2, bias=True)
        # state dim: 256 x 4 x 4
    )



def build_pixelcnn_decoder(input_size, output_shape, filter_depth=32,
                           activation_fn=nn.ELU, normalization_str="none", **kwargs):
    ''' modified from jmtomczak's github, do not use, use submodule pixelcnn '''
    bilinear_size = (32, 32) if 'bilinear_size' not in kwargs else kwargs['bilinear_size']
    warnings.warn("use pixelcnn from helpers submodule instead, this is not tested")
    chans = output_shape[0]
    act = nn.SELU(True)  # nn.ReLU(True)
    return nn.Sequential(
        add_normalization(MaskedConv2d('A', input_size, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        add_normalization(MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
                          normalization_str, 2, 64, num_groups=32), act,
        nn.Conv2d(64, chans, 1, 1, 0, dilation=1, bias=True)
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
        'none': {
            1: lambda nfeatures, **kwargs: Identity(),
            2: lambda nfeatures, **kwargs: Identity(),
            3: lambda nfeatures, **kwargs: Identity()
        }
    }

    if normalization_str == 'groupnorm':
        assert 'num_groups' in kwargs, "need to specify groups for GN"
        assert ndims > 1, "group norm needs channels to operate"

    if normalization_str == 'weightnorm':
        return norm_map[normalization_str][ndims](nfeatures, **kwargs)

    return nn.Sequential(module, norm_map[normalization_str][ndims](nfeatures, **kwargs))


def build_gated_conv_encoder(input_shape, output_size, filter_depth=32,
                             activation_fn=Identity, num_layers=4,
                             normalization_str="none", **kwargs):
    return _build_conv_encoder(input_shape=input_shape, output_size=output_size, layer_type=GatedConv2d,
                               filter_depth=filter_depth, activation_fn=activation_fn,
                               num_layers=num_layers, normalization_str=normalization_str, **kwargs)

def build_conv_encoder(input_shape, output_size, filter_depth=32,
                       activation_fn=nn.SELU, num_layers=4,
                       normalization_str="none", **kwargs):
    return _build_conv_encoder(input_shape=input_shape, output_size=output_size, layer_type=nn.Conv2d,
                               filter_depth=filter_depth, activation_fn=activation_fn,
                               num_layers=num_layers, normalization_str=normalization_str, **kwargs)


def build_resnet_encoder(input_shape, output_size, filter_depth=32,
                         activation_fn=nn.SELU, num_layers=None,
                         normalization_str="none", **kwargs):
    warnings.warn("resnet encoder does not currently use num_layers")
    chans = input_shape[0]
    bilinear_size = kwargs['bilinear_size'] if 'bilinear_size' in kwargs else (64, 64)
    return nn.Sequential(
        nn.Upsample(size=bilinear_size, mode='bilinear', align_corners=True),
        ResnetBlock(chans, filter_depth, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        ResnetBlock(filter_depth, filter_depth*2, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        ResnetBlock(filter_depth*2, filter_depth*4, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        ResnetBlock(filter_depth*4, filter_depth*2, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        ResnetBlock(filter_depth*2, filter_depth, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        ResnetBlock(filter_depth, output_size, stride=2, normalization_str=normalization_str, downsample=True, activation_fn=activation_fn, **kwargs),
        nn.Conv2d(output_size, output_size, kernel_size=1, stride=1),
        Squeeze()
    )


def _build_conv_encoder(input_shape, output_size, layer_type=nn.Conv2d,
                        filter_depth=32, activation_fn=Identity, num_layers=4,
                        normalization_str="none", **kwargs):
    bilinear_size = (32, 32) if 'bilinear_size' not in kwargs else kwargs['bilinear_size']
    upsampler = nn.Upsample(size=bilinear_size, mode='bilinear', align_corners=True)
    chans = input_shape[0]

    def _make_layer(input_size, filter_depth, kernel_size=4, stride=1):
        gn_groups = max(int(min(np.ceil(filter_depth / 2), 32)), 1)
        return nn.Sequential(
            add_normalization(layer_type(input_size, filter_depth, kernel_size, stride),
                              normalization_str, 2, filter_depth, num_groups=gn_groups),
            activation_fn()
        )

    # build the second to N-1 layers
    strides = [2 if i % 2 == 0 else 1 for i in range(num_layers)]
    intermediary_layers = [
        _make_layer(filter_depth*(2**i), filter_depth*(2**j), stride=s) for i, j, s in zip(range(0, num_layers),
                                                                                           range(1, num_layers+1),
                                                                                           strides)
    ]

    num_init_groups = max(int(min(np.ceil(filter_depth / 2), 32)), 1)
    num_final_groups = max(int(min(np.ceil(filter_depth*(2**num_layers) / 2), 32)), 1)
    return nn.Sequential(
        upsampler if input_shape[1:] != bilinear_size else Identity(),
        # input dim: num_channels x 32 x 32
        add_normalization(layer_type(chans, filter_depth, 5, stride=1),
                          normalization_str, 2, filter_depth, num_groups=num_init_groups),
        activation_fn(),

        # add dynamic intermediary layers
        *intermediary_layers,

        # state dim: 512 x 1 x 1
        add_normalization(layer_type(filter_depth*(2**num_layers), filter_depth*(2**num_layers), 1, stride=1),
                          normalization_str, 2, filter_depth*(2**num_layers), num_groups=num_final_groups),
        activation_fn(),
        # state dim: 512 x 1 x 1
        layer_type(filter_depth*(2**num_layers), output_size, 1, stride=1),
        # output dim: opt.z_dim x 1 x 1
        Squeeze()
    )

def _build_conv_decoder(input_size, output_shape, layer_type=nn.ConvTranspose2d,
                        filter_depth=32, activation_fn=nn.SELU, num_layers=3,
                        normalization_str='none', reupsample=True, **kwargs):
    '''Conv/FC --> BN --> Activ --> Dropout'''
    bilinear_size = (32, 32) if 'bilinear_size' not in kwargs else kwargs['bilinear_size']
    chans = output_shape[0]
    upsampler = nn.Upsample(size=output_shape[1:], mode='bilinear', align_corners=True)

    def _make_layer(input_size, filter_depth, kernel_size=4, stride=1):
        gn_groups = max(int(min(np.ceil(filter_depth / 2), 32)), 1)
        return nn.Sequential(
            add_normalization(layer_type(input_size, filter_depth, kernel_size, stride),
                              normalization_str, 2, filter_depth, num_groups=gn_groups),
            activation_fn()
        )

    # build the second to N-1 layers
    strides = [2 if i % 2 == 0 else 1 for i in range(num_layers)]
    intermediary_layers = [
        _make_layer(filter_depth*(2**i), filter_depth*(2**j), stride=s) for i, j, s in zip(range(num_layers, -1, -1),
                                                                                           range(num_layers-1, -1, -1),
                                                                                           strides)
    ]

    num_init_groups = max(int(min(np.ceil(filter_depth*(2**num_layers) / 2), 32)), 1)
    num_final_groups = max(int(min(np.ceil(filter_depth / 2), 32)), 1)
    return nn.Sequential(
        View([-1, input_size, 1, 1]),
        # input dim: z_dim x 1 x 1
        add_normalization(layer_type(input_size, filter_depth*(2**num_layers), 4, stride=1, bias=True),
                          normalization_str, 2, filter_depth*(2**num_layers), num_groups=num_init_groups),
        activation_fn(),

        # add intermediary layers
        *intermediary_layers,

        # state dim: 32 x 28 x 28
        add_normalization(layer_type(filter_depth, filter_depth, 5, stride=1),
                          normalization_str, 2, filter_depth, num_groups=num_final_groups),
        activation_fn(),

        # state dim: 32 x 32 x 32
        nn.Conv2d(filter_depth, chans, 1, stride=1, bias=True),
        # output dim: num_channels x 32 x 32
        upsampler if output_shape[1:] != bilinear_size and reupsample else Identity()
    )


def build_gated_conv_decoder(input_size, output_shape, filter_depth=32,
                             activation_fn=Identity, num_layers=3,
                             normalization_str="none", reupsample=True, **kwargs):
    ''' helper that calls the builder function with GatedConvTranspose2d'''
    return _build_conv_decoder(input_size=input_size, output_shape=output_shape,
                               filter_depth=filter_depth, activation_fn=activation_fn,
                               num_layers=num_layers, normalization_str=normalization_str,
                               reupsample=reupsample, layer_type=GatedConvTranspose2d, **kwargs)


def build_conv_decoder(input_size, output_shape, filter_depth=32,
                       activation_fn=nn.SELU, num_layers=3, normalization_str='none',
                       reupsample=True, **kwargs):
    ''' helper that calls the builder function with nn.ConvTranspose2d'''
    return _build_conv_decoder(input_size=input_size, output_shape=output_shape,
                               filter_depth=filter_depth, activation_fn=activation_fn,
                               num_layers=num_layers, normalization_str=normalization_str,
                               reupsample=reupsample, layer_type=nn.ConvTranspose2d, **kwargs)


def _build_dense(input_shape, output_shape, latent_size=512, num_layers=2,
                 activation_fn=nn.SELU, normalization_str="none",
                 layer=nn.Linear, **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
    input_flat = int(np.prod(input_shape))
    output_flat = int(np.prod(output_shape))
    output_shape = [output_shape] if not isinstance(output_shape, list) else output_shape

    layers = [('view0', View([-1, input_flat])),
              ('l0', add_normalization(layer(input_flat, latent_size),
                                       normalization_str, 1, latent_size, num_groups=32)),
              ('act0', activation_fn())]

    for i in range(num_layers - 2): # 2 for init layer[above] + final layer[below]
        layers.append(
            ('l{}'.format(i+1), add_normalization(layer(latent_size, latent_size),
                                                  normalization_str, 1, latent_size, num_groups=32))
        )
        layers.append(('act{}'.format(i+1), activation_fn()))

    layers.append(('output', layer(latent_size, output_flat)))
    layers.append(('viewout', View([-1] + output_shape)))

    return nn.Sequential(OrderedDict(layers))


def build_dense_encoder(input_shape, output_size, latent_size=512, num_layers=2,
                         activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
    return _build_dense(input_shape, output_size, latent_size, num_layers,
                        activation_fn, normalization_str, layer=nn.Linear, **kwargs)

def build_gated_dense_encoder(input_shape, output_size, latent_size=512, num_layers=2,
                              activation_fn=nn.SELU, normalization_str="none", **kwargs):
    ''' flatten --> layer + norm --> activation -->... --> Linear output --> view'''
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


def get_encoder(config, name='encoder'):
    ''' helper to return the correct encoder function from argparse'''
    net_map = {
        'relational': build_relational_conv_encoder,
        'resnet': {
            # True for gated, False for non-gated
            True: functools.partial(build_resnet_encoder,
                                    layer_type=ResnetBlock.gated_conv3x3,
                                    filter_depth=config['filter_depth'],
                                    normalization_str=config['conv_normalization']),
            False: functools.partial(build_resnet_encoder,
                                     layer_type=ResnetBlock.conv3x3,
                                     filter_depth=config['filter_depth'],
                                     normalization_str=config['conv_normalization'])
        },
        'conv': {
            # True for gated, False for non-gated
            True: functools.partial(build_gated_conv_encoder,
                                    filter_depth=config['filter_depth'],
                                    normalization_str=config['conv_normalization']),
            False: functools.partial(build_conv_encoder,
                                     filter_depth=config['filter_depth'],
                                     normalization_str=config['conv_normalization'])
        },
        'dense': {
            # True for gated, False for non-gated
            True: functools.partial(build_gated_dense_encoder,
                                    normalization_str=config['dense_normalization']),
            False: functools.partial(build_dense_encoder,
                                     normalization_str=config['dense_normalization'])

        }
    }

    fn = net_map[config['encoder_layer_type']][not config['disable_gated']]
    print("using {} {} for {}".format(
        "gated" if not config['disable_gated'] else "standard",
        config['encoder_layer_type'],
        name
    ))
    return fn


def get_decoder(config, reupsample=True, name='decoder'):
    ''' helper to return the correct decoder function'''
    net_map = {
        'conv': {
            # True for gated, False for non-gated
            True: functools.partial(build_gated_conv_decoder,
                                    filter_depth=config['filter_depth'],
                                    reupsample=reupsample,
                                    normalization_str=config['conv_normalization']),
            False: functools.partial(build_conv_decoder,
                                     filter_depth=config['filter_depth'],
                                     reupsample=reupsample,
                                     normalization_str=config['conv_normalization'])
        },
        'dense': {
            # True for gated, False for non-gated
            True: functools.partial(build_gated_dense_decoder,
                                    normalization_str=config['dense_normalization']),
            False: functools.partial(build_dense_decoder,
                                     normalization_str=config['dense_normalization'])
        }
    }

    # NOTE: pixelcnn is added later
    layer_type = "conv" if config['decoder_layer_type'] == "pixelcnn" \
       or config['decoder_layer_type'] == "conv" else "dense"
    fn = net_map[layer_type][not config['disable_gated']]
    print("using {} {} for {}".format(
        "gated" if not config['disable_gated'] else "standard",
        config['decoder_layer_type'],
        name
    ))
    return fn
