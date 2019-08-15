# coding: utf-8

import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import scipy as sp
import contextlib
import torch.nn.functional as F
import torch.distributions as D

from itertools import chain
from collections import Counter
from copy import deepcopy
from torch.autograd import Variable


@contextlib.contextmanager
def dummy_context():
    ''' for conditional with statements'''
    yield None


def flatten(input_list):
    """ Flatten a list of lists

    :param input_list: the input list-of-lists
    :returns: flattened single list
    :rtype: list

    """
    return list(chain.from_iterable(input_list))

def pca_reduce(x, num_reduce):
    '''reduced matrix X to num_reduce features'''
    xm = x - torch.mean(x, 1, keepdim=True)
    xc = torch.matmul(xm, torch.transpose(xm, 0, -1))
    u, s, v = torch.svd(xc)
    return torch.matmul(u[:, 0:num_reduce], torch.diag(s[0:num_reduce]))


def pca_smaller(mat1, mat2):
    ''' returns both matrices to have the same #features'''
    c1shp = mat1.size()
    c2shp = mat2.size()

    # reduce appropriately
    if c1shp[1] > c2shp[1]:
        mat1 = pca_reduce(mat1, c2shp[1])
    elif c2shp[1] > c1shp[1]:
        mat2 = pca_reduce(mat2, c1shp[1])

    return [mat1, mat2]


def recurse_print_keys(m):
    for k, v in m.items():
        if isinstance(v, dict):
            return recurse_print_keys(v)

        print(k)


def expand_dims(tensor, dim=0):
    shape = list(tensor.size())
    shape.insert(dim, 1)
    return tensor.view(*shape)


def squeeze_expand_dim(tensor, axis):
    ''' helper to squeeze a multi-dim tensor and then
        unsqueeze the axis dimension if dims < 4'''
    tensor = torch.squeeze(tensor)
    if len(list(tensor.size())) < 4:
        return tensor.unsqueeze(axis)

    return tensor


def inv_perm(arr, perm):
    idx_perm = torch.cat([(perm == i).nonzero() for i in range(len(perm))], 0).squeeze()
    return arr[idx_perm]


def normalize_images(imgs, mu=None, sigma=None, eps=1e-9):
    ''' normalize imgs with provided mu /sigma
        or computes them and returns with the normalized
       images and tabulated mu / sigma'''
    if mu is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            mu = np.asarray(
                [np.mean(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            mu = np.asarray(
                [np.mean(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    if sigma is None:
        if len(imgs.shape) == 4:
            chans = imgs.shape[1]
            sigma = np.asarray(
                [np.std(imgs[:, i, :, :]) for i in range(chans)]
            ).reshape(1, -1, 1, 1)
        elif len(imgs.shape) == 5:  # glimpses
            chans = imgs.shape[2]
            sigma = np.asarray(
                [np.std(imgs[:, :, i, :, :]) for i in range(chans)]
            ).reshape(1, 1, -1, 1, 1)
        else:
            raise Exception("unknown number of dims for normalization")

    return (imgs - mu) / (sigma + eps), [mu, sigma]


def normalize_train_test_images(train_imgs, test_imgs, eps=1e-9):
    ''' simple helper to take train and test images
        and normalize the test images by the train mu/sigma '''
    assert len(train_imgs.shape) == len(test_imgs.shape) >= 4

    train_imgs , [mu, sigma] = normalize_images(train_imgs, eps=eps)
    return [train_imgs,
            (test_imgs - mu) / (sigma + eps)]


def add_noise_to_imgs(imgs, disc_level=256.):
    # x(i) + u with u ∼ U(0, a), where a is determined by the
    # discretization level of the data
    return imgs + uniform(imgs.shape, cuda=imgs.is_cuda,
                          a=0, b=np.log(disc_level)/disc_level,
                          dtype=get_dtype(imgs))


def num_samples_in_loader(data_loader):
    ''' simple helper to return the correct number of samples
        if we have used our class-splitter filter '''
    if hasattr(data_loader, "sampler") \
       and hasattr(data_loader.sampler, "num_samples"):
        num_samples = data_loader.sampler.num_samples
    else:
        num_samples = len(data_loader.dataset)

    return num_samples


def append_to_csv(data, filename):
    with open(filename, 'ab') as f:
        np.savetxt(f, data, delimiter=",")


def is_half(tensor):
    return tensor.dtype == torch.float16


def is_cuda(tensor_or_var):
    return tensor_or_var.is_cuda


def zeros_like(tensor):
    return torch.zeros_like(tensor)

def ones(shape, cuda, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).zero_() + 1


def zeros(shape, cuda, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).zero_()


def normal(shape, cuda, mean=0, sigma=1, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).normal_(mean, sigma)


def discrete_uniform(shape, cuda, a=0, b=2, dtype='float32'):
    ''' NOTE: this generates discrete values up to b - 1'''
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).random_(a, b)


def uniform(shape, cuda, a=0, b=1, dtype='float32'):
    shape = list(shape) if isinstance(shape, tuple) else shape
    return type_map[dtype](cuda)(*shape).uniform_(a, b)


def eye(num_elems, cuda, dtype='float32'):
    return torch.diag(ones([num_elems], cuda=cuda, dtype=dtype))


def ones_like(tensor):
    return torch.ones_like(tensor)


def scale(val, src, dst):
    """Helper to scale val from src range to dst range
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]


def generate_random_categorical(num_targets, batch_size, use_cuda=False):
    ''' Helper to return a categorical of [batch_size, num_targets]
        where one of the num_targets are chosen at random[uniformly]'''
    # indices = scale(torch.rand(batch_size), [0, 1], [0, num_targets])
    indices = long_type(use_cuda)(batch_size, 1).random_(0, to=num_targets)
    return one_hot((batch_size, num_targets), indices, use_cuda=use_cuda)


def merge_masks_into_imgs(imgs, masks_list):
    # (B x C x H x W)
    # masks are always 2d + batch, so expand
    masks_gathered = torch.cat([expand_dims(m, 0) for m in masks_list], 0)
    # masks_gathered = masks_gathered.repeat(1, 1, 3, 1, 1)

    # drop the zeros in the G & B channels and the masks in R
    if masks_gathered.size()[2] < 3:
        zeros = torch.zeros(masks_gathered.size())
        if masks_gathered.is_cuda:
            zeros = zeros.cuda()

        masks_gathered = torch.cat([masks_gathered, zeros, zeros], 2)

    #print("masks gathered = ", masks_gathered.size())

    # add C - channel
    imgs_gathered = expand_dims(imgs, 1) if len(imgs.size()) < 4 else imgs
    if imgs_gathered.size()[1] == 1:
        imgs_gathered = torch.cat([imgs_gathered,
                                   imgs_gathered,
                                   imgs_gathered], 1)

    # tile the images over 0th dimension to make 5d
    # imgs_gathered = imgs_gathered.repeat(masks_gathered.size()[0], 1, 1, 1, 1)
    # super_imposed = imgs_gathered + masks_gathered

    # add all the filters onto the mask
    super_imposed = imgs_gathered
    for mask in masks_gathered:
        super_imposed += mask

    # normalize to one everywhere
    ones = torch.ones(super_imposed.size())
    if masks_gathered.is_cuda:
        ones = ones.cuda()

    super_imposed = torch.min(super_imposed.data, ones)
    return super_imposed


def one_hot_np(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


def one_hot(num_cols, indices, use_cuda=False):
    """ Creates a matrix of one hot vectors.

        - num_cols: int
        - indices: FloatTensor array
    """
    batch_size = indices.size(0)
    mask = long_type(use_cuda)(batch_size, num_cols).fill_(0)
    ones = 1
    if isinstance(indices, Variable):
        ones = Variable(long_type(use_cuda)(indices.size()).fill_(1))
        mask = Variable(mask, volatile=indices.volatile)

    return mask.scatter_(1, indices, ones)


def plot_tensor_grid(batch_tensor, save_filename=None):
    ''' Helper to visualize a batch of images.
        A non-None filename saves instead of doing a show()'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    if save_filename is not None:
        torchvision.utils.save_image(batch_tensor, save_filename, padding=5)
    else:
        plt.show()


def plot_gaussian_tsne(z_mu_tensor, classes_tensor, prefix_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_mu_tensor.data.cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes_tensor.data.cpu().numpy()
    fig666 = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig666.savefig(str(prefix_name)+'_embedding_'+str(ic)+'.png')

    fig666.savefig(str(prefix_name)+'_embedding.png')


def ewma(data, window):
    ''' from https://tinyurl.com/y7ko7n8z '''
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def zero_pad_smaller_cat(cat1, cat2):
    c1shp = cat1.size()
    c2shp = cat2.size()
    cuda = is_cuda(cat1)
    diff = abs(c1shp[1] - c2shp[1])

    # blend in extra zeros appropriately
    if c1shp[1] > c2shp[1]:
        cat2 = torch.cat(
            [cat2,
             Variable(float_type(cuda)(c2shp[0], diff).zero_())],
            dim=-1)
    elif c2shp[1] > c1shp[1]:
        cat1 = torch.cat(
            [cat1,
             Variable(float_type(cuda)(c1shp[0], diff).zero_())],
            dim=-1)

    return [cat1, cat2]


def to_data(tensor_or_var):
    '''simply returns the data'''
    if isinstance(tensor_or_var, Variable):
        return tensor_or_var.data

    return tensor_or_var


def eps(half=False):
    return 1e-2 if half else 1e-6


def get_dtype(tensor):
    ''' returns the type of the tensor as an str'''
    dtype_map = {
        torch.float32: 'float32',
        torch.float16: 'float16',
        torch.double: 'float64',
        torch.float64: 'float64',
        torch.int32: 'int32',
        torch.int64: 'int64',
        torch.long: 'int64'
    }
    return dtype_map[tensor.dtype]


def same_type(half, cuda):
    return half_type(cuda) if half else float_type(cuda)


def double_type(use_cuda):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def float_type(use_cuda):
    return torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def half_type(use_cuda):
    return torch.cuda.HalfTensor if use_cuda else torch.HalfTensor


def pad(tensor_or_var, num_pad, value=0, prepend=False, dim=-1):
    if num_pad == 0:
        return tensor_or_var

    cuda = is_cuda(tensor_or_var)
    pad_val = float_type(cuda)(num_pad).zero_() + value
    if isinstance(tensor_or_var, Variable):
        pad_val = Variable(pad_val)

    if not prepend:
        return torch.cat([tensor_or_var, pad_val], dim=dim)

    return torch.cat([pad_val, tensor_or_var], dim=dim)


def int_type(use_cuda):
    return torch.cuda.IntTensor if use_cuda else torch.IntTensor

def hash_to_size(text, size=-1):
    """ Get a hashed value for the provided text upto size

    :param text: string text
    :param size: number to truncate upto (default: don't truncate)
    :returns: text hashed
    :rtype: str

    """
    import hashlib
    hash_object = hashlib.sha1(str.encode(text))
    hex_dig = hash_object.hexdigest()
    return hex_dig[0:size]

def get_aws_instance_id(timeout=2):
    """ Returns the AWS instance id or None

    :param timeout: seconds to timeout
    :returns: None or str id
    :rtype: str

    """
    # curl --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id
    import requests
    try:
        r = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
    except:
        return None

    return hash_to_size(r.text, 4)

def long_type(use_cuda):
    return torch.cuda.LongTensor if use_cuda else torch.LongTensor


def oneplus(x):
    return F.softplus(x, beta=1)

def number_of_parameters(model, only_required_grad=False):
    if only_required_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_weight_norm(module):
    params = [p[0] for p in module.named_parameters()]
    for param in params:
        if 'weight' in param or 'W_' in param or 'U_' in param:
            print("adding wn to ", param)
            module = torch.nn.utils.weight_norm(
                module, param)

    return module


def check_or_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class ToFP16(nn.Module):
    def __init__(self):
        super(ToFP16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):
    ''' helper to set params (needed for fp16)'''
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))

        param.grad.data.copy_(param_w_grad.grad.data)


def convert_bn_to_float(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()

    for child in module.children():
        convert_bn_to_float(child)

    return module


def network_to_half(network):
    return convert_bn_to_float(network.half())


def nan_check_and_break(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check_and_break(tensor, name))
    else:
        if nan_check(tensor, name) is True:
            exit(-1)


def nan_check(tensor, name=""):
    if isinstance(input, list) or isinstance(input, tuple):
        for tensor in input:
            return(nan_check(tensor, name))
    else:
        if torch.sum(torch.isnan(tensor)) > 0:
            print("Tensor {} with shape {} was NaN.".format(name, tensor.shape))
            return True

        elif torch.sum(torch.isinf(tensor)) > 0:
            print("Tensor {} with shape {} was Inf.".format(name, tensor.shape))
            return True

    return False


def zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() > 0:
        print("tensor {} of {} dim contained ZERO!!".format(name, tensor.shape))
        exit(-1)


def all_zero_check_and_break(tensor, name=""):
    if torch.sum(tensor == 0).item() == np.prod(list(tensor.shape)):
        print("tensor {} of {} dim was all zero".format(name, tensor.shape))
        exit(-1)


def reduce(tensor, dim=None, reduction='sum'):
    """ Helper to reduce a tensor over dim using reduction style.
        Choices are sum, mean or none.

    :param tensor: the tensor to reduce
    :param dim: the dimension to reduce, if None do full reduction
    :param reduction: type of reduction
    :returns: reduced tensor
    :rtype: torch.Tensor or primitive type

    """
    reduction_map = {
        'sum': torch.sum,
        'mean': torch.mean,
        'none': lambda x, dim=None: x
    }
    if dim is None:
        return reduction_map[reduction](tensor)

    return reduction_map[reduction](tensor, dim=dim)


def get_name(args):
    """Takes the argparse and returns kv_kv1_...
        Note that k is refactored as: 'conv_normalization' --> 'cn'
        Also first takes the keys and asserts that there are no dupes

    :param args: argparse
    :returns: a unique name based on args
    :rtype: str

    """

    vargs = deepcopy(vars(args))
    blacklist_keys = ['visdom_url', 'visdom_port', 'data_dir', 'download', 'cuda', 'uid',
                      'debug_step', 'detect_anomalies', 'model_dir', 'calculate_fid_with',
                      'input_shape', 'fid_model_dir']
    filtered = {k:v for k,v in vargs.items() if k not in blacklist_keys} # remove useless info

    def _factor(name):
        ''' returns first characters of strings with _ separations'''
        if '_' not in name:
            return name[0]

        splits = name.split('_')
        return ''.join([s[0] for s in splits])

    # test if there are duplicate keys
    factored_keys = [_factor(k) for k in filtered.keys()]
    dupes_in_factored = set([x for x in factored_keys if factored_keys.count(x) > 1])
    assert len(dupes_in_factored) == 0, \
        "argparse truncation key duplicate detected: {}".format(dupes_in_factored)

    def _clean_task_str(task_str):
        ''' helper to reduce string length.
            eg: mnist+svhn+mnist --> mnist2svhn1 '''
        result_str = ''
        if '+' in task_str:
            splits = Counter(task_str.split('+'))
            for k, v in splits.items():
                result_str += '{}{}'.format(k, v)

            return result_str

        return task_str

    # now filter into the final filter map and return
    bool2int = lambda v: int(v) if isinstance(v, bool) else v
    none2bool = lambda v: 0 if v is None else v
    nonestr2bool = lambda v: 0 if isinstance(v, str) and v.lower().strip() == 'none' else v
    # clip2int = lambda v: int(v) if isinstance(v, (float, np.float32, np.float64)) and v == 0.0 else v
    clip2int = lambda v: int(v) if isinstance(v, float) and v - int(v) == 0 else v
    filtered = {_factor(k):clip2int(nonestr2bool(none2bool(bool2int(v)))) for k,v in filtered.items()}
    name = _clean_task_str("{}_{}".format(
        args.uid if args.uid else "",
        "_".join(["{}{}".format(k, v) for k, v in filtered.items()])
    ).replace('batchnorm', 'bn').replace('batch_groupnorm', 'bgn')
                           .replace('groupnorm', 'gn')
                           .replace('instancenorm', 'in')
                           .replace('weightnorm', 'wn')
                           .replace('binarized_mnist', 'bmnist')
                           .replace("binarized_omniglot_burda", "bboglot")
                           .replace('binarized_omniglot', 'boglot')
                           .replace('omniglot', 'oglot')
                           .replace('disc_mix_logistic', 'dml')
                           .replace('log_logistic_256', 'll256')
                           .replace('pixelcnn', 'pcnn')
                           .replace('isotropic_gaussian', 'ig')
                           .replace('bernoulli', 'bern')
                           .replace('mixture', 'mix')
                           .replace('parallel', 'par')
                           .replace('coordconv', 'cc')
                           .replace('dense', 'd')
                           .replace('batch_conv', 'bc')
                           .replace('conv', 'c')
                           .replace('resnet', 'r')
                           .replace('xavier_normal', 'xn')
                           .replace('xavier_uniform', 'xu')
                           .replace('kaiming_uniform', 'kl')
                           .replace('kaiming_normal', 'kn')
                           .replace('softplus', 'sp')
                           .replace('softmax', 'sm')
                           .replace('identity', 'i')
                           .replace('zeros', 'z')
                           .replace('normal', 'n')
                           .replace('uniform', 'u')
                           .replace('orthogonal', 'o')
    )

    # sanity check to ensure filename is 255 chars or less for being able to write to filesystem
    assert len(name) + len('.json') < np.power(2, 8), "rethink argparse to shorten name: {}".format(name)
    return name


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        #print(module)
        if  any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
            exit(-1)

    model.apply(lambda module: module.register_backward_hook(check_grad))


type_map = {
    'float32': float_type,
    'float64': double_type,
    'double': double_type,
    'half': half_type,
    'float16': half_type,
    'int32': int_type,
    'int64': long_type
}
