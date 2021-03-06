# coding: utf-8

import os
import git
import socket
import torch
import torchvision
import torch.nn as nn
import numpy as np
import contextlib
import torch.nn.functional as F

from pathlib import Path
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
    """Given an array [B, *] and a permutation array [B], invert the operation returning array"""
    idx_perm = torch.cat([(perm == i).nonzero() for i in range(len(perm))], 0).squeeze()
    return arr[idx_perm]


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


def git_root_dir(path: str = '.'):
    """Returns the path of the git root in path.

    :param path: string path, anywhere in git root is fine.
    :returns: str git root
    :rtype: str

    """
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def read_files_from_dir_to_dict(source_dir: str, file_extensions=['*.py', '*.sh']):
    """Reads all files from the source dir that match file_extensions and return a dict

    :param source_dir: a string path
    :param file_extensions: the extensions we want to save
    :returns: a dict containing each file-path as key and value the bytes
    :rtype: dict[str, bytes]

    """
    requested_files = flatten([Path(source_dir).rglob(f) for f in file_extensions])
    cleaned_filenames = [str(filename.relative_to(Path(source_dir))) for filename in requested_files]
    return {name: path.open(mode='r').read()
            for name, path in zip(cleaned_filenames, requested_files)}


def restore_files_from_dict_to_dir(source_dict, dest_dir: str, overwrite: bool = False):
    """Restores data from the result of read_files_from_dir_to_dict into dest_dir

    :param source_dict: the source
    :param dest_dir: the destination directory
    :param overwrite: (optional) overwrite existing files if they exist
    :returns: nothing
    :rtype: None

    """
    for k, v in source_dict.items():
        new_file_path = Path(dest_dir) / Path(k)
        new_file_path.parents[0].mkdir(parents=True, exist_ok=True)  # make the containing dir
        if new_file_path.is_file() and overwrite is False:
            raise Exception("File {} exists and overwrite=False specified".format(k))

        new_file_path.open(mode='w').write(v)


def read_from_csv(filename):
    """Simple helper to read a csv."""
    return np.genfromtxt(filename, delimiter=',')


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


def scale(val, newmin, newmax, oldmin, oldmax):
    ''' helper to scale [oldmin, oldmax] --> [newmin, newmax] '''
    return (((val - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin


def l2_normalize(x, dim=None, eps=1e-12):
    """Normalize a tensor over dim using the L2-norm."""
    sq_sum = torch.sum(torch.square(x), dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(torch.max(sq_sum, torch.ones_like(sq_sum)*eps))
    return x * inv_norm


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

    # print("masks gathered = ", masks_gathered.size())

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
    """Creates a TSNE plot given a [B, F] z_mu tensor, a list of corresponding classes and prefix for the file."""
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

    alpha = 2 / (window + 1.0)
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
    """Given two categoricals zero pad the smaller one to make equivalent to the bigger one."""
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
    return 1e-3 if half else 1e-6


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


def number_of_gpus():
    """Returns an int describing availabe GPU devices respecting env vars."""
    env_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env_devices is not None:
        return 1 if ',' not in env_devices else len(env_devices.split(','))

    return torch.cuda.device_count()


def get_ip_address_and_hostname(hostname=None):
    """Simple helper to get the ip address and hostname.

    :param hostname: None here grabs the local machine hostname
    :returns: ip address, hostname
    :rtype: str, str

    """
    hostname = socket.gethostname() if hostname is None else hostname
    ip_addr = socket.gethostbyname(hostname)
    return ip_addr, hostname


def get_slurm_id():
    """Simple helper to get the slurm job id."""
    return os.environ.get('SLURM_JOBID', None)


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


def slerp(val, low, high):
    def _slerp_pytorch():
        """Spherical linear interpolation, from https://bit.ly/3krBVQ7 """
        value = torch.dot(low / torch.norm(low), high / torch.norm(high))
        omega = torch.acos(torch.clamp(value, -1, 1))
        so = torch.sin(omega)
        if so == 0:
            # L'Hopital's rule / LERP
            return (1.0-val) * low + val * high

        return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high

    def _slerp_np():
        """Spherical linear interpolation, from https://bit.ly/3krBVQ7 """
        value = np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high))
        omega = np.arccos(np.clip(value, -1, 1))
        so = np.sin(omega)
        if so == 0:
            # L'Hopital's rule / LERP
            return (1.0-val) * low + val * high

        return np.sin((1.0-val) * omega) / so * low + np.sin(val * omega) / so * high

    return _slerp_np() if not isinstance(val, torch.Tensor) else _slerp_pytorch()


def spherical_interpolate(p1, p2, n_steps=8):
    """Spherical interpolation between two vectors."""
    assert p1.dim() == p2.dim() == 1, "Only 1d vectors are currently supported."

    def _spherical_interpolate_pytorch():
        ratios = torch.linspace(0, 1, steps=n_steps)
        vectors = [slerp(ratio, p1, p2).unsqueeze(0) for ratio in ratios]
        return torch.cat(vectors, 0)

    def _spherical_interpolate_np():
        ratios = np.linspace(0, 1, num=n_steps)
        vectors = [slerp(ratio, p1, p2) for ratio in ratios]
        return np.asarray(vectors)

    return _spherical_interpolate_np() if not isinstance(p1, torch.Tensor) \
        else _spherical_interpolate_pytorch()


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
    if nan_check(tensor, name) is True:
        exit(-1)


def nan_check(tensor, name=""):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        nan_list = torch.cat([torch.sum(nan_check(t, name)).unsqueeze(0) for t in tensor], 0)
        return torch.sum(nan_list) > 0
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
    blacklist_keys = ['visdom_url', 'visdom_port', 'wandb_url', 'wandb_port', 'data_dir', 'log_dir',
                      'download', 'cuda', 'uid', 'debug_step', 'detect_anomalies', 'model_dir',
                      'calculate_fid_with', 'calculate_msssim', 'input_shape', 'fid_model_dir',
                      'output_dir', 'gpu', 'fid_server', 'metrics_server', 'num_train_samples',
                      'num_test_samples', 'num_valid_samples', 'workers_per_replica',
                      'steps_per_train_epoch', 'total_train_steps', 'distributed_master', 'distributed_port',
                      'distributed_rank', 'multi_gpu_distributed', 'slurm_job_id',
                      'num_fixed_point_generation_iterations', 'generative_scale_var']
    filtered = {k: v for k, v in vargs.items() if k not in blacklist_keys}  # remove useless info

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
    def bool2int(v): return int(v) if isinstance(v, bool) else v
    def none2bool(v): return 0 if v is None else v
    def nonestr2bool(v): return 0 if isinstance(v, str) and v.lower().strip() == 'none' else v
    def clip2int(v): return int(v) if isinstance(v, float) and v - int(v) == 0 else v
    filtered = {_factor(k): clip2int(nonestr2bool(none2bool(bool2int(v)))) for k, v in filtered.items()}
    filtered = {k: v for k, v in filtered.items() if v != 0}  # Removed 0-valued entries to save space

    name = _clean_task_str("{}_{}".format(
        args.uid if args.uid else "",
        "_".join(["{}{}".format(k, v) for k, v in filtered.items()])
    ).replace('groupnorm', 'gn')
                           .replace('dmlab_mazes', 'maze')
                           .replace('realnvp', 'rnvp')
                           .replace('maf_split', 'mafsp')
                           .replace('maf_split_glow', 'mafspg')
                           .replace('clamp', 'C')
                           .replace('celeba', 'CA')
                           .replace('l2msssim', 'l2M')
                           .replace('evonorm', 'en')
                           .replace('spectralnorm', 'sn')
                           .replace('class_conditioned', 'ccvae')
                           .replace('sync_batchnorm', 'sbn')
                           .replace('batchnorm', 'bn')
                           .replace('batch_groupnorm', 'bgn')
                           .replace('instancenorm', 'in')
                           .replace('weightnorm', 'wn')
                           .replace('pixel_wise', 'pw')
                           .replace('binarized_mnist', 'bmnist')
                           .replace("binarized_omniglot_burda", "bboglot")
                           .replace('binarized_omniglot', 'boglot')
                           .replace('omniglot', 'oglot')
                           .replace('disc_mix_logistic', 'dml')
                           .replace('log_logistic_256', 'll256')
                           .replace('pixelcnn', 'pcnn')
                           .replace('isotropic_gaussian', 'N')
                           .replace('bernoulli', 'B')
                           .replace('discrete', 'D')
                           .replace('mixture', 'M')
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
                           .replace('additive_vrnn', 'avrnn')
                           .replace('orthogonal', 'o')
                           .replace('gaussian', 'N')
                           .replace('cosine', 'cos')
                           .replace('lars_', 'l')
                           .replace('momentum', 'mom')
                           .replace('autoencoder', 'ae')
                           .replace('nih_chest_xray', 'xray')
                           .replace('dali_image_folder', 'dimfolder')
                           .replace('dali_multi_augment_image_folder', 'dmaimfolder')
                           .replace('crop_dual_imagefolder', 'cdimfolder')
                           .replace('multi_image_folder', 'mimfolder')
                           .replace('multi_augment_image_folder', 'maimfolder')
                           .replace('image_folder', 'imfolder')
                           .replace('celeba_sequential', 'sceleba')
                           .replace('starcraft_predict_battle', 'sc2')
                           .replace('svhn_centered', 'svhn')
    )

    # sanity check to ensure filename is 255 chars or less for being able to write to filesystem
    assert len(name) + len('.json') < np.power(2, 8), "rethink argparse to shorten name: {}".format(name)
    return name


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        if any(np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
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
