import torch
import torch.nn.functional as F
import torch.distributions as D

from .utils import ones_like, eps, is_half
from .pixel_cnn.utils import discretized_mix_logistic_loss, \
    discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d, \
    sample_from_discretized_mix_logistic


def nll_activation(logits, nll_type, **kwargs):
    """ Helper to activate logits based on the NLL
        NOTE: ignores variance here!

    :param logits: the unactivated logits.
    :param nll_type: the string type of the likelihood
    :returns: the **MEAN** representation of the likelihood; ignored variance. dim(tensor) == dim(logits) except PCNN.
    :rtype: torch.Tensor

    """
    if nll_type == "disc_mix_logistic":
        assert 'chans' in kwargs, "need channels for disc_mix_logistic"
        fn = sample_from_discretized_mix_logistic_1d if kwargs['chans'] == 1 \
            else sample_from_discretized_mix_logistic
        # ideally do this, but needs parameterization
        # return fn(logits, **kwargs)
        return fn(logits, nr_mix=10)
    elif nll_type == 'log_logistic_256':
        assert len(logits.shape) == 4, "need 4-dim tensor for log-logistic-256"
        assert 'chans' in kwargs, "need channels for disc_mix_logistic"

        if kwargs['chans'] == 1:
            return torch.sigmoid(logits)
        else:
            logits_mu, logits_sigma = get_loc_and_scale(logits)
            #return torch.bernoulli(F.sigmoid(logits_mu))
            return torch.sigmoid(logits_mu)
    elif nll_type == 'pixel_wise':
        return pixel_wise_activation(logits)
    elif nll_type == "gaussian":
        logits_mu, logits_sigma = get_loc_and_scale(logits)
        #return D.Normal(loc=logits_mu, scale=logits_sigma).sample()
        return logits_mu
    elif nll_type == "laplace":
        logits_mu, logits_sigma = get_loc_and_scale(logits)
        return D.Laplace(loc=logits_mu, scale=logits_sigma).sample()
    elif nll_type == "bernoulli":
        return torch.sigmoid(logits)
    else:
        raise Exception("unknown nll provided")


def pixel_wise_activation(logits):
    """ Helper to take [B, 256*C, _, _] and returns [B, C, _, _]

    :param logits: the logit unactivated tensor
    :returns: a batch of images, [B, C, _, _]
    :rtype: torch.Tensor

    """
    # sanity checks
    assert logits.shape[1] % 256 == 0, "tensor needs to be mod 256 for pixewise activation"
    assert logits.dim() == 4, "need 4d image for pixelwise activation"

    chans = logits.shape[1]
    # activ = torch.cat([torch.argmax(F.softmax(logits[:, begin_c:end_c, :, :], dim=1), dim=1).unsqueeze(1)
    #                    for begin_c, end_c in zip(range(0, chans, 256),
    #                                              range(256, chans+1 , 256))], dim=1)
    activ = torch.cat([torch.argmax(logits[:, begin_c:end_c, :, :], dim=1).unsqueeze(1)
                       for begin_c, end_c in zip(range(0, chans, 256),
                                                 range(256, chans+1 , 256))], dim=1)
    return activ.type(torch.float32) / 255.0


def nll_has_variance(nll_str):
    """ A simple helper to return whether we have variance in the likelihood.

    :param nll_str: the type of negative log-likelihood.
    :returns: true or false
    :rtype: bool

    """
    nll_map = {
        'gaussian': True,
        'laplace': True,
        'pixel_wise': False,
        'bernoulli': False,
        'log_logistic_256': True,
        'disc_mix_logistic': True
    }

    assert nll_str in nll_map
    return nll_map[nll_str]


def nll(x, recon_x, nll_type):
    """ Router helper to get the actual NLL evaluation.

    :param x: input tenso
    :param recon_x: reconstrubion logits
    :param nll_type: string type
    :returns: tensor of batch_size of the NLL loss
    :rtype: torch.Tensor

    """
    nll_map = {
        "gaussian": nll_gaussian,
        "bernoulli": nll_bernoulli,
        "laplace": nll_laplace,
        "pixel_wise": nll_pixel_wise,
        "log_logistic_256": nll_log_logistic_256,
        "disc_mix_logistic": nll_disc_mix_logistic
    }
    return nll_map[nll_type](x.contiguous(), recon_x.contiguous())


def nll_bernoulli(x, recon_x_logits):
    """ Negative log-likelihood for bernoulli distribution.

    :param x: input tensor
    :param recon_x_logits: reconstruction logits
    :returns: tensor of batch size for the NLL
    :rtype: torch.Tensor

    """
    batch_size = x.size(0)
    nll = D.Bernoulli(logits=recon_x_logits.view(batch_size, -1)).log_prob(
        x.view(batch_size, -1)
    )
    return -torch.sum(nll, dim=-1)


def pixel_wise_label(x):
    """ Takes an image \in [0, 1], re-scales to 255 and returns the long-tensor

    :param x: input image
    :returns: label tensor
    :rtype: torch.Tensor

    """
    assert x.max() <= 1 and x.min() >= 0, "pixel-wise label generation required x \in [0, 1]"
    labels = (x * 255.0).type(torch.int64)
    # labels = F.one_hot(labels, num_classes=256)
    return labels


def nll_pixel_wise(x, recon_x_logits):
    """ NLL for pixel wise loss. Basically softmax cross-entropy \in [0, 255] per pixel.

    :param x: the target tensor
    :param recon_x_logits: the predictions
    :returns: tensor of batch size for the NLL
    :rtype: torch.Tensor

    """
    batch_size = x.shape[0]
    chans = recon_x_logits.shape[1]
    chan_dims = 3 if chans == 256*3 else 1

    # sanity checks
    assert x.shape[1] == chan_dims, "target tensor needs to be [B, {}, _, _], got {}".format(chan_dims, x.shape)
    assert x.dim() == 4 and recon_x_logits.dim() == 4, "need 4d tensors for nll-pixelwise"
    assert chans % 256 == 0, "reconstruction needs to have chan dim be modulo 256"

    #loss = torch.zeros(x.shape[0], device=x.device) # create a running buffer
    loss = torch.zeros_like(x)
    targets = pixel_wise_label(x)                   # re-scale to make the image its own labels

    for idx, (begin_c, end_c) in enumerate(zip(range(0, recon_x_logits.shape[1], 256),
                                               range(256, recon_x_logits.shape[1]+1 , 256))):
        loss_t = F.cross_entropy(input=recon_x_logits[:, begin_c:end_c, :, :],
                                target=targets[:, idx, :, :], reduction='none')

        # handle the grayscale issue
        if loss_t.dim() < 4:
            loss_t = loss_t.unsqueeze(1)

        loss += loss_t

    loss /= chan_dims # remove channel contribution
    return torch.sum(loss.view(batch_size, -1), -1)



def nll_log_logistic_256(x, recon_x_logits):
    """ Negative log-likelihood for log-logistic 256 activation.
        Modified from jmtomczak's github.

    :param x: the true tensor
    :param recon_x_logits: the reconstruction logits
    :returns: tensor of batch size of the NLL
    :rtype: torch.Tensor

    """
    batch_size = x.shape[0]
    bin_size = 1. / 256.
    assert recon_x_logits.shape[1] % 2 == 0, "need variance for reconstruction"
    half_chans = recon_x_logits.size(1) // 2
    mean = torch.clamp(
        torch.sigmoid(recon_x_logits[:, 0:half_chans, :, :].contiguous().view(batch_size, -1)),
        min=0.+1./512., max=1.-1./512.
    )
    logvar = F.hardtanh(
        recon_x_logits[:, half_chans:, :, :].contiguous().view(batch_size, -1),
        min_val=-4.5, max_val=0.
    )

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x_scaled = (torch.floor(x.view(batch_size, -1) / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x_scaled + bin_size/scale)
    cdf_minus = torch.sigmoid(x_scaled)

    # calculate final log-likelihood for an image
    log_logist_256 = -torch.log(cdf_plus - cdf_minus + 1.e-7)
    return torch.sum(log_logist_256, dim=-1)


def nll_disc_mix_logistic(x, recon):
    """ Discretized Log-Logistic Mixture used in PixelCNN++

    :param x: the true tensor
    :param recon: the reconstruction logits
    :returns: batch size of NLL
    :rtype: torch.Tensor

    """
    assert len(x.shape) == len(recon.shape) == 4, "expecting 4d input for discretized logistic loss"
    fn = discretized_mix_logistic_loss_1d if x.shape[1] == 1 \
        else discretized_mix_logistic_loss
    return fn(x, recon)


def get_loc_and_scale(recon):
    """ Helper to get the location and scale parameters by splitting a tensor.

    :param recon: the tensor to split
    :returns: loc and scale split by either channel or feature
    :rtype: torch.Tensor, torch.Tensor

    """
    if len(recon.shape) == 5:  # for JIT type ops, eg: [10, 4, 6, 28, 28]
        num_half_chans = recon.size(2) // 2
        recon_mu = recon[:, :, 0:num_half_chans, :, :].contiguous()
        recon_sigma = recon[:, :, num_half_chans:, :, :].contiguous()
    elif len(recon.shape) == 4:
        num_half_chans = recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
        recon_sigma = recon[:, num_half_chans:, :, :].contiguous()
    elif len(recon.shape) == 3:
        num_half_chans = recon.size(-1) // 2
        recon_mu = recon[:, :, 0:num_half_chans].contiguous()
        recon_sigma = recon[:, :, num_half_chans:].contiguous()
    elif len(recon.shape) == 2:
        num_half_feat = recon.size(-1) // 2
        recon_mu = recon[:, 0:num_half_chans].contiguous()
        recon_sigma = recon[:, num_half_chans:].contiguous()
    else:
        raise Exception("unknown dimension for gausian NLL")

    return recon_mu, recon_sigma


def nll_laplace(x, recon):
    """ Negative log-likelihood for laplace distribution

    :param x: the true tensor
    :param recon: the reconstruction logits
    :returns: tensor of batch size NLL
    :rtype: torch.Tensor

    """
    batch_size = x.size(0)
    loc, scale = get_loc_and_scale(recon)

    # compute the NLL based on the flattened features
    nll = D.Laplace(
        loc=loc.view(batch_size, -1),
        # F.hardtanh(scale.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
        scale=scale.view(batch_size, -1) #+ eps(half)
        ).log_prob(x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)


def nll_gaussian(x, recon):
    """ Negative log-likelihood for gaussian distribution

    :param x: true target
    :param recon: the reconstruction logits
    :returns: batch size NLL
    :rtype: torch.Tensor

    """
    batch_size = x.size(0)
    loc, scale = get_loc_and_scale(recon)
    # compute the NLL based on the flattened features

    nll = F.mse_loss(input=x.view(batch_size, -1),
                     target=loc.view(batch_size, -1),
                       reduction='none')
    return torch.sum(nll, -1)


    # nll = D.Normal(loc=loc.view(batch_size, -1),
    #                scale=torch.clamp(scale.view(batch_size, -1), eps(is_half(x)), 0.999) # sigm can be 0 --> -inf -inf = nan
    # ).log_prob(x.view(batch_size, -1))
    # return -torch.sum(nll, dim=-1)
