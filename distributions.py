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
            num_half_chans = logits.size(1) // 2
            logits_mu = logits[:, 0:num_half_chans, :, :]
            # logits_sigma = logits[:, num_half_chans:, :, :]
            #return torch.bernoulli(F.sigmoid(logits_mu))
            return torch.sigmoid(logits_mu)
    elif nll_type == "gaussian" or nll_type == "laplace":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        # return torch.sigmoid(logits_mu)
        return logits_mu
    elif nll_type == "bernoulli":
        return torch.sigmoid(logits)
    else:
        raise Exception("unknown nll provided")


def nll_has_variance(nll_str):
    """ A simple helper to return whether we have variance in the likelihood.

    :param nll_str: the type of negative log-likelihood.
    :returns: true or false
    :rtype: bool

    """
    nll_map = {
        'gaussian': True,
        'laplace': True,
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


def nll_laplace(x, recon):
    """ Negative log-likelihood for laplace distribution

    :param x: the true tensor
    :param recon: the reconstruction logits
    :returns: tensor of batch size NLL
    :rtype: torch.Tensor

    """
    if len(recon.shape) == 4:
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
        recon_logvar = recon[:, num_half_chans:, :, :].contiguous()

        nll = D.Laplace(
            # recon_mu.view(batch_size, -1),
            recon_mu.view(batch_size, -1),
            # F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
            recon_logvar.view(batch_size, -1) + eps(is_half(recon))
        ).log_prob(x.view(batch_size, -1))
        return -torch.sum(nll, dim=-1)
    elif len(recon.shape) == 2:
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans].contiguous()
        recon_logvar = recon[:, num_half_chans:].contiguous()

        nll = D.Laplace(
            recon_mu.view(batch_size, -1),
            # F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
            recon_logvar.view(batch_size, -1) + eps(half)
        ).log_prob(x.view(batch_size, -1))
        return -torch.sum(nll, dim=-1)

    raise Exception("unknown dimension for laplace NLL")


def nll_gaussian(x, recon):
    """ Negative log-likelihood for gaussian distribution

    :param x: true target
    :param recon: the reconstruction logits
    :returns: batch size NLL
    :rtype: torch.Tensor

    """
    if len(recon.shape) == 4:
        batch_size, num_half_chans = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
        return torch.sum(
            F.mse_loss(input=recon_mu.view(batch_size, -1),
                       target=x.view(batch_size, -1),
                       reduction='none')
        )
    elif len(recon.shape) == 2:
        batch_size, num_half_feat = x.size(0), recon.size(1) // 2
        recon_mu = recon[:, 0:num_half_chans].contiguous()
        return torch.sum(
            F.mse_loss(input=recon_mu.view(batch_size, -1),
                       target=x.view(batch_size, -1),
                       reduction='none')
        )

    raise Exception("unknown dimension for gausian NLL")
