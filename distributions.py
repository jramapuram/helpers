import torch
import torch.nn.functional as F
import torch.distributions as D

from .utils import ones_like
from .pixel_cnn.utils import discretized_mix_logistic_loss, \
    discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d, \
    sample_from_discretized_mix_logistic


def nll_activation(logits, nll_type, **kwargs):
    ''' helper to activate logits based on the NLL '''
    if nll_type == "clamp":
        fn = sample_from_discretized_mix_logistic_1d if logits.shape[1] == 1 \
            else sample_from_discretized_mix_logistic
        # ideally do this, but needs parameterization
        # return fn(logits, **kwargs)
        return fn(logits, nr_mix=10)
    elif nll_type == "gaussian":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        #return F.sigmoid(logits_mu)
        return logits_mu
    elif nll_type == "bernoulli":
        return F.sigmoid(logits)
    else:
        raise Exception("unknown nll provided")


def nll(x, recon_x, nll_type):
    ''' helper to get the actual NLL evaluation '''
    nll_map = {
        "gaussian": nll_gaussian,
        "bernoulli": nll_bernoulli,
        "laplace": nll_laplace,
        "clamp": nll_clamp
    }
    return nll_map[nll_type](x, recon_x)


def nll_bernoulli(x, recon_x_logits):
    batch_size = x.size(0)
    nll = D.Bernoulli(logits=recon_x_logits.view(batch_size, -1)).log_prob(
        x.view(batch_size, -1)
    )
    return -torch.sum(nll, dim=-1)


def nll_clamp(x, recon):
    fn = discretized_mix_logistic_loss_1d if x.shape[1] == 1 \
        else discretized_mix_logistic_loss
    return fn(x, recon)


def nll_laplace(x, recon):
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()

    nll = D.Laplace(
        # recon_mu.view(batch_size, -1),
        recon_mu.view(batch_size, -1),
        # F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
        recon_logvar.view(batch_size, -1)
    ).log_prob(x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)


def nll_gaussian(x, recon):
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()

    # XXX: currently broken, so set var to 1
    recon_logvar = ones_like(recon_mu)

    nll = D.Normal(
        recon_mu.view(batch_size, -1),
        #F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
        recon_logvar.view(batch_size, -1)
    ).log_prob(x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)
