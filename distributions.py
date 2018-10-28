import torch
import torch.nn.functional as F
import torch.distributions as D

from .utils import ones_like
from .pixel_cnn.utils import discretized_mix_logistic_loss, \
    discretized_mix_logistic_loss_1d, sample_from_discretized_mix_logistic_1d, \
    sample_from_discretized_mix_logistic


def nll_activation(logits, nll_type, **kwargs):
    ''' helper to activate logits based on the NLL '''
    if nll_type == "disc_mix_logistic":
        assert 'chans' in kwargs, "need channels for disc_mix_logistic"
        fn = sample_from_discretized_mix_logistic_1d if kwargs['chans'] == 1 \
            else sample_from_discretized_mix_logistic
        # ideally do this, but needs parameterization
        # return fn(logits, **kwargs)
        return fn(logits, nr_mix=10)
    elif nll_type == "gaussian":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        #return torch.sigmoid(logits_mu)
        return logits_mu
    elif nll_type == "bernoulli":
        return torch.sigmoid(logits)
    else:
        raise Exception("unknown nll provided")


def nll_has_variance(nll_str):
    ''' a simple helper to return whether we have variance in the likelihood'''
    nll_map = {
        'gaussian': True,
        'laplace': True,
        'bernoulli': False,
        'disc_mix_logistic': True
    }

    assert nll_str in nll_map
    return nll_map[nll_str]


def nll(x, recon_x, nll_type):
    ''' helper to get the actual NLL evaluation '''
    nll_map = {
        "gaussian": nll_gaussian,
        "bernoulli": nll_bernoulli,
        "laplace": nll_laplace,
        "disc_mix_logistic": nll_disc_mix_logistic
    }
    return nll_map[nll_type](x, recon_x.contiguous())


def nll_bernoulli(x, recon_x_logits):
    batch_size = x.size(0)
    nll = D.Bernoulli(logits=recon_x_logits.view(batch_size, -1)).log_prob(
        x.view(batch_size, -1)
    )
    return -torch.sum(nll, dim=-1)


def nll_disc_mix_logistic(x, recon):
    assert len(x.shape) == len(recon.shape) == 4, "expecting 4d input for logistic loss"
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
