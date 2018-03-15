import torch
import torch.nn.functional as F
import torch.distributions as D

from .utils import ones_like

def log_logistic_256(x, mean, log_s, average=False, dim=None):
    ''' from jmtomczak's github'''
    binsize, scale = 1. / 256., torch.exp(log_s)
    x = torch.floor(x/binsize) * binsize  # make sure image fit proper values
    x_plus = (x + binsize - mean) / scale # calculate normalized values for a bin
    x_minus = (x - mean) / scale
    cdf_plus = F.sigmoid(x_plus)          # calculate logistic CDF for a bin
    cdf_minus = F.sigmoid(x_minus)

    # calculate final log-likelihood for an image
    # and return a mean or sum
    likelihood = - torch.log(cdf_plus - cdf_minus + 1.e-7)
    reduction_fn = torch.mean if average else torch.sum
    return reduction_fn(likelihood, dim )


def nll_activation(logits, nll_type):
    ''' helper to activate logits based on the NLL '''
    if nll_type == "clamp":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        return torch.clamp(logits_mu, min=0.+1./512., max=1.-1./512.)
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
    ''' log-logistic with clamping '''
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()
    return log_logistic_256(x.view(batch_size, -1),
                            torch.clamp(recon_mu.view(batch_size, -1), min=0.+1./512., max=1.-1./512.),
                            F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0),
                            dim=-1)


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
