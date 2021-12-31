import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# MoL FUNCTIONS
def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, (nr_mix * 2):(nr_mix * 3)]
    l = l[:, :, :, :(nr_mix * 2)].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, :nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    x_plus = (x * 127.5 + .5).round() / 127.5
    x_min = (x * 127.5 - .5).round() / 127.5
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (x_plus - means)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (x_min - means)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * (x - means)
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
#     inner_inner_out = cdf_delta.log()
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

#     return log_sum_exp(log_probs)
    return -torch.sum(log_sum_exp(log_probs))

def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1] #[3]

    # unpack parameters
    logit_probs = l[:, :, :, nr_mix * 2:]
    l = l[:, :, :, :nr_mix * 2].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    temp = temp.to(l.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, :nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    u = u.to(l.device)
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out

# NITS FUNCTIONS

def discretized_nits_loss(x, l, arch, nits_model):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()

    nits_model = nits_model.to(x.device)
    x = x.reshape(-1, 1)
    params = l.reshape(-1, nits_model.n_params)

    x_plus = (x * 127.5 + .5).round() / 127.5
    x_min = (x * 127.5 - .5).round() / 127.5

    cdf_delta = nits_model.cdf(x_plus, params) - nits_model.cdf(x_min, params)
#     cdf_delta = nits_model.forward_(x_plus, params) - nits_model.forward_(x_min, params)
    log_cdf_plus = nits_model.cdf(x_plus, params).log()
    log_one_minus_cdf_min = (1 - nits_model.cdf(x_min, params)).log()
    log_pdf_mid = nits_model.pdf(x, params).log()

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
#     inner_inner_out = cdf_delta.log()
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out

    return -log_probs.sum()

def nits_sample(params, arch, i, j, nits_model):
    params = params.permute(0, 2, 3, 1)
    batch_size, height, width, params_per_pixel = params.shape

    nits_model= nits_model.to(params.device)

    n_params = nits_model.n_params
    n_channels = int(params_per_pixel / n_params)

    data = torch.zeros((batch_size, n_channels, height, width))

    imgs = nits_model.sample(1, params[:,i,j,:].reshape(-1, n_params)).clamp(min=-1., max=1.)
    data[:,:,i,j] = imgs.reshape((batch_size, n_channels))

    return data
