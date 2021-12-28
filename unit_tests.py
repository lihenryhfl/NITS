import torch
from model import *
from autograd_model import *

device = 'cpu'

############################
# DEFINE PARAMETERS:  #
############################

n = 1024
d = 1
monotonic_const = 1e-3
constraint_type = 'exp'
final_layer_constraint = 'softmax'
start, end = -2., 2.
arch = [1, 8, 8, 1]
model = MultiDimNITS(d=d, start=start, end=end, arch=arch,
                     monotonic_const=monotonic_const, constraint_type=constraint_type,
                     final_layer_constraint=final_layer_constraint).to(device)
params = torch.randn((n, d * model.n_params)).to(device)

############################
# SANITY CHECK:  #
############################

# check that the function integrates to 1
assert torch.allclose(torch.ones((n, d)).to(device),
                      model.cdf(model.end, params) - model.cdf(model.start, params), atol=1e-5)

# check that the pdf is all positive
z = torch.linspace(start, end, steps=n, device=device)[:,None].tile((1, d))
assert (model.pdf(z, params) >= 0).all()

# check that the cdf is the inverted
cdf = model.cdf(z, params[0:1])
icdf = model.icdf(cdf, params[0:1])
assert (z - icdf <= 1e-3).all()

############################
# COMPARE TO AUTOGRAD NITS #
############################
autograd_model = ModelInverse(arch=arch, start=start, end=end, store_weights=False,
                              constraint_type=constraint_type, monotonic_const=monotonic_const,
                              final_layer_constraint=final_layer_constraint)

def zs_params_to_forwards(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
            autograd_model.set_params(param[start_idx:end_idx])
            out.append(autograd_model.apply_layers(z[d_:d_+1][None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = zs_params_to_forwards(z, params)
outs = model.forward_(z, params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

def zs_params_to_cdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
            autograd_model.set_params(param[start_idx:end_idx])
            out.append(autograd_model.cdf(z[d_:d_+1][None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = zs_params_to_cdfs(z, params)
outs = model.cdf(z, params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)

def zs_params_to_pdfs(zs, params):
    out = []
    for z, param in zip(zs, params):
        for d_ in range(d):
            start_idx, end_idx = d_ * autograd_model.n_params, (d_ + 1) * autograd_model.n_params
            autograd_model.set_params(param[start_idx:end_idx])
            out.append(autograd_model.pdf(z[d_:d_+1][None,:]))

    out = torch.cat(out, axis=0).reshape(-1, d)
    return out

autograd_outs = zs_params_to_pdfs(z, params)
outs = model.pdf(z, params)
assert torch.allclose(autograd_outs, outs, atol=1e-4)
