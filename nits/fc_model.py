import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from nits.layer import *
from nits.resmade import ResidualMADE
    
class Normalizer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, trainable=True):
        super(Normalizer, self).__init__()
        assert in_features == out_features
        self.weight_diag = nn.Parameter(torch.zeros(size=(in_features,)), requires_grad=trainable)
        self.bias = nn.Parameter(torch.zeros(size=(in_features,)), requires_grad=trainable)
        self.register_buffer("weights_set", torch.tensor(False).bool())
        self.d = in_features
        
    def set_weights(self, x, device):
        assert x.device == torch.device('cpu')
        m, v = x.mean(dim=(0,)), x.var(dim=(0,))
        self.bias.data = -m
        self.weight_diag.data = 1 / (v + 1e-8).sqrt()
        self.weights_set.data = torch.tensor(True).bool().to(self.weights_set.device)

    def forward(self, x):
        if not self.weights_set:
            raise Exception("Need to set weights first!")
            
        return self.weight_diag * x + self.bias
    
class Model(nn.Module):
    def __init__(self, d, nits_model, param_model, n_blocks=4, hidden_dim=512, 
                 dropout_probability=0., rotate=False, use_normalizer=False, **param_model_args):
        super().__init__()
        self.d = d
        self.n_params = nits_model.n_params
        self.nits_model = nits_model
        
        if use_normalizer:
            assert not rotate
            self.normalizer = Normalizer(d, d)
        
        self.mlp = param_model(
            input_dim=self.d,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            output_dim_multiplier=nits_model.n_params,
            dropout_probability=dropout_probability,
            **param_model_args
        )
        
        self.rotate = rotate
        if rotate:
            self.A = nn.Parameter(torch.randn(self.d, self.d))
            
    def get_P(self):
        P, _ = torch.linalg.qr(self.A)
        
        return P
        
    def proj(self, x, transpose=False):
        if not self.rotate:
            return x
        
        P = self.get_P()
        
        if transpose:
            P = P.T
        
        return x.mm(P)
    
    def add_normalizer_weights(self, params):
        idx = torch.arange(self.d).reshape(1, -1)
        idx = idx.tile(len(params), 1).reshape(-1)
        A = self.normalizer.weight_diag[idx].reshape(-1, self.d, 1)
        b = self.normalizer.bias[idx].reshape(-1, self.d, 1)
        
        reshaped_params = params.reshape(-1, self.d, self.nits_model.n_params)
        new_params = torch.cat([A, b, reshaped_params[:,:,2:]], axis=2)
        return new_params.reshape(-1, self.nits_model.tot_params)
        
    def forward(self, x):
        ll = self.pdf(x).log().sum()
        
        return ll
    
    def sample(self, n):
        with torch.no_grad():
            data = torch.zeros((n, self.d), device=A.device)

            for i in range(self.d):
                # rotate x
                x_proj = self.proj(data)

                # obtain parameters
                params = self.mlp(data)
                
                sample = self.nits_model.sample(1, params)
                data[:,i] = sample[:,i]

            # apply the inverse projection to the data
            data = self.proj(data, transpose=True)

        return data
    
    def pdf(self, x):
        if hasattr(self, 'normalizer'):
            x = self.normalizer(x)
        
        # rotate x
        x = self.proj(x)
        
        # obtain parameters
        params = self.mlp(x)
        
        if hasattr(self, 'normalizer'):
            params = self.add_normalizer_weights(params)
        
        # compute log likelihood
        pdf = (self.nits_model.pdf(x, params) + 1e-10)
        
        return pdf
    
    def lin_pdf(self, x, dim, n=128):
        assert len(x) == 1
        shape = x.shape
        xmin = self.nits_model.start[0, dim].item()
        xmax = self.nits_model.end[0, dim].item()
        
        with torch.no_grad():
            # obtain parameters
            params = self.mlp(x)
            data = x.tile((n,) + (1,) * (len(shape) - 1))
            data[:, dim] = torch.linspace(xmin, xmax, n)
            
            pdf = self.nits_model.pdf(data, params)

        return pdf[:, dim]
    
    def plot_pdf(self, x, dim, n=128):
        shape = x.shape
        xmin = self.nits_model.start[0, dim].item()
        xmax = self.nits_model.end[0, dim].item()
        
        xval = torch.linspace(xmin, xmax, n)
        yval = self.lin_pdf(x, dim, n)
        
        plt.scatter(xval, yval.cpu())
    