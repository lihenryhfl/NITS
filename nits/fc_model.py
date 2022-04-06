import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from nits.layer import *
from nits.resmade import ResidualMADE

class MLP(nn.Module):
    def __init__(self, arch, residual=False):
        super(MLP, self).__init__()
        self.layers = self.build_net(arch, name_str='d', linear_final_layer=True)
        self.residual = residual
            
    def build_net(self, arch, name_str='', linear_final_layer=True):
        net = nn.ModuleList()
        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            net.append(Linear(a1, a2))
            
            # add nonlinearities
            if i < len(arch) - 2 or not linear_final_layer:
                net.append(nn.ReLU())
                
        return net

    def forward(self, x):
        for l in self.layers:
            residual = self.residual and l.weight.shape[0] == l.weight.shape[1]
            x = l(x) + x if residual else l(x)
        
        return x
    
class RotationParamModel(nn.Module):
    def __init__(self, arch, nits_model, rotate=True, residual=False):
        super(RotationParamModel, self).__init__()
        self.arch = arch
        self.mlp = MLP(arch, residual=residual)
        self.d = arch[0]
        self.n_params = arch[-1]
        self.nits_model = nits_model
        
        self.rotate = rotate
        if rotate:
            self.A = nn.Parameter(torch.randn(self.d, self.d))
        
    def proj(self, x, transpose=False):
        if not self.rotate:
            return x
        
        Q, R = torch.linalg.qr(self.A)
        P = Q.to(x.device)
        
        if transpose:
            P = P.T
        
        return x.mm(P)

    def apply_mask(self, x):   
        x = x.clone()
        
        x_vec = []
        for i in range(self.d):
            tmp = torch.cat([x[:,:i], torch.zeros(len(x), self.d - i, device=x.device)], axis=-1)
            x_vec.append(tmp.unsqueeze(1))
            
        x = torch.cat(x_vec, axis=1).to(x.device)
            
        return x.reshape(-1, self.d)
        
    def forward(self, x):
        n = len(x)
        
        # rotate x
        x = self.proj(x)
        
        # obtain parameters
        x_masked = self.apply_mask(x)
        params = self.mlp(x_masked).reshape(n, self.n_params * self.d)
        
        # compute log likelihood
        ll = (self.nits_model.pdf(x, params) + 1e-10).log().sum()
        
        return ll
    
    def sample(self, n):
        with torch.no_grad():
            data = torch.zeros((n, d), device=device)

            for i in range(d):
                # rotate x
                x = self.proj(x)

                # obtain parameters
                x_masked = self.apply_mask(x)
                params = self.mlp(x_masked).reshape(n, self.n_params * self.d)
                
                sample = self.nits_model.sample(1, params)
                data[:,i] = sample[:,i]

            # apply the inverse projection to the data
            data = self.proj(data, transpose=True)

        return data
    
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
    
class ResMADEModel(nn.Module):
    def __init__(self, d, nits_model, n_residual_blocks=4, hidden_dim=512, 
                 dropout_probability=0., use_batch_norm=False, 
                 zero_initialization=True, weight_norm=False,
                 rotate=False, use_normalizer=False):
        super(ResMADEModel, self).__init__()
        self.d = d
        self.n_params = nits_model.n_params
        self.nits_model = nits_model
        
        if use_normalizer:
            assert not rotate
            self.normalizer = Normalizer(d, d)
        
        self.mlp = ResidualMADE(
            input_dim=self.d,
            n_residual_blocks=n_residual_blocks,
            hidden_dim=hidden_dim,
            output_dim_multiplier=nits_model.n_params,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            weight_norm=weight_norm
        )
        
        self.rotate = rotate
        if rotate:
            self.A = nn.Parameter(torch.randn(self.d, self.d))
        
    def proj(self, x, transpose=False):
        if not self.rotate:
            return x
        
        Q, R = torch.linalg.qr(self.A)
        P = Q.to(x.device)
        
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
        if hasattr(self, 'normalizer'):
            x = self.normalizer(x)
        
        # rotate x
        x = self.proj(x)
        
        # obtain parameters
        params = self.mlp(x)
        
        if hasattr(self, 'normalizer'):
            params = self.add_normalizer_weights(params)
        
        # compute log likelihood
        ll = (self.nits_model.pdf(x, params) + 1e-10).log().sum()
        
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