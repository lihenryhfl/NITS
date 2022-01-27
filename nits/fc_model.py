import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, arch):
        super(MLP, self).__init__()
        self.layers = self.build_net(arch, name_str='d', linear_final_layer=True)
            
    def build_net(self, arch, name_str='', linear_final_layer=True):
        net = nn.ModuleList()
        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            net.append(nn.Linear(a1, a2))
            
            # add nonlinearities
            if i < len(arch) - 2 or not linear_final_layer:
                net.append(nn.ReLU())
                
        return net

    def forward(self, x):
        y = x
        for l in self.layers:
            y = l(y)
        
        return y
    
class ParamModel(nn.Module):
    def __init__(self, arch):
        super(ParamModel, self).__init__()
        self.arch = arch
        self.mlp = MLP(arch)
        self.d = arch[0]
        self.n_params = arch[-1]

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
        x = self.apply_mask(x)
        params = self.mlp(x).reshape(n, self.n_params * self.d)
        return params