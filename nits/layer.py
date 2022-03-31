from collections import OrderedDict
from sys import stderr

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

class EMA(nn.Module):
    def __init__(self, model: nn.Module, shadow: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = shadow

        self.update(copy_all=True)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self, copy_all=False):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            if copy_all:
                shadow_params[name].copy_(param)
            else:
                shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model(inputs)
        else:
            return self.shadow(inputs)



class ChannelLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ChannelLinear, self).__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        k = 1 / dim_in
        sqrt_k = np.sqrt(k)
        self.weight = torch.rand(size=(dim_out, dim_in)) * 2 * sqrt_k - sqrt_k
        self.bias = torch.rand(size=(dim_out,)) * 2 * sqrt_k - sqrt_k

        self.weight, self.bias = nn.Parameter(self.weight), nn.Parameter(self.bias)

    def forward(self, x):
        assert len(x.shape) == 4
        xs = [int(k) for k in x.shape]
        assert xs[1] == self.dim_in, 'xs[1]: {}, self.dim_in: {}'.format(xs[1], self.dim_in)
        x = x.reshape(xs[0], xs[1], 1, xs[2], xs[3])
        y = torch.einsum('ij,njkml->nikml', self.weight, x) + self.bias.reshape(-1, self.dim_out, 1, 1, 1)

        return y[:,:,0,:,:]

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, channel_linear=False):
        super(Linear, self).__init__()
        if channel_linear:
            self.linear = wn(ChannelLinear(dim_in, dim_out))
        else:
            self.linear = wn(nn.Linear(dim_in, dim_out))
        self.channel_linear = channel_linear
        self.register_buffer("first_forward", torch.tensor(True).bool())

    def forward(self, x, **kwargs):
        if self.first_forward:
            x_ = self.linear(x, **kwargs)
            if self.channel_linear:
                m, v = x_.mean(dim=(0,2,3)), x_.var(dim=(0,2,3))
            else:
                m, v = x_.mean(dim=(0,)), x_.var(dim=(0,))
            scale_init = 1 / (v + 1e-8).sqrt()
            self.linear.weight_v.data = torch.randn_like(self.linear.weight_v) * 0.05
            self.linear.weight_g.data = torch.ones_like(self.linear.weight_g) * scale_init.reshape(-1, 1)
            self.linear.bias.data = torch.zeros_like(self.linear.bias) - (m * scale_init)
            # we only ever run this once
            self.first_forward = torch.tensor(False).bool().to(self.first_forward.device)

        return self.linear(x, **kwargs)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_channels, out_channels, kernel_size, stride, **kwargs))
        self.register_buffer("first_forward", torch.tensor(True).bool())

    def forward(self, x, **kwargs):
        if self.first_forward:
            x_ = self.conv(x, **kwargs)
            m, v = x_.mean(dim=(0, 2, 3)), x_.var(dim=(0, 2, 3))
            scale_init = 1 / (v + 1e-10).sqrt()
            self.conv.weight_v.data = torch.randn_like(self.conv.weight_v) * 0.05
            self.conv.weight_g.data = torch.ones_like(self.conv.weight_g) * scale_init.reshape(-1, 1, 1, 1)
            self.conv.bias.data = torch.zeros_like(self.conv.bias) - (m * scale_init)
            # we only ever run this once
            self.first_forward = torch.tensor(False).bool().to(self.first_forward.device)

        return self.conv(x, **kwargs)