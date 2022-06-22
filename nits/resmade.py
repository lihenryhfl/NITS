import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init
from torch.nn.utils import weight_norm as wn

def tile(x, n):
    assert isinstance(n, int) and n > 0, 'Argument \'n\' must be an integer.'
    return x.unsqueeze(-1).tile((1, n)).reshape(-1)

def get_mask(in_features, out_features, autoregressive_features, mask_type=None):
    max_ = max(1, autoregressive_features - 1)
    min_ = min(1, autoregressive_features - 1)

    if mask_type == 'input':
        in_degrees = torch.arange(1, autoregressive_features + 1)
        out_degrees = torch.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees).float()

    elif mask_type == 'output':
        in_degrees = torch.arange(in_features) % max_ + min_
        out_degrees = tile(
            torch.arange(1, autoregressive_features + 1),
            out_features // autoregressive_features
        )
        mask = (out_degrees[..., None] > in_degrees).float()

    else:
        in_degrees = torch.arange(in_features) % max_ + min_
        out_degrees = torch.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees).float()

    return mask

# class MaskedLinear(nn.Linear):
#     def __init__(self, in_features, out_features, autoregressive_features, kind=None, bias=True, weight_norm=False):
#         super(MaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
#         assert not weight_norm
#         mask = get_mask(in_features, out_features, autoregressive_features,
#                         mask_type=kind)
#         self.register_buffer('mask', mask)

#     def forward(self, x):
#         return F.linear(x, self.weight * self.mask, self.bias)

class PreMaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, autoregressive_features, kind=None, bias=True):
        super(PreMaskedLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        mask = get_mask(in_features, out_features, autoregressive_features,
                        mask_type=kind)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, autoregressive_features, 
                 kind=None, bias=True, weight_norm=False):
        super(MaskedLinear, self).__init__()
        self.weight_norm = weight_norm
        linear = PreMaskedLinear(in_features, out_features, 
                                      autoregressive_features, kind=kind, bias=bias)
        if weight_norm:
            self.linear = wn(linear)
            self.register_buffer("first_forward", torch.tensor(True).bool())
        else:
            self.linear = linear
            self.weight, self.bias, self.mask = linear.weight, linear.bias, linear.mask

    def forward(self, x, **kwargs):
        if self.weight_norm and self.first_forward:
            x_ = self.linear(x, **kwargs)
            m, v = x_.mean(dim=(0,)), x_.var(dim=(0,))
            scale_init = 1 / (v + 1e-8).sqrt()
            self.linear.weight_v.data = torch.randn_like(self.linear.weight_v) * 0.05
            self.linear.weight_g.data = torch.ones_like(self.linear.weight_g) * scale_init.reshape(-1, 1)
            self.linear.bias.data = torch.zeros_like(self.linear.bias) - (m * scale_init)
            # we only ever run this once
            self.first_forward = torch.tensor(False).bool().to(self.first_forward.device)

        return self.linear(x, **kwargs)

class MaskedResidualBlock(nn.Module):
    def __init__(self, features, autoregressive_features, activation=F.relu,
                 zero_initialization=True, dropout_probability=0., 
                 use_batch_norm=False, weight_norm=False):
        super().__init__()
        self.features = features
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])
        self.layers = nn.ModuleList([
            MaskedLinear(features, features, autoregressive_features, weight_norm=weight_norm)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            eps = 1e-5
            init.uniform_(self.layers[-1].weight, a=-eps, b=eps)
            init.uniform_(self.layers[-1].bias, a=-eps, b=eps)
#             init.zeros_(self.layers[-1].weight)
#             init.zeros_(self.layers[-1].bias)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.layers[1](temps)
        return temps + inputs


class ResidualMADE(nn.Module):
    def __init__(self, input_dim, n_residual_blocks, hidden_dim,
                 output_dim_multiplier, conditional=False, conditioning_dim=None,
                 activation=F.relu, use_batch_norm=False,
                 dropout_probability=None, zero_initialization=True,
                 weight_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim_multiplier = output_dim_multiplier
        self.conditional = conditional

        self.initial_layer = MaskedLinear(
            input_dim,
            hidden_dim,
            input_dim,
            kind='input',
            weight_norm=weight_norm
        )
        if conditional:
            assert conditioning_dim is not None, 'Dimension of condition variables must be specified.'
            self.conditional_layer = nn.Linear(conditioning_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [MaskedResidualBlock(
                features=hidden_dim,
                autoregressive_features=input_dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_probability=0 if dropout_probability is None else dropout_probability,
                zero_initialization=zero_initialization,
                weight_norm=weight_norm
            )
                for _ in range(n_residual_blocks)]
        )
        self.final_layer = MaskedLinear(
            hidden_dim,
            input_dim * output_dim_multiplier,
            input_dim,
            kind='output',
            weight_norm=weight_norm
        )

        self.activation = activation

    def forward(self, inputs, conditional_inputs=None):
        temps = self.initial_layer(inputs)
        del inputs  # free GPU memory
        if self.conditional:
            temps += self.conditional_layer(conditional_inputs)
        for block in self.blocks:
            temps = block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        return outputs


def check_connectivity():
    input_dim = 3
    output_dim_multiplier = 3
    n_residual_blocks = 2
    hidden_dim = 256
    conditional = False
    model = ResidualMADE(
        input_dim=input_dim,
        n_residual_blocks=n_residual_blocks,
        hidden_dim=hidden_dim,
        output_dim_multiplier=output_dim_multiplier,
        conditional=conditional,
        activation=F.relu
    )
    inputs = (torch.rand(1, input_dim) > 0.5).float()
    inputs.requires_grad = True
    res = []
    for k in range(input_dim * output_dim_multiplier):
        outputs = model(inputs)
        outputs[0, k].backward()
        depends = (inputs.grad.data[0].numpy() != 0).astype(np.uint8)
        depends_ix = list(np.where(depends)[0])
        is_valid = k // output_dim_multiplier not in depends_ix
        res.append((len(depends_ix), k, depends_ix, is_valid))

    res.sort()
    for _, k, ix, is_valid in res:
        print('Output {} depends on inputs {} : {}'.format(k, ix, is_valid))


def check_masks():
    input_dim = 3
    hidden_dim = 6
    output_dim_multiplier = 2
    n_residual_blocks = 2

    made = ResidualMADE(
        input_dim=input_dim,
        n_residual_blocks=n_residual_blocks,
        hidden_dim=hidden_dim,
        output_dim_multiplier=output_dim_multiplier
    )

    for module in made.modules():
        if isinstance(module, MaskedLinear):
            print(module.mask.t())


def check_conditional():
    input_dim = 3
    n_hidden_layers = 1
    hidden_dim = 6
    output_dim_multiplier = 2
    conditional = True
    conditional_dim = 4
    made = ResidualMADE(
        input_dim,
        n_hidden_layers,
        hidden_dim,
        output_dim_multiplier,
        conditional,
        conditional_dim
    )
    batch_size = 16
    inputs = torch.randn(batch_size, input_dim)
    conditional_inputs = torch.randn(batch_size, conditional_dim)
    outputs = made(inputs, conditional_inputs)
    print(outputs)


def main():
    check_connectivity()
    check_masks()
    check_conditional()


if __name__ == '__main__':
    main()