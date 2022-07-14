import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init
from torch.nn.utils import weight_norm as wn

# parts adapted from https://github.com/conormdurkan/autoregressive-energy-machines

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
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        mask = get_mask(in_features, out_features, autoregressive_features,
                        mask_type=kind)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, autoregressive_features, 
                 kind=None, bias=True, weight_norm=False):
        super().__init__()
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
    
class AttnLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        k = 1 / dim_in
        sqrt_k = np.sqrt(k)
        self.weight = torch.rand(size=(dim_out, dim_in)) * 2 * sqrt_k - sqrt_k
        self.bias = torch.rand(size=(dim_out,)) * 2 * sqrt_k - sqrt_k

        self.weight, self.bias = nn.Parameter(self.weight), nn.Parameter(self.bias)

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[1] == self.dim_in, 'xs[1]: {}, self.dim_in: {}'.format(x.shape[1], self.dim_in)
        x = x.unsqueeze(2)
        x = torch.einsum('ij,njkm->nikm', self.weight, x)[...,0,:] + self.bias[:,None]

        return x
    
def concat_elu(x, axis=1):
    return F.silu(torch.cat([x, -x], dim=axis))
    
class ResAttnLinear(nn.Module):
    def __init__(self, features, dropout, skip=False):
        super().__init__()
        self.linear_in = AttnLinear(features * 2, features)
        self.linear_out = AttnLinear(features * 2, features * 2)
        self.nonlinearity = concat_elu
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        
        self.skip = skip
        if skip:
            self.linear_skip = AttnLinear(int(2 * (features // 2)), features)
        
    def forward(self, orig_x, a=None):
        x = self.linear_in(self.nonlinearity(orig_x))
        if a is not None:
            x += self.linear_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x) if self.dropout is not None else x
        x = self.linear_out(x)
        
        chunks = torch.chunk(x, 2, dim=1)
        x = chunks[0] * chunks[1].sigmoid()
        return orig_x + x
    
class AttentionBlock(nn.Module):
    def __init__(self, x_dim, features, dropout, K=16):
        super().__init__()
        self.features = features
        self.x_dim = x_dim
        self.K = K
        self.V = features // 2
        self.grn_k = ResAttnLinear(2 + features, dropout=dropout)
        self.grn_q = ResAttnLinear(1 + features, dropout=dropout)
        self.grn_v = ResAttnLinear(2 + features, dropout=dropout)
        self.nin_k = AttnLinear(2 + features, self.K)
        self.nin_q = AttnLinear(1 + features, self.K)
        self.nin_v = AttnLinear(2 + features, self.V)
        self.grn_out = ResAttnLinear(features, skip=True, dropout=dropout)

    def apply_causal_mask(self, x):
        return torch.tril(x, diagonal=-1)

    def causal_softmax(self, x, dim=-1, eps=1e-7):
        x = self.apply_causal_mask(x)
        x = x.softmax(dim=dim)
        x = self.apply_causal_mask(x)

        # renormalize
        x = x / (x.sum(dim=dim).unsqueeze(dim) + eps)

        return x

    def forward(self, x, inputs, b):
        assert x.shape[1] == self.features and x.shape[2] == self.x_dim
        n, d = len(x), self.x_dim
        
        x_b = torch.cat([x, b], axis=1)
        inputs_x_b = torch.cat([inputs, x_b], axis=1)

        # compute attention -- presoftmax[:,i,j] = <queries[:,:,i], keys[:,:,j]>
        keys = self.nin_k(self.grn_k(inputs_x_b))
        keys = keys.reshape(n, self.K, d)
        queries = self.nin_q(self.grn_q(x_b)).reshape(n, self.K, d)
        values = self.nin_v(self.grn_v(inputs_x_b)).reshape(n, self.V, d)
        presoftmax = torch.einsum('nji,njk->nki', keys, queries)

        # apply causal mask and softmax
        att_weights = self.causal_softmax(presoftmax)

        # apply attention
        att_values = torch.einsum('nij,nkj->nki', att_weights, values)

        # reshape
        att_values = att_values.reshape(n, self.V, d)

        # add back ul
        result = self.grn_out(x, a=att_values)

        return result
    
def causal_shift(x):
    n, d = x.shape
    zeros = torch.zeros((n, 1), device=x.device)
    x = torch.cat([zeros, x[:,:-1]], axis=1)
    return x

class CausalTransformer(nn.Module):
    def __init__(self, input_dim, n_blocks, hidden_dim, output_dim_multiplier,
                 activation=F.relu, dropout_probability=None, K=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim_multiplier = output_dim_multiplier

        self.initial_layer = MaskedLinear(
            input_dim,
            (input_dim - 1) * hidden_dim,
            input_dim,
            kind='input'
        )
        self.blocks = nn.ModuleList(
            [AttentionBlock(
                x_dim=(input_dim - 1),
                features=hidden_dim,
                dropout=dropout_probability,
                K=K,
            )
                for _ in range(n_blocks)]
        )
        self.final_layer = MaskedLinear(
            (input_dim - 1) * hidden_dim,
            input_dim * output_dim_multiplier,
            input_dim,
            kind='output'
        )
        
        self.activation = activation
        self.background = nn.Parameter(self.get_background(), requires_grad=False)
        
    def get_background(self):
        d = self.input_dim - 1
        background = ((torch.arange(d).float() - d / 2) / d)
        background = background[None, None, :]

        return background
        
    def forward(self, inputs, conditional_inputs=None):
        # configure background
        b = self.background.tile(len(inputs), 1, 1)
        
        x = inputs
        x = self.initial_layer(x)
        
        # apply causal shift
        inputs = inputs[:, 1:]
        
        # reshape
        x = x.reshape(-1, self.hidden_dim, self.input_dim - 1)
        inputs = inputs.reshape(-1, 1, self.input_dim - 1)
        
        for block in self.blocks:
            x = block(x, inputs, b)
            
        # reshape back
        x = x.reshape(-1, (self.input_dim - 1) * self.hidden_dim)
        
        x = self.activation(x)
        x = self.final_layer(x)
        return x
    

class ResidualMADE(nn.Module):
    def __init__(self, input_dim, n_blocks, hidden_dim,
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
                for _ in range(n_blocks)]
        )
        self.final_layer = MaskedLinear(
            hidden_dim,
            input_dim * output_dim_multiplier,
            input_dim,
            kind='output',
            weight_norm=weight_norm
        )

        self.activation = activation

    def forward(self, x, conditional_inputs=None):
        x = self.initial_layer(x)
        if self.conditional:
            x += self.conditional_layer(conditional_inputs)
        for block in self.blocks:
            x = block(x)
        x = self.activation(x)
        x = self.final_layer(x)
        return x


def check_connectivity():
    input_dim = 6
    output_dim_multiplier = 2
    n_blocks = 2
    hidden_dim = 16
    conditional = False
#     for Encoder in [ResidualMADE, CausalTransformer]:
    for Encoder in [CausalTransformer]:
        print('Testing {}...'.format(Encoder))
        model = Encoder(
            input_dim=input_dim,
            n_blocks=n_blocks,
            hidden_dim=hidden_dim,
            output_dim_multiplier=output_dim_multiplier,
        )
#         inputs = (torch.rand(1, input_dim) > 0.5).float()
        inputs = torch.rand(1, input_dim)
        inputs.requires_grad = True
        res = []
        for k in range(input_dim * output_dim_multiplier):
            outputs = model(inputs)
            outputs[0, k].backward()
            depends = (inputs.grad.data[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            is_valid = k // output_dim_multiplier not in depends_ix
            res.append((len(depends_ix), k, depends_ix, is_valid, k // output_dim_multiplier))

        res.sort()
        for _, k, ix, is_valid, tmp in res:
            print('Output {} depends on inputs {}. Expected? {}, {}'.format(k, ix, is_valid, tmp))


def check_masks():
    input_dim = 3
    hidden_dim = 6
    output_dim_multiplier = 2
    n_blocks = 2
    
    made = ResidualMADE(
        input_dim=input_dim,
        n_blocks=n_blocks,
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
