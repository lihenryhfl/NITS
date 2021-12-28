import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NITSMonotonicInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, input, params):
        with torch.no_grad():
            b = self.bisection_search(input, params)

        dy = 1 / self.pdf(b, params)
        ctx.save_for_backward(dy.reshape(len(input), -1))
        return b

    @staticmethod
    def backward(ctx, grad_output):
        dy, = ctx.saved_tensors
        return None, dy

class NITS(nn.Module):
    def __init__(self, arch, start=0., end=1., constraint_type='neg_exp',
                 monotonic_const=1e-3, activation='sigmoid', final_layer_constraint='softmax'):
        super(NITS, self).__init__()
        self.arch = arch
        self.monotonic_const = monotonic_const
        self.constraint_type = constraint_type
        self.final_layer_constraint = final_layer_constraint
        self.last_layer = len(arch) - 2
        self.activation = activation

        # count parameters
        self.n_params = 0

        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            self.n_params += (a1 * a2)
            if i < self.last_layer or final_layer_constraint != 'softmax':
                self.n_params += a2

        # set start and end tensors
        self.register_buffer('start', torch.tensor(start).reshape(1, arch[0]))
        self.register_buffer('end', torch.tensor(end).reshape(1, arch[0]))

    def apply_constraint(self, A, constraint_type):
        if constraint_type == 'neg_exp':
            A = (-A).exp()
        if constraint_type == 'exp':
            A = A.exp()
        elif constraint_type == 'clamp':
            A = A.clamp(min=0.)
        elif constraint_type == 'softmax':
            A = F.softmax(A, dim=-1)
        elif constraint_type == '':
            pass

        return A

    def apply_act(self, x):
        if self.activation == 'tanh':
            return x.tanh()
        elif self.activation == 'sigmoid':
            return x.sigmoid()
        elif self.activation == 'linear':
            return x

    def forward_(self, x, params, return_intermediaries=False):
        orig_x = x

        # store pre-activations and weight matrices
        pre_activations = []
        nonlinearities = []
        As = []
        bs = []

        cur_idx = 0

        # compute layers
        for i, (in_features, out_features) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            # get linear weights
            A_end = cur_idx + in_features * out_features
            A = params[:,cur_idx:A_end].reshape(-1, out_features, in_features)
            cur_idx = A_end

            constraint = self.constraint_type if i < self.last_layer else self.final_layer_constraint
            A = self.apply_constraint(A, constraint)
            As.append(A)
            x = torch.einsum('nij,nj->ni', A, x)

            # get bias weights if not softmax layer
            if i < self.last_layer or self.final_layer_constraint != 'softmax':
                b_end = A_end + out_features
                b = params[:,A_end:b_end].reshape(-1, out_features)
                bs.append(b)
                cur_idx = b_end
                if self.constraint_type == 'neg_exp':
                    x = x - (b.unsqueeze(-1) * A).mean(axis=-1)
                else:
                    x = x + b
                pre_activations.append(x)
                x = self.apply_act(x)
                nonlinearities.append(self.activation)
            else:
                pre_activations.append(x)
                nonlinearities.append('linear')
        
        x = x + self.monotonic_const * orig_x

        if return_intermediaries:
            return x, pre_activations, As, bs, nonlinearities
        else:
            return x

    def cdf(self, x, params, return_intermediaries=False):
        # get scaling factors
        start = self.forward_(self.start, params)
        end = self.forward_(self.end, params)

        # compute pre-scaled cdf, then scale
        y, pre_activations, As, bs, nonlinearities = self.forward_(x, params, return_intermediaries=True)
        scale = 1 / (end - start)
        y_scaled = (y - start) * scale

        # accounting
        pre_activations.append(y_scaled)
        As.append(scale.reshape(-1, 1, 1))
        nonlinearities.append('linear')

        if return_intermediaries:
            return y_scaled, pre_activations, As, bs, nonlinearities
        else:
            return y_scaled

    def fc_gradient(self, grad, pre_activation, A, activation):
        if activation == 'linear':
            pass
        elif activation == 'tanh':
            grad = grad * (1 - pre_activation.tanh() ** 2)
        elif activation == 'sigmoid':
            sig_act = pre_activation.sigmoid()
            grad = grad * sig_act * (1 - sig_act)

        return torch.einsum('ni,nij->nj', grad, A)

    def backward_primitive_(self, y, pre_activations, As, bs, nonlinearities):
        pre_activations.reverse()
        As.reverse()
        nonlinearities.reverse()
        grad = torch.ones_like(y, device=y.device)

        for i, (A, pre_activation, nonlinearity) in enumerate(zip(As, pre_activations, nonlinearities)):
            grad = self.fc_gradient(grad, pre_activation, A, activation=nonlinearity)

        return grad

    def backward_(self, x, params):
        y, pre_activations, As, bs, nonlinearities = self.forward_(x, params, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities)

        return grad + self.monotonic_const

    def pdf(self, x, params):
        y, pre_activations, As, bs, nonlinearities = self.cdf(x, params, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities)

        return grad + self.monotonic_const * As[0].reshape(-1, 1)

    def sample(self, params):
        z = torch.rand((len(params), 1), device=params.device)

        with torch.no_grad():
            x = self.icdf(z, params)

        return x

    def icdf(self, z, params):
        func = NITSMonotonicInverse.apply

        return func(self, z, params)

    def bisection_search(self, y, params, eps=1e-3):
        low = torch.ones((len(y), 1), device=y.device) * self.start
        high = torch.ones((len(y), 1), device=y.device) * self.end

        while ((high - low) > eps).any():
            x_hat = (low + high) / 2
            y_hat = self.cdf(x_hat, params)
            low = torch.where(y_hat > y, low, x_hat)
            high = torch.where(y_hat > y, x_hat, high)

        return high

class MultiDimNITS(NITS):
    def __init__(self, d, arch, start=-2., end=2., constraint_type='neg_exp',
                 monotonic_const=1e-2, final_layer_constraint='softmax'):
        super(MultiDimNITS, self).__init__(arch, start, end,
                                           constraint_type=constraint_type,
                                           monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint)
        self.d = d
        self.tot_params = self.n_params * d
        self.final_layer_constraint = final_layer_constraint
        
        self.register_buffer('start', torch.tensor(start).reshape(1, 1).tile(1, d))
        self.register_buffer('end', torch.tensor(end).reshape(1, 1).tile(1, d))
        
        self.nits = NITS(arch, start, end, 
                         constraint_type=constraint_type,
                         monotonic_const=monotonic_const,
                         final_layer_constraint=final_layer_constraint)

    def multidim_reshape(self, x, params):
        n = max(len(x), len(params))
        _, d = x.shape
        assert d == self.d
        assert params.shape[1] == self.tot_params
        assert len(x) == len(params) or len(x) == 1 or len(params) == 1
        
        if len(params) == 1:
            params = params.reshape(self.d, self.n_params).tile((n, 1))
        elif len(params) == n:
            params = params.reshape(-1, self.n_params)
        else:
            raise NotImplementedError('len(params) should be 1 or {}, but it is {}.'.format(n, len(params)))
        
        if len(x) == 1:
            x = x.reshape(1, self.d).tile((n, 1)).reshape(-1, 1)
        elif len(x) == n:
            x = x.reshape(-1, 1)
        else:
            raise NotImplementedError('len(params) should be 1 or {}, but it is {}.'.format(n, len(x)))
            

        return x, params

    def forward_(self, x, params, return_intermediaries=False):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)

        if return_intermediaries:
            x, pre_activations, As, bs, nonlinearities = self.nits.forward_(x, params, return_intermediaries)
            x = x.reshape((n, self.d))
            return x, pre_activations, As, bs, nonlinearities
        else:
            x = self.nits.forward_(x, params, return_intermediaries)
            x = x.reshape((n, self.d))
            return x

    def backward_(self, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)

        return self.nits.backward_(x, params).reshape((n, self.d))

    def cdf(self, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)

        return self.nits.cdf(x, params).reshape((n, self.d))

    def icdf(self, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)

        return self.nits.icdf(x, params).reshape((n, self.d))

    def pdf(self, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)

        return self.nits.pdf(x, params).reshape((n, self.d))

    def sample(self, n, params):
        if len(params) == 1:
            params = params.reshape(self.d, self.n_params).tile((n, 1))
        elif len(params) == n:
            params = params.reshape(-1, self.n_params)

        return self.nits.sample(params).reshape((-1, self.d))

    def initialize_parameters(self, n, constraint_type):
        params = torch.rand((self.d * n, self.n_params))

        def init_constant(params, in_features, constraint_type):
            const = np.sqrt(1 / in_features)
            if constraint_type == 'clamp':
                params = params.abs() * const
            elif constraint_type == 'exp':
                params = params * np.log(const)
            elif constraint_type == 'tanh':
                params = params * np.arctanh(const - 1)

            return params

        cur_idx = 0

        for i, (a1, a2) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            next_idx = cur_idx + (a1 * a2)
            if i < len(self.arch) - 2 or self.final_layer_constraint != 'softmax':
                 next_idx = next_idx + a2
            params[:,cur_idx:next_idx] = init_constant(params[:,cur_idx:next_idx], a2, constraint_type)
            cur_idx = next_idx

        return params.reshape((n, self.d * self.n_params))