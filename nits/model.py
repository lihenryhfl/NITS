import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NITSMonotonicInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, input, params, given_x):
        with torch.no_grad():
            b = self.bisection_search(input, params, given_x)

        dy = 1 / self.pdf(b, params)
        ctx.save_for_backward(dy.reshape(len(input), -1))
        b = b[:, self.non_conditional_dim].unsqueeze(-1)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        dy, = ctx.saved_tensors
        return None, dy

class NITSPrimitive(nn.Module):
    def __init__(self, arch, start=0., end=1., constraint_type='exp',
                 monotonic_const=1e-3, activation='sigmoid',
                 final_layer_constraint='exp', non_conditional_dim=0):
        super(NITSPrimitive, self).__init__()
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
        self.d = arch[0]
        assert non_conditional_dim < self.d
        self.non_conditional_dim = non_conditional_dim
        self.register_buffer('start_val', torch.tensor(start))
        self.register_buffer('end_val', torch.tensor(end))

    def start_(self, x):
        if x is None:
            assert self.d == 1
            start = torch.ones((1, 1), device=self.start_val.device) * self.start_val
        else:
            start = x.clone().detach()
            start[:, self.non_conditional_dim] = self.start_val

        return start

    def end_(self, x):
        if x is None:
            assert self.d == 1
            end = torch.ones((1, 1), device=self.start_val.device) * self.end_val
        else:
            end = x.clone().detach()
            end[:, self.non_conditional_dim] = self.end_val

        return end

    def apply_constraint(self, A, constraint_type):
        if constraint_type == 'neg_exp':
            A = (-A.clamp(min=-7.)).exp()
        if constraint_type == 'exp':
            A = A.clamp(max=7.).exp()
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
        # the monotonic constant is only applied w.r.t. the first input dimension
        monotonic_x = x[:,self.non_conditional_dim].unsqueeze(-1)

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
                if i < self.last_layer and self.constraint_type == 'neg_exp':
                    x = x - (b.unsqueeze(-1) * A).mean(axis=-1)
                elif i == self.last_layer and self.final_layer_constraint == 'neg_exp':
                    x = x - (b.unsqueeze(-1) * A).mean(axis=-1)
                else:
                    x = x + b
                pre_activations.append(x)
                x = self.apply_act(x)
                nonlinearities.append(self.activation)
            else:
                pre_activations.append(x)
                nonlinearities.append('linear')

        x = x + self.monotonic_const * monotonic_x

        if return_intermediaries:
            return x, pre_activations, As, bs, nonlinearities
        else:
            return x

    def cdf(self, x, params, return_intermediaries=False):
        # get scaling factors
        start = self.forward_(self.start_(x), params)
        end = self.forward_(self.end_(x), params)

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

        # we only want gradients w.r.t. the first input dimension
        return grad[:,self.non_conditional_dim].unsqueeze(-1)

    def backward_(self, x, params):
        y, pre_activations, As, bs, nonlinearities = self.forward_(x, params, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities)

        return grad + self.monotonic_const

    def pdf(self, x, params):
        y, pre_activations, As, bs, nonlinearities = self.cdf(x, params, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities)

        return grad + self.monotonic_const * As[0].reshape(-1, 1)

    def sample(self, params, given_x=None):
        z = torch.rand((len(params), 1), device=params.device)
        x = self.icdf(z, params, given_x=given_x)

        return x

    def icdf(self, z, params, given_x=None):
        func = NITSMonotonicInverse.apply

        return func(self, z, params, given_x)

    def bisection_search(self, y, params, given_x, eps=1e-3):
        low = self.start_(given_x)
        high = self.end_(given_x)

        while ((high - low) > eps).any():
            x_hat = (low + high) / 2
            y_hat = self.cdf(x_hat, params)
            low = torch.where(y_hat > y, low, x_hat)
            high = torch.where(y_hat > y, x_hat, high)

        result = ((high + low) / 2)

        return result

class NITS(NITSPrimitive):
    def __init__(self, d, arch, start=-2., end=2., constraint_type='neg_exp',
                 monotonic_const=1e-2, final_layer_constraint='softmax'):
        super(NITS, self).__init__(arch, start, end,
                                           constraint_type=constraint_type,
                                           monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint)
        self.d = d
        self.tot_params = self.n_params * d
        self.final_layer_constraint = final_layer_constraint

        self.register_buffer('start', torch.tensor(start).reshape(1, 1).tile(1, d))
        self.register_buffer('end', torch.tensor(end).reshape(1, 1).tile(1, d))

        self.nits = NITSPrimitive(arch, start, end,
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

    def apply_conditional_func(self, func, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)
        result = func(x, params)

        if isinstance(result, tuple):
            return (result[0].reshape((n, self.d)),) + result[1:]
        else:
            return result.reshape((n, self.d))

    def forward_(self, x, params, return_intermediaries=False):
        func = lambda x, params: self.nits.forward_(x, params, return_intermediaries)
        return self.apply_conditional_func(func, x, params)

    def backward_(self, x, params):
        return self.apply_conditional_func(self.nits.backward_, x, params)

    def cdf(self, x, params):
        return self.apply_conditional_func(self.nits.cdf, x, params)

    def icdf(self, x, params):
        return self.apply_conditional_func(self.nits.icdf, x, params)

    def pdf(self, x, params):
        return self.apply_conditional_func(self.nits.pdf, x, params)

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

class ConditionalNITS(NITSPrimitive):
    # TODO: for now, just implement ConditionalNITS such that it sequentially evaluates each dimension
    # this process is (probably) possible to vectorize, but since we're currently only doing 3 dimensions,
    # there's no need to speed things up, because we only gain a factor of 3 speedup
    def __init__(self, d, arch, start=-2., end=2., constraint_type='neg_exp',
                 monotonic_const=1e-2, final_layer_constraint='softmax',
                 autoregressive=True):
        super(ConditionalNITS, self).__init__(arch=arch, start=start, end=end,
                                           constraint_type=constraint_type,
                                           monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint)
        self.d = d
        self.tot_params = self.n_params * d
        self.final_layer_constraint = final_layer_constraint
        self.autoregressive = autoregressive

        self.register_buffer('start', torch.tensor(start).reshape(1, 1).tile(1, d))
        self.register_buffer('end', torch.tensor(end).reshape(1, 1).tile(1, d))

        assert arch[0] == d
        self.nits_list = torch.nn.ModuleList()
        for i in range(self.d):
            model = NITSPrimitive(arch=arch, start=start, end=end,
                         constraint_type=constraint_type,
                         monotonic_const=monotonic_const,
                         final_layer_constraint=final_layer_constraint,
                         non_conditional_dim=i)
            self.nits_list.append(model)

    def causal_mask(self, x, i):
        if self.autoregressive:
            x = x.clone()
            x[:,i+1:] = 0.
        return x

    def apply_conditional_func(self, func, x, params):
        n = max(len(x), len(params))
        result = func(x, params)

        if isinstance(result, tuple):
            return (result[0].reshape((n, -1)),) + result[1:]
        else:
            return result.reshape((n, -1))

    def forward_(self, x, params, return_intermediaries=False):
        result = []
        for i in range(self.d):
            x_masked = self.causal_mask(x, i)
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            func = lambda x, params: self.nits_list[i].forward_(x, params, return_intermediaries)
            result.append(self.apply_conditional_func(func, x_masked, params[:,start_idx:end_idx]))

        result = torch.cat(result, axis=1)
        return result

    def backward_(self, x, params):
        result = []
        for i in range(self.d):
            x_masked = self.causal_mask(x, i)
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            func = self.nits_list[i].backward_
            result.append(self.apply_conditional_func(func, x_masked, params[:,start_idx:end_idx]))

        result = torch.cat(result, axis=1)
        return result

    def cdf(self, x, params):
        result = []
        for i in range(self.d):
            x_masked = self.causal_mask(x, i)
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            func = self.nits_list[i].cdf
            result.append(self.apply_conditional_func(func, x_masked, params[:,start_idx:end_idx]))

        result = torch.cat(result, axis=1)
        return result

    def pdf(self, x, params):
        result = []
        for i in range(self.d):
            x_masked = self.causal_mask(x, i)
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            func = self.nits_list[i].pdf
            result.append(self.apply_conditional_func(func, x_masked, params[:,start_idx:end_idx]))

        result = torch.cat(result, axis=1)
        return result

    def icdf(self, x, params, given_x=None):
        if self.autoregressive and given_x is not None:
            raise NotImplementedError('given_x cannot be supplied if autoregressive == True')

        result = []
        for i in range(self.d):
            if self.autoregressive:
                given_x = torch.cat(result + [torch.zeros(len(x), self.d - len(result))], axis=1)
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            func = lambda x, params: self.nits_list[i].icdf(x, params, given_x=given_x)
            result.append(self.apply_conditional_func(func, x, params[:,start_idx:end_idx]))

        result = torch.cat(result, axis=1)
        return result

    def sample(self, n, params):
        if len(params) == 1:
            params = params.reshape(self.d, self.n_params).tile((n, 1))
        elif len(params) == n:
            params = params.reshape(-1, self.n_params)
            
        return self.nits.sample(params).reshape((-1, self.d))
