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
    def __init__(self, arch, start=0., end=1., A_constraint='exp',
                 monotonic_const=1e-3, activation='sigmoid',
                 final_layer_constraint='exp', non_conditional_dim=0,
                 add_residual_connections=False, pixelrnn=False,
                 normalize_inverse=True, softmax_temperature=True):
        super(NITSPrimitive, self).__init__()
        self.arch = arch
        self.monotonic_const = monotonic_const
        self.A_constraint = A_constraint
        self.final_layer_constraint = final_layer_constraint
        self.last_layer = len(arch) - 2
        self.activation = activation
        self.add_residual_connections = add_residual_connections
        self.pixelrnn = pixelrnn
        self.normalize_inverse = normalize_inverse
        self.softmax_temperature = softmax_temperature

        # count parameters
        self.n_params = 0
        
        if pixelrnn:
            assert arch[0] == 1
            self.n_params += 3 * arch[1]

        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            self.n_params += (a1 * a2)
            if i < self.last_layer or final_layer_constraint != 'softmax':
                self.n_params += a2
                
        if self.softmax_temperature:
            self.n_params += 1 # softmax temperature

        # set start and end tensors
        self.d = arch[0]
        assert non_conditional_dim < self.d or pixelrnn
        self.non_conditional_dim = non_conditional_dim
        self.register_buffer('start_val', torch.tensor(start))
        self.register_buffer('end_val', torch.tensor(end))

    def start_(self, x):
        if x is None:
            assert self.d == 1
            start = torch.ones((1, 1), device=self.start_val.device) * self.start_val
        else:
            start = x.clone()
            start[:, self.non_conditional_dim] = self.start_val

        return start

    def end_(self, x):
        if x is None:
            assert self.d == 1
            end = torch.ones((1, 1), device=self.start_val.device) * self.end_val
        else:
            end = x.clone()
            end[:, self.non_conditional_dim] = self.end_val

        return end

    def apply_A_constraint(self, A, constraint, params):
        if constraint == 'neg_exp':
            A = (-A.clamp(min=-7.)).exp()
        if constraint == 'exp':
            A = A.clamp(max=7.).exp()
        elif constraint == 'clamp':
            A = A.clamp(min=0.)
        elif constraint == 'softmax':
            if self.softmax_temperature:
                T = params[:, -1].reshape(-1, 1, 1)
            else:
                T = 1
            A = F.softmax(A / T, dim=-1)
        elif constraint == '':
            pass

        return A

    def apply_act(self, x):
        if self.activation == 'tanh':
            return x.tanh()
        elif self.activation == 'sigmoid':
            return x.sigmoid()
        elif self.activation == 'linear':
            return x

    def forward_(self, x, params, x_unrounded=None, return_intermediaries=False):
        # the monotonic constant is only applied w.r.t. the first input dimension
        x = x.clone()
        prev_x = monotonic_x = x[:,self.non_conditional_dim].unsqueeze(-1)

        # store pre-activations and weight matrices
        pre_activations = []
        nonlinearities = []
        As = []
        bs = []
        residuals = []
        
        if self.pixelrnn:
            cur_idx = 3 * self.arch[1]
            linear_weights = params[:,:cur_idx].reshape(-1, 3, self.arch[1]).tanh()
            x_masked = x_unrounded.clone() if x_unrounded is not None else x.clone()
#             x_masked[:, self.non_conditional_dim:] = 0.
            dim = self.non_conditional_dim
            x_masked = torch.cat([x_masked[:,:dim], torch.zeros_like(x_masked)[:,dim:]], axis=1)
            b_linear = torch.einsum('ni,nij->nj', x_masked, linear_weights)
            x = x[:, self.non_conditional_dim].unsqueeze(-1)
        else:
            cur_idx = 0

        # compute layers
        for i, (in_features, out_features) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            # get linear weights
            A_end = cur_idx + in_features * out_features
            A = params[:,cur_idx:A_end].reshape(-1, out_features, in_features)
            cur_idx = A_end

            A_constraint = self.A_constraint if i < self.last_layer else self.final_layer_constraint
            A = self.apply_A_constraint(A, A_constraint, params)
            As.append(A)
            x = torch.einsum('nij,nj->ni', A, x)

            # get bias weights if not softmax layer
            if i < self.last_layer or self.final_layer_constraint != 'softmax':
                b_end = A_end + out_features
                b = params[:, A_end:b_end].reshape(-1, out_features)
                if i == 0 and self.pixelrnn:
                    b = b + b_linear
                bs.append(b)
                cur_idx = b_end
                if A_constraint == 'neg_exp':
                    x = x - (b.unsqueeze(-1) * A).mean(axis=-1)
                else:
                    x = x + b
                pre_activations.append(x)
                nonlinearities.append(self.activation)
                
                # apply activation
                x = self.apply_act(x)
                
                # add residual connection if applicable
                if i > 0 and self.add_residual_connections and x.shape[1] == pre_activations[-2].shape[1]:
                    assert prev_x.shape == x.shape
                    x = x + prev_x
                    residuals.append(True)
                else:
                    residuals.append(False)
            else:
                pre_activations.append(x)
                nonlinearities.append('linear')
                residuals.append(False)
                
            prev_x = x
        
        x = x + self.monotonic_const * monotonic_x

        if return_intermediaries:
            return x, pre_activations, As, bs, nonlinearities, residuals
        else:
            return x

    def cdf(self, x, params, x_unrounded=None, return_intermediaries=False):
        # get scaling factors
        start = self.forward_(self.start_(x), params, x_unrounded=x_unrounded)
        end = self.forward_(self.end_(x), params, x_unrounded=x_unrounded)

        # compute pre-scaled cdf, then scale
        y, pre_activations, As, bs, nonlinearities, residuals = self.forward_(x, params, x_unrounded=x_unrounded, return_intermediaries=True)
        scale = 1 / (end - start)
        idx = (end - start).argmin()
        y_scaled = (y - start) * scale

        # accounting
        pre_activations.append(y_scaled)
        As.append(scale.reshape(-1, 1, 1))
        nonlinearities.append('linear')
        residuals.append(False)

        if return_intermediaries:
            return y_scaled, pre_activations, As, bs, nonlinearities, residuals
        else:
            return y_scaled

    def fc_gradient(self, grad, pre_activation, A, activation, residual):
        orig_grad = grad
        if activation == 'linear':
            pass
        elif activation == 'tanh':
            grad = grad * (1 - pre_activation.tanh() ** 2)
        elif activation == 'sigmoid':
            sig_act = pre_activation.sigmoid()
            grad = grad * sig_act * (1 - sig_act)
            
        result = torch.einsum('ni,nij->nj', grad, A)
        
        if residual:
            result = result + orig_grad

        return result

    def backward_primitive_(self, y, pre_activations, As, bs, nonlinearities, residuals):
        pre_activations.reverse()
        As.reverse()
        nonlinearities.reverse()
        residuals.reverse()
        grad = torch.ones_like(y, device=y.device)

        for i, (A, pre_activation, nonlinearity, residual) in enumerate(zip(As, pre_activations, nonlinearities, residuals)):
            grad = self.fc_gradient(grad, pre_activation, A, activation=nonlinearity, residual=residual)

        if self.pixelrnn:
            return grad
        else:
            # we only want gradients w.r.t. the first input dimension
            return grad[:,self.non_conditional_dim].unsqueeze(-1)

    def backward_(self, x, params, x_unrounded=None, return_intermediaries=False):
        y, pre_activations, As, bs, nonlinearities, residuals = self.forward_(x, params, x_unrounded=x_unrounded, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities, residuals)
        grad = grad + self.monotonic_const
        
        if return_intermediaries:
            return grad, pre_activations, As, bs, nonlinearities, residuals
        else:
            return grad

    def pdf(self, x, params, x_unrounded=None, return_intermediaries=False):
        y, pre_activations, As, bs, nonlinearities, residuals = self.cdf(x, params, x_unrounded=x_unrounded, return_intermediaries=True)

        grad = self.backward_primitive_(y, pre_activations, As, bs, nonlinearities, residuals)
        grad = grad + self.monotonic_const * As[0].reshape(-1, 1)
        
        if return_intermediaries:
            return grad, pre_activations, As, bs, nonlinearities, residuals
        else:
            return grad

    def sample(self, params, given_x=None):
        z = torch.rand((len(params), 1), device=params.device)
        x = self.icdf(z, params, given_x=given_x)

        return x

    def icdf(self, z, params, given_x=None):
        func = NITSMonotonicInverse.apply
        
        return func(self, z, params, given_x)

    def bisection_search(self, y, params, given_x, eps=1e-3):
        y = y[:,self.non_conditional_dim].unsqueeze(-1)
        low = self.start_(given_x)
        high = self.end_(given_x)

        while ((high - low) > eps).any():
            x_hat = (high + low) / 2
            y_hat = self.cdf(x_hat, params) if self.normalize_inverse else self.forward_(x_hat, params)
            low = torch.where(y_hat > y, low, x_hat)
            high = torch.where(y_hat > y, x_hat, high)

        result = ((high + low) / 2)

        return result

class NITS(NITSPrimitive):
    def __init__(self, d, arch, start=-2., end=2., A_constraint='neg_exp',
                 monotonic_const=1e-2, final_layer_constraint='softmax',
                 add_residual_connections=False, normalize_inverse=True,
                 softmax_temperature=True):
        super(NITS, self).__init__(arch, start, end,
                                           A_constraint=A_constraint,
                                           monotonic_const=monotonic_const,
                                           final_layer_constraint=final_layer_constraint)
        self.d = d
        self.tot_params = self.n_params * d
        self.final_layer_constraint = final_layer_constraint
        self.add_residual_connections=add_residual_connections
        self.normalize_inverse = normalize_inverse
        self.softmax_temperature = softmax_temperature

        self.register_buffer('start', torch.tensor(start).reshape(1, 1).tile(1, d))
        self.register_buffer('end', torch.tensor(end).reshape(1, 1).tile(1, d))

        self.nits = NITSPrimitive(arch, start, end,
                                  A_constraint=A_constraint,
                                  monotonic_const=monotonic_const,
                                  final_layer_constraint=final_layer_constraint,
                                  add_residual_connections=add_residual_connections,
                                  normalize_inverse=normalize_inverse,
                                  softmax_temperature=softmax_temperature)

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

    def apply_func(self, func, x, params):
        n = max(len(x), len(params))
        x, params = self.multidim_reshape(x, params)
        result = func(x, params)

        if isinstance(result, tuple):
            return (result[0].reshape((n, self.d)),) + result[1:]
        else:
            return result.reshape((n, self.d))

    def forward_(self, x, params, return_intermediaries=False):
        func = lambda x, params: self.nits.forward_(x, params, return_intermediaries)
        return self.apply_func(func, x, params)

    def backward_(self, x, params):
        return self.apply_func(self.nits.backward_, x, params)

    def cdf(self, x, params):
        return self.apply_func(self.nits.cdf, x, params)

    def icdf(self, x, params):
        return self.apply_func(self.nits.icdf, x, params)

    def pdf(self, x, params):
        return self.apply_func(self.nits.pdf, x, params)

    def sample(self, n, params):
        if len(params) == 1:
            params = params.reshape(self.d, self.n_params).tile((n, 1))
        elif len(params) == n:
            params = params.reshape(-1, self.n_params)

        return self.nits.sample(params).reshape((-1, self.d))

    def initialize_parameters(self, n, A_constraint):
        params = torch.rand((self.d * n, self.n_params))

        def init_constant(params, in_features, A_constraint):
            const = np.sqrt(1 / in_features)
            if A_constraint == 'clamp':
                params = params.abs() * const
            elif A_constraint == 'exp':
                params = params * np.log(const)
            elif A_constraint == 'tanh':
                params = params * np.arctanh(const - 1)

            return params

        cur_idx = 0

        for i, (a1, a2) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            next_idx = cur_idx + (a1 * a2)
            if i < len(self.arch) - 2 or self.final_layer_constraint != 'softmax':
                 next_idx = next_idx + a2
            params[:,cur_idx:next_idx] = init_constant(params[:,cur_idx:next_idx], a2, A_constraint)
            cur_idx = next_idx

        return params.reshape((n, self.d * self.n_params))

class ConditionalNITS(nn.Module):
    # TODO: for now, just implement ConditionalNITS such that it sequentially evaluates each dimension
    # this process is (probably) possible to vectorize, but since we're currently only doing 3 dimensions,
    # there's no need to speed things up, because we only gain a factor of 3 speedup
    def __init__(self, d, arch, start=-2., end=2., A_constraint='neg_exp',
                 monotonic_const=1e-2, final_layer_constraint='softmax',
                 autoregressive=True, pixelrnn=False, normalize_inverse=True,
                 add_residual_connections=False, softmax_temperature=True):
        super(ConditionalNITS, self).__init__()
        
        self.d = d
        self.final_layer_constraint = final_layer_constraint
        self.autoregressive = autoregressive
        self.add_residual_connections = add_residual_connections

        self.register_buffer('start', torch.tensor(start).reshape(1, 1).tile(1, d))
        self.register_buffer('end', torch.tensor(end).reshape(1, 1).tile(1, d))
        
        self.pixelrnn = pixelrnn
        self.normalize_inverse = normalize_inverse
        self.softmax_temperature = softmax_temperature

        assert arch[0] == d or pixelrnn
        self.nits_list = torch.nn.ModuleList()
        for i in range(self.d):
            model = NITSPrimitive(arch=arch, start=start, end=end,
                                  A_constraint=A_constraint,
                                  monotonic_const=monotonic_const,
                                  final_layer_constraint=final_layer_constraint,
                                  non_conditional_dim=i, pixelrnn=pixelrnn,
                                  normalize_inverse=normalize_inverse,
                                  add_residual_connections=add_residual_connections,
                                  softmax_temperature=softmax_temperature)
            self.nits_list.append(model)
            
        if pixelrnn:
            self.tot_params = sum([m.n_params for m in self.nits_list]) - 20
            self.n_params = self.nits_list[0].n_params - 10
        else:
            self.tot_params = sum([m.n_params for m in self.nits_list])
            self.n_params = self.nits_list[0].n_params

    def causal_mask(self, x, i):
        if self.autoregressive:
            x = x.clone()
            x = torch.cat([x[:,:i+1], torch.zeros_like(x)[:,i+1:]], axis=1)
#             x[:,i+1:] = 0.
        return x
    
    def get_params(self, params, i):
        if self.pixelrnn:
            start_idx, end_idx = 10 + i * self.n_params, 10 + (i + 1) * self.n_params
            return torch.cat([params[:,start_idx:end_idx], params[:,:10]], axis=1)
        else:
            start_idx, end_idx = i * self.n_params, (i + 1) * self.n_params
            return params[:,start_idx:end_idx]

    def apply_conditional_func(self, func_name, x, params, x_unrounded=None, given_x=None, return_intermediaries=False):
        n = max(len(x), len(params))
        results = []
        if return_intermediaries:
            pre_activations, As, bs, nonlinearities, residuals = [], [], [], [], []
        for i in range(self.d):
            params_ = self.get_params(params, i)
            func = getattr(self.nits_list[i], func_name)
            if func_name == 'icdf':
                if self.autoregressive:
                    given_x = torch.cat(results + [torch.zeros(len(x), self.d - len(results), device=params.device)], axis=1)
                result = func(x, params_, given_x=given_x)
            else:
                x_masked = self.causal_mask(x, i)
                result = func(x_masked, params_, x_unrounded=x_unrounded, return_intermediaries=return_intermediaries)

            if isinstance(result, tuple):
                results.append(result[0].reshape((n, -1)))
                pre_activations.append(result[1])
                As.append(result[2])
                bs.append(result[3])
                nonlinearities.append(result[4])
                residuals.append(result[5])
            else:
                results.append(result.reshape((n, -1)))

        results = torch.cat(results, axis=1)
        
        if return_intermediaries:
            return results, pre_activations, As, bs, nonlinearities, residuals
        else:
            return results

    def forward_(self, x, params, x_unrounded=None, return_intermediaries=False):
        return self.apply_conditional_func('forward_', x, params, x_unrounded=x_unrounded, return_intermediaries=return_intermediaries)

    def backward_(self, x, params, x_unrounded=None, return_intermediaries=False):
        return self.apply_conditional_func('backward_', x, params, x_unrounded=x_unrounded, return_intermediaries=return_intermediaries)

    def cdf(self, x, params, x_unrounded=None, return_intermediaries=False):
        return self.apply_conditional_func('cdf', x, params, x_unrounded=x_unrounded, return_intermediaries=return_intermediaries)

    def pdf(self, x, params, x_unrounded=None, return_intermediaries=False):
        return self.apply_conditional_func('pdf', x, params, x_unrounded=x_unrounded, return_intermediaries=return_intermediaries)

    def icdf(self, x, params, given_x=None):
        return self.apply_conditional_func('icdf', x, params, given_x=given_x)

    def sample(self, n, params):
        if len(params) == 1:
            params = params.reshape(-1, self.tot_params).tile((n, 1))
        elif len(params) == n:
            params = params.reshape(-1, self.tot_params)
        elif n == 1:
            n = len(params)
        else:
            raise NotImplementedError("Either n == len(params) or one of n, len(params) must be 1. ")
            
        z = torch.rand((n, self.d)).to(params.device)
            
        return self.icdf(z, params)

