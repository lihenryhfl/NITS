import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MonotonicInverse(torch.autograd.Function):
    # computes the inverse and gradient of a monotonic function F(u) = x (think: CDF)
    # i.e., u = F^{-1}(x)
    @staticmethod
    def forward(ctx, params, z, search_func, f, b, idx):
        x = search_func(z, params)

        ctx.save_for_backward(params, x)
        ctx.f = f
        ctx.b = b

        return x[:, idx].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        params, x = ctx.saved_tensors

        with torch.enable_grad():
            params_ = params.detach().clone().requires_grad_(True)
            F = ctx.f(x, params_)
            F.backward(gradient=grad_output)

        dzdx = ctx.b(x, params)
        grad = -params_.grad / (dzdx + 1e-6)

        return grad, 1 / dzdx * grad_output, None, None, None, None

class NITSPrimitive(nn.Module):
    def __init__(self, arch, start=0., end=1., A_constraint='neg_exp',
                 monotonic_const=1e-6, activation='sigmoid',
                 final_layer_constraint='softmax', non_conditional_dim=0,
                 add_residual_connections=False, pixelrnn=False,
                 softmax_temperature=True, normalize_inverse=True,
                 bisection_eps=1e-6):
        super(NITSPrimitive, self).__init__()
        self.arch = arch
        self.monotonic_const = monotonic_const
        self.A_constraint = A_constraint
        self.final_layer_constraint = final_layer_constraint
        self.last_layer = len(arch) - 2
        self.activation = activation
        self.add_residual_connections = add_residual_connections
        self.pixelrnn = pixelrnn
        self.softmax_temperature = softmax_temperature
        self.normalize_inverse = normalize_inverse
        self.bisection_eps = bisection_eps

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
        return self.clone_x_and_fill_conditional_dim_with(x, self.start_val)

    def end_(self, x):
        return self.clone_x_and_fill_conditional_dim_with(x, self.end_val)

    def clone_x_and_fill_conditional_dim_with(self, x, fill_with):
        if x is None:
            assert self.d == 1
            output = torch.ones((1, 1), device=fill_with.device) * fill_with
        else:
            output = x.clone()
            output[:, self.non_conditional_dim] = fill_with

        return output

    def apply_A_constraint(self, A, constraint, params):
        if constraint == 'neg_exp':
            A = (-A.clamp(min=-7.)).exp()
        if constraint == 'exp':
            A = A.clamp(max=7.).exp()
        elif constraint == 'clamp':
            A = A.clamp(min=0.)
        elif constraint == 'softmax':
            if self.softmax_temperature:
                T = params[:, -1].reshape(-1, 1, 1).clamp(max=5., min=-5.).exp()
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
        # store pre-activations and weight matrices
        pre_activations = []
        nonlinearities = []
        As = []
        bs = []
        residuals = []

        # the monotonic constant is only applied w.r.t. the first input dimension
        x = x.clone()
        prev_x = monotonic_x = x[:,self.non_conditional_dim].unsqueeze(-1)

        if self.pixelrnn:
            cur_idx = 3 * self.arch[1]
            linear_weights = params[:,:cur_idx].reshape(-1, 3, self.arch[1]).tanh()
            x_masked = x_unrounded.clone() if x_unrounded is not None else x.clone()
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

    def normalized_forward(self, x, params, x_unrounded=None, return_intermediaries=False):
        # get scaling factors
        start = self.forward_(self.start_(x), params, x_unrounded=x_unrounded)
        end = self.forward_(self.end_(x), params, x_unrounded=x_unrounded)

        # compute pre-scaled cdf, then scale
        y, pre_activations, As, bs, nonlinearities, residuals = self.forward_(x, params, x_unrounded=x_unrounded, return_intermediaries=True)

        scale = 1 / (end - start)
        bias = -start

        y_scaled = (y + bias) * scale

        # accounting
        pre_activations.append(y_scaled)
        As.append(scale.reshape(-1, 1, 1))
        nonlinearities.append('linear')
        residuals.append(False)

        if return_intermediaries:
            return y_scaled, pre_activations, As, bs, nonlinearities, residuals
        else:
            return y_scaled

    def cdf(self, x, params, x_unrounded=None, return_intermediaries=False):
        return self.normalized_forward(x, params, x_unrounded, return_intermediaries)

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

        pre_activations.reverse()
        As.reverse()
        nonlinearities.reverse()
        residuals.reverse()

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
        grad = grad + self.monotonic_const * As[-1].reshape(-1, 1)

        if return_intermediaries:
            return grad, pre_activations, As, bs, nonlinearities, residuals
        else:
            return grad

    def sample(self, params, given_x=None):
        z = torch.rand((len(params), 1), device=params.device)
        x = self.icdf(z, params, given_x=given_x)

        return x

    def bisection_search(self, y, params, given_x, x_unrounded, max_iters=2e3, bisection_eps=1e-4):
        y = y.clone()[:,self.non_conditional_dim].unsqueeze(-1)
        low = self.start_(given_x)
        high = self.end_(given_x)

        n_iters = 0
        while ((high - low) > bisection_eps).any() and n_iters < max_iters:
            x_hat = (high + low) / 2
            if self.normalize_inverse:
                y_hat = self.normalized_forward(x_hat, params, x_unrounded=x_unrounded)
            else:
                y_hat = self.forward_(x_hat, params, x_unrounded=x_unrounded)
            low = torch.where(y_hat > y, low, x_hat)
            high = torch.where(y_hat > y, x_hat, high)

            n_iters += 1
            if n_iters == max_iters:
                print("BISECTION ERROR, n_iters, max_iters", n_iters, max_iters)

        result = ((high + low) / 2)

        return result

    def icdf(self, z, params, given_x=None):
        forward = self.cdf if self.normalize_inverse else self.forward_
        backward = self.pdf if self.normalize_inverse else self.backward_

        search_func = lambda x0, x1: self.bisection_search(x0, x1, given_x=given_x, x_unrounded=None)
        return MonotonicInverse.apply(params, z, search_func, forward, backward, self.non_conditional_dim)
#         return self.fpi(z, params, given_x, None)[:, self.non_conditional_dim].unsqueeze(-1)

    def fpi(self, y, params, given_x, x_unrounded, max_iters=1):
        with torch.no_grad():
            x_hat = self.bisection_search(y, params, given_x, x_unrounded,
                                          max_iters=500, bisection_eps=self.bisection_eps)
        y = y.clone()[:,self.non_conditional_dim].unsqueeze(-1)

        def f(x_hat):
            if self.normalize_inverse:
                return self.normalized_forward(x_hat, params, x_unrounded=x_unrounded)
            else:
                return self.forward_(x_hat, params, x_unrounded=x_unrounded)

        def dfdx(x_hat):
            if self.normalize_inverse:
                return self.pdf(x_hat, params, x_unrounded=x_unrounded)
            else:
                return self.backward_(x_hat, params, x_unrounded=x_unrounded)

        guess = f(x_hat)

        n_iters = 0
        while n_iters < max_iters:
            x_hat = x_hat - (guess - y) / dfdx(x_hat)
            guess = f(x_hat)
            n_iters += 1

        return x_hat

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
        self.add_residual_connections = add_residual_connections
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
        elif len(params) == n or n == 1:
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
                 add_residual_connections=False, softmax_temperature=True,
                 bisection_eps=1e-4, single_mixture=True):
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
        self.bisection_eps = bisection_eps
        self.single_mixture = single_mixture

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
                                  softmax_temperature=softmax_temperature,
                                  bisection_eps=bisection_eps)
            self.nits_list.append(model)

        if pixelrnn and self.single_mixture:
            self.tot_params = sum([m.n_params for m in self.nits_list]) - 20
            self.n_params = self.nits_list[0].n_params - 10
        else:
            self.tot_params = sum([m.n_params for m in self.nits_list])
            self.n_params = self.nits_list[0].n_params

    def causal_mask(self, x, i):
        if self.autoregressive:
            x = x.clone()
            x = torch.cat([x[:,:i+1], torch.zeros_like(x)[:,i+1:]], axis=1)
        return x

    def get_params(self, params, i):
        if self.pixelrnn and self.single_mixture:
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='')
    args = parser.parse_args()
    device = 'cuda:' + args.gpu if args.gpu else 'cpu'
    
    print("Beginning unit test.")
    
    print("Testing gradients.")

    n = 1000
    start, end = -2., 2.
    
    for arch in [[1, 10, 1], [1, 10, 10, 1]]:
        model = NITSPrimitive(arch=arch, start=start, end=end)
        params = torch.randn((n, model.n_params), device=device)
        x = torch.randn((n, 1), device=device)

        out1 = model.backward_(x, params).reshape(-1)

        func = lambda x_: model.forward_(x_, params)
        out2 = torch.autograd.functional.jacobian(func, x, vectorize=True)[:,0,:,0].diagonal()
        assert((out1 - out2).norm() < 1e-5)

    for d in [1, 5]:
        for arch in [[1, 10, 1], [1, 10, 10, 1]]:
            model = NITS(d=d, arch=arch, start=start, end=end)
            params = torch.randn((n, model.tot_params), device=device)
            x = torch.randn((n, d), device=device)

            out1 = model.backward_(x, params)

            func = lambda x_: model.forward_(x_, params)
            tmp = torch.autograd.functional.jacobian(func, x, vectorize=True)
            out2 = torch.cat([tmp[:,i,:,i].diagonal().unsqueeze(-1) for i in range(d)], axis=1)
            assert((out1 - out2).norm() < 1e-5)
            
            # check that the function integrates to 1
            assert torch.allclose(torch.ones((n, d)).to(device),
                                  model.cdf(model.end, params) - model.cdf(model.start, params), atol=1e-5)

            # check that the pdf is all positive
            z = torch.linspace(start, end, steps=n, device=device)[:,None].tile((1, d))
            assert (model.pdf(z, params) >= 0).all()

            # check that the cdf is the inverse of the inverted cdf
            cdf = model.cdf(z, params[0:1])
            icdf = model.icdf(cdf, params[0:1])
            assert (z - icdf <= 1e-3).all()

    d = 3
    for pixelrnn in [True, False]:
        for base_arch in [[10, 1], [10, 10, 1]]:
            arch = [1] + base_arch if pixelrnn else [d] + base_arch
            model = ConditionalNITS(d=d, arch=arch, pixelrnn=pixelrnn, start=start, end=end)
            params = torch.randn((n, model.tot_params), device=device)
            x = torch.randn((n, d), device=device)

            out1 = model.backward_(x, params)

            func = lambda x_: model.forward_(x_, params)
            out2 = torch.autograd.functional.jacobian(func, x, vectorize=True).permute(0,2,1,3).diagonal().diagonal()
            assert((out1 - out2).norm() < 1e-5)
            
            
    from nits.discretized_mol import discretized_mix_logistic_loss, discretized_mix_logistic_loss_1d
    from nits.cnn_model import cnn_nits_loss

    print("Testing single-channel NITS-Conv.")

    model = NITS(
        d=1, start=-1e5, end=1e5, arch=[1, 10, 1],
        monotonic_const=0., A_constraint='neg_exp',
        final_layer_constraint='softmax', 
        softmax_temperature=False).to(device)
    params = torch.randn((n, model.tot_params, 1, 1))
    z = torch.randn((n, 1, 1, 1))

    loss1 = discretized_mix_logistic_loss_1d(z, params)
    loss2 = cnn_nits_loss(z, params, nits_model=model, discretized=True)

    assert (loss1 - loss2).norm() < 1e-2, (loss1 - loss2).norm()

    model = NITS(
        d=1, start=-1e5, end=1e5, arch=[1, 10, 1],
        monotonic_const=0., A_constraint='neg_exp',
        final_layer_constraint='softmax', 
        softmax_temperature=False).to(device)
    params = torch.randn((n, model.tot_params, 2, 2))
    z = torch.randn((n, 1, 2, 2))

    loss1 = discretized_mix_logistic_loss_1d(z, params)
    loss2 = cnn_nits_loss(z, params, nits_model=model, discretized=True)

    assert (loss1 - loss2).norm() < 1e-2, (loss1 - loss2).norm()

    print("Testing multi-channel NITS-Conv.")

    start, end = -2., 2.
    batch_size = 1024

    c_model = ConditionalNITS(d=3, start=start, end=end, arch=[1, 10, 1],
                              monotonic_const=0.,
                              autoregressive=True, 
                              pixelrnn=True, 
                              normalize_inverse=False,
                              softmax_temperature=False).to(device)

    c_params = torch.randn(batch_size, c_model.tot_params, 2, 2, device=device)
    z = torch.rand(batch_size, 3, 2, 2, device=device) * 2 - 1

    # make sure outputs align with pixelrnn
    loss1 = discretized_mix_logistic_loss(z, c_params, bad_loss=True)
    loss2 = cnn_nits_loss(z, c_params, c_model, discretized=True)

    dist_per_dim = (loss1 - loss2).abs() / np.prod(z.shape)

    assert dist_per_dim < 1e-5, dist_per_dim

    # make sure that cdf and icdf return the correct result
    c_params = torch.randn(batch_size, c_model.tot_params, device=device)
    z = torch.rand(batch_size, 3, device=device) * 2 - 1
    cdf_ = c_model.forward_(z, c_params)
    icdf_ = c_model.icdf(cdf_, c_params)

    assert (cdf_ <= 1.).all() and (cdf_ >= 0).all()
    assert (cdf_ <= 1.).all() and (cdf_ >= 0).all()
    assert (z - icdf_).abs().max() < 1e-2


    # test icdf, when normalize_inverse == True (i.e. not EXACTLY pixelrnn anymore)
    c_model = ConditionalNITS(d=3, start=start, end=end, arch=[1, 10, 1],
                              monotonic_const=0.,
                              autoregressive=True, pixelrnn=True, 
                              normalize_inverse=True).to(device)

    # make sure that cdf and icdf return the correct result
    c_params = torch.randn(batch_size, c_model.tot_params, device=device)
    z = torch.rand(batch_size, 3, device=device) * 2 - 1
    cdf_ = c_model.cdf(z, c_params)
    icdf_ = c_model.icdf(cdf_, c_params)

    assert (cdf_ <= 1.).all() and (cdf_ >= 0).all()
    assert (cdf_ <= 1.).all() and (cdf_ >= 0).all()
    assert (z - icdf_).abs().max() < 1e-1

    print("All tests passed!")

    print("Unit test complete. All tests passed!")


