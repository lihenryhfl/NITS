import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, constraint_type='clamp', store_weights=True):
        super(PositiveLinear, self).__init__()
        self.constraint_type = constraint_type
        self.in_features, self.out_features = in_features, out_features

        if store_weights:
            # initialize weight in Unif(+eps, sqrt(k)), i.e. log_weight in Unif(log(eps), log(sqrt(k)))
            # where k = 1 / in_features
            pre_weight = torch.rand((out_features, in_features))

            if self.constraint_type == 'exp':
                init_min, init_max = np.log(1e-2), np.log(np.sqrt(1 / in_features))
            elif self.constraint_type == 'clamp':
                init_min, init_max = 1e-2, np.sqrt(1 / in_features)
            elif self.constraint_type == '':
                init_min, init_max = -np.sqrt(1 / in_features), np.sqrt(1 / in_features)
            self.pre_weight = (pre_weight * (init_max - init_min)) + init_min

            bias = torch.rand((out_features))
            scale = 1 / in_features
            self.bias = bias * 2 * scale - scale

            self.pre_weight, self.bias = nn.Parameter(self.pre_weight), nn.Parameter(self.bias)

    def forward(self, x):
        if self.constraint_type == 'neg_exp':
            weight = 1 / self.pre_weight.exp()
            return x.mm(weight.T) - (self.bias.unsqueeze(-1) * weight).mean(axis=-1)
        elif self.constraint_type == 'exp':
            weight = self.pre_weight.exp()
        elif self.constraint_type == 'softmax':
            weight = F.softmax(self.pre_weight, dim=-1)
        elif self.constraint_type == 'clamp':
            weight = self.pre_weight.clamp(min=0.)
        elif self.constraint_type == '':
            weight = self.pre_weight
        return x.mm(weight.T) + self.bias

def bisection_search(increasing_func, target, start, end, n_iter=20, eps=1e-3):
    query = (start + end) / 2
    result = increasing_func(query)

    if n_iter == 0:
        print("bottomed out recursion depth, return best guess epsilon =", (result - target).norm())
        return query
    elif (result - target).norm() < eps:
        return query
    elif result > target:
        return bisection_search(increasing_func, target, start, query, n_iter-1, eps)
    else:
        return bisection_search(increasing_func, target, query, end, n_iter-1, eps)

class MonotonicInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, input, given_x):
        with torch.no_grad():
            b = bisection_search(self.F, input, self.start_(given_x), self.end_(given_x), n_iter=20)

        dy = 1 / torch.autograd.functional.jacobian(self.F, b, create_graph=True, vectorize=True)
        ctx.save_for_backward(dy.reshape(len(input), -1))
        return b[:, self.non_conditional_dim].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        dy, = ctx.saved_tensors
        return None, dy

class ModelInverse(nn.Module):
    def __init__(self, arch, start=0., end=1., store_weights=True,
                 constraint_type='exp', monotonic_const=1e-3,
                 final_layer_constraint='exp', non_conditional_dim=0):
        super(ModelInverse, self).__init__()
        self.d = arch[0]
        self.monotonic_const = monotonic_const
        self.store_weights = store_weights
        self.constraint_type = constraint_type
        self.final_layer_constraint = final_layer_constraint
        self.last_layer = len(arch) - 2
        self.layers = self.build_layers(arch)

        # set start and end tensors
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

    def build_layers(self, arch):
        self.n_params = 0
        layers = nn.ModuleList()

        for i, (a1, a2) in enumerate(zip(arch[:-1], arch[1:])):
            # add nonlinearities
            self.n_params += (a1 * a2)
            if i < self.last_layer:
                layers.append(PositiveLinear(a1, a2, store_weights=self.store_weights,
                                         constraint_type=self.constraint_type))
                layers.append(nn.Sigmoid())
                self.n_params += a2
            else:
                layers.append(PositiveLinear(a1, a2, store_weights=self.store_weights,
                                         constraint_type=self.final_layer_constraint))
                if self.final_layer_constraint != 'softmax':
                    layers.append(nn.Sigmoid())
                    self.n_params += a2

        return layers

    def set_params(self, param_tensor):
        if self.store_weights:
            raise NotImplementedError("set_parameters() should not be called if store_weights == True!")

        assert len(param_tensor) == self.n_params, "{} =/= {}".format(str(param_tensor.shape), str(self.n_params))

        cur_idx = 0
        i = 0
        for layer in self.layers:
            if isinstance(layer, PositiveLinear):
                weight_shape = (layer.out_features, layer.in_features)
                n_params = np.prod(weight_shape)
                layer.pre_weight = param_tensor[cur_idx:cur_idx+n_params].reshape(weight_shape)
                cur_idx += n_params

                if i < self.last_layer or self.final_layer_constraint != 'softmax':
                    layer.bias = param_tensor[cur_idx:cur_idx+layer.out_features]
                    cur_idx += layer.out_features
                else:
                    layer.bias = torch.zeros(layer.out_features).to(param_tensor.device)

                i += 1

    def apply_layers(self, x):
        y = x
        for l in self.layers:
            y = l(y)

        return y + self.monotonic_const * x[:,self.non_conditional_dim].unsqueeze(-1)

    def scale(self, y, x):
        start, end = self.apply_layers(self.start_(x)), self.apply_layers(self.end_(x))
        return (y - start) / (end - start)

    def forward(self, x):
        raise NotImplementedError("forward() should not be used!")

    def f_primitive(self, x, func):
        # compute df/dx
        dy = []
        for x_ in x:
            dy_ = torch.autograd.functional.jacobian(func, x_.reshape(-1, self.d),
                                                     create_graph=True, vectorize=False)
            dy.append(dy_.reshape(-1, self.d)[:,self.non_conditional_dim].unsqueeze(-1))

        dy = torch.cat(dy, axis=0)
        return dy

    def f(self, x):
        return self.f_primitive(x, self.F)

    def f_(self, x):
        return self.f_primitive(x, self.apply_layers)

    def F(self, x):
        return self.scale(self.apply_layers(x), x)

    def pdf(self, x):
        return self.f(x)

    def cdf(self, x):
        return self.F(x)

    def F_inv(self, x, given_x=None):
        inverse = MonotonicInverse.apply

        z = []
        for x_ in x:
            z.append(inverse(self, x_.reshape(1, 1), given_x).reshape(1, 1))

        z = torch.cat(z, axis=0)

        return z

    def sample(self, n, given_x=None, batch_size=1):
        x = []
        while n > batch_size:
            start, end = self.start_(given_x), self.end_(given_x)
            z = torch.rand(batch_size, device=self.start_val.device) * (end - start) + start
            x.append(self.F_inv(z.reshape(-1, 1)))
            n -= batch_size
        else:
            start, end = self.start_(given_x), self.end_(given_x)
            z = torch.rand(n, device=self.start_val.device) * (end - start) + start
            x.append(self.F_inv(z.reshape(-1, 1)))

        return torch.cat(x, axis=0)
