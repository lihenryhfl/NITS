import pdb

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

from nits.layer import *

def load_part_of_model(model, path, device=None):
    params = torch.load(path, map_location=device)
    added = 0
    for name, param in params.items():
        if name in model.state_dict().keys():
            try :
                model.state_dict()[name].copy_(param)
                added += 1
            except Exception as e:
                print(e)
                pass
    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))

def ll_to_bpd(ll, dataset='cifar', bits=8):
    if dataset == 'cifar':
        n_pixels = (32 ** 2) * 3
    elif dataset == 'mnist':
        n_pixels = (28 ** 2)

    bpd = -((ll / n_pixels) - np.log(2 ** (bits - 1))) / np.log(2)
    return bpd

def cnn_nits_loss(x, params, nits_model, eps=1e-7, discretized=False, dequantize=False):
    x = x.permute(0, 2, 3, 1)
    params = params.permute(0, 2, 3, 1)

    nits_model = nits_model.to(x.device)
    x = x.reshape(-1, nits_model.d)
    params = params.reshape(-1, nits_model.tot_params)

    if nits_model.normalize_inverse:
        pre_cdf = nits_model.cdf
        pre_pdf = nits_model.pdf
    else:
        pre_cdf = nits_model.forward_
        pre_pdf = nits_model.backward_

    if nits_model.pixelrnn:
        cdf = lambda x_, params: pre_cdf(x_, params, x_unrounded=x)
        pdf = lambda x_, params: pre_pdf(x_, params, x_unrounded=x)
    else:
        cdf = pre_cdf
        pdf = pre_pdf

    if discretized:
        x_plus = (x * 127.5 + .5).round() / 127.5
        x_min = (x * 127.5 - .5).round() / 127.5

        cdf_plus = cdf(x_plus, params).clamp(max=1-eps, min=eps)
        cdf_min = cdf(x_min, params).clamp(max=1-eps, min=eps)

        cdf_delta = cdf_plus - cdf_min
        log_cdf_plus = (cdf_plus).log()
        log_one_minus_cdf_min = (1 - cdf_min).log()
        log_pdf_mid = (pdf(x, params) + eps).log()

        inner_inner_cond = (cdf_delta > 1e-5).float()
        inner_inner_out  = inner_inner_cond * torch.clamp(cdf_delta, min=1e-12).log() + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
        inner_cond       = (x > 0.999).float()
        inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond             = (x < -0.999).float()
        log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    else:
        if dequantize:
            # add dequantization noise:
            x = x + (torch.rand(x.shape, device=x.device) - 0.5) / 127.5
        log_probs = (pdf(x, params) + eps).log()

    return -log_probs.sum()

def cnn_nits_sample(params, nits_model, quantize=False):
    params = params.permute(0, 2, 3, 1)
    batch_size, height, width, params_per_pixel = params.shape

    nits_model = nits_model.to(params.device)

    imgs = nits_model.sample(1, params.reshape(-1, nits_model.tot_params)).clamp(min=-1., max=1.)
    imgs = imgs.reshape(batch_size, height, width, nits_model.d).permute(0, 3, 1, 2)

    if quantize:
        imgs = ((imgs + 1) * 127.5).round() / 127.5 - 1

    return imgs

class Batcher:
    def __init__(self, x, batch_size):
        self.orig_x = x
        self.batch_size = batch_size

    def __iter__(self):
        self.idx = 0
        p = torch.randperm(len(self.orig_x))
        self.x = self.orig_x[p]

        return self

    def process_x(self, x):
        x = torch.cat([x, torch.zeros((len(x), 1), device=device) + x[:,-1].unsqueeze(-1)], axis=1)
        x = x.reshape(-1, 8, 8).unsqueeze(1)
        return x

    def __next__(self):
        if self.idx + self.batch_size < len(self.x):
            x_ = self.process_x(self.x[self.idx:self.idx+self.batch_size])
            self.idx += self.batch_size
            return x_, None

        raise StopIteration

class AttentionBlock(nn.Module):
    def __init__(self, x_dim, num_filters, K=16):
        super(AttentionBlock, self).__init__()
        self.K = K
        self.V = num_filters // 2
        self.grn_k = GatedResNet(x_dim * 3 + num_filters, NetworkInNetwork)
        self.grn_q = GatedResNet(x_dim * 2 + num_filters, NetworkInNetwork)
        self.grn_v = GatedResNet(x_dim * 3 + num_filters, NetworkInNetwork)
        self.nin_k = NetworkInNetwork(x_dim * 3 + num_filters, self.K)
        self.nin_q = NetworkInNetwork(x_dim * 2 + num_filters, self.K)
        self.nin_v = NetworkInNetwork(x_dim * 3 + num_filters, self.V)
        self.grn_out = GatedResNet(num_filters, NetworkInNetwork, skip_connection=0.5)

    def apply_causal_mask(self, x):
        return torch.tril(x, diagonal=-1)

    def causal_softmax(self, x, dim=-1, eps=1e-7):
        x = self.apply_causal_mask(x)
        x = x.softmax(dim=dim)
        x = self.apply_causal_mask(x)

        # renormalize
        x = x / (x.sum(dim=dim).unsqueeze(dim) + eps)

        return x

    def forward(self, x, ul, b):
        n, c, h, w = x.shape

        ul_b = torch.cat([ul, b], axis=1)
        x_ul_b = torch.cat([x, ul_b], axis=1)

        # compute attention -- presoftmax[:,i,j] = <queries[:,:,i], keys[:,:,j]>
        keys = self.nin_k(self.grn_k(x_ul_b)).reshape(n, self.K, h * w)
        queries = self.nin_q(self.grn_q(ul_b)).reshape(n, self.K, h * w)
        values = self.nin_v(self.grn_v(x_ul_b)).reshape(n, self.V, h * w)
        presoftmax = torch.einsum('nji,njk->nki', keys, queries)

        # apply causal mask and softmax
        att_weights = self.causal_softmax(presoftmax)

        # apply attention
        att_values = torch.einsum('nij,nkj->nki', att_weights, values)

        # reshape
        att_values = att_values.reshape(n, self.V, h, w)

        # add back ul
        result = self.grn_out(ul, a=att_values)

        return result

def causal_shift(x):
    n, c, h, w = x.shape
    x = x.reshape(n, c, h * w)
    zeros = torch.zeros((n, c, 1), device=x.device)
    x = torch.cat([zeros, x[:,:,:-1]], axis=2)
    return x.reshape(n, c, h, w)

def get_background(xs, device):
    # create vertical pattern
    v_pattern = ((torch.arange(xs[2], device=device).float() - xs[2] / 2) / xs[2])
    v_pattern = v_pattern[None, None, :, None].tile(xs[0], xs[1], 1, xs[3])

    # create horizontal pattern
    h_pattern = ((torch.arange(xs[3], device=device).float() - xs[3] / 2) / xs[3])
    h_pattern = h_pattern[None, None, None, :].tile(xs[0], xs[1], xs[2], 1)

    background = torch.cat([v_pattern, h_pattern], axis=1)

    return background

class ACNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, n_params=200, input_channels=3, n_layers=12):
        super(ACNN, self).__init__()

        self.resnet_nonlinearity = concat_elu

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.n_params = n_params
        self.n_layers = n_layers

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.layers = nn.ModuleList([ACNNLayer(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(self.n_layers)])

        self.att_layers = nn.ModuleList([AttentionBlock(input_channels, nr_filters) for _ in range(self.n_layers)])

        self.ul_init = nn.ModuleList([DownShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       DownRightShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        self.nin_out = NetworkInNetwork(nr_filters, n_params)


    def forward(self, x, sample=False):
        if not hasattr(self, 'init_padding') or len(self.init_padding) != len(x):
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device)

        background = get_background(x.shape, x.device)
        x_pad = torch.cat((x, self.init_padding), 1)
        x_causal = causal_shift(x)
        ul = self.ul_init[0](x_pad) + self.ul_init[1](x_pad)

        for i in range(self.n_layers):
            ul = self.layers[i](ul)
            ul = self.att_layers[i](x_causal, ul, background)

        x_out = self.nin_out(F.elu(ul))

        return x_out

class ACNNLayer(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(ACNNLayer, self).__init__()
        self.nr_resnet = nr_resnet
        self.ul_stream = nn.ModuleList([GatedResNet(nr_filters, DownRightShiftedConv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

    def forward(self, ul):
        for i in range(self.nr_resnet):
            ul = self.ul_stream[i](ul)

        return ul

class FACNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, n_params=200, input_channels=3, half_att=True):
        super(FACNN, self).__init__()

        self.resnet_nonlinearity = concat_elu

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.n_params = n_params
        self.half_att = half_att

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([CNNLayerDown(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([CNNLayerUp(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.att_ul_layers = nn.ModuleList([AttentionBlock(input_channels, nr_filters)
                                              for _ in range(6)])
        if not half_att:
            self.att_u_layers = nn.ModuleList([AttentionBlock(input_channels, nr_filters)
                                                  for _ in range(6)])

        self.downsize_u_stream  = nn.ModuleList([DownShiftedConv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([DownRightShiftedConv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([DownShiftedDeconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([DownRightShiftedDeconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([DownShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       DownRightShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        self.nin_out = NetworkInNetwork(nr_filters, n_params)


    def forward(self, x):
        if not hasattr(self, 'init_padding') or len(self.init_padding) != len(x):
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device)

        x_pad = torch.cat((x, self.init_padding), 1)
        x_causal = causal_shift(x)
        u_list  = [self.u_init(x_pad)]
        ul_list = [self.ul_init[0](x_pad) + self.ul_init[1](x_pad)]
        x_list = [x_causal]
        b_list = [get_background(x.shape, x.device)]

        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            # TODO: is this the best way to update ul?
            u, ul = u_list[-1], ul_list[-1]
            ul = self.att_ul_layers[i](x_list[-1], ul, b_list[-1])
            if not self.half_att:
                u = self.att_u_layers[i](x_list[-1], u, b_list[-1])

            if i != 2:
                u_list  += [self.downsize_u_stream[i](u)]
                ul_list += [self.downsize_ul_stream[i](ul)]
                x_list += [F.max_pool2d(x_list[-1], kernel_size=(2,2))]
                b_list += [get_background(x_list[-1].shape, x_list[-1].device)]

        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            x_, b_ = x_list.pop(), b_list.pop()
            ul = self.att_ul_layers[i + 3](x_, ul, b_)
            if not self.half_att:
                u = self.att_u_layers[i + 3](x_, u, b_)

            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

def concat_elu(x):
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

class CNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, n_params=200, input_channels=3, background=False):
        super(CNN, self).__init__()

        self.resnet_nonlinearity = concat_elu

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.n_params = n_params
        self.background = background

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([CNNLayerDown(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([CNNLayerUp(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([DownShiftedConv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([DownRightShiftedConv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([DownShiftedDeconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([DownRightShiftedDeconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        # input shape is slightly different due to various types of padding
        padded_input_channels = input_channels + 7 if background else input_channels + 1

        self.u_init = DownShiftedConv2d(padded_input_channels, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([DownShiftedConv2d(padded_input_channels, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       DownRightShiftedConv2d(padded_input_channels, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        self.nin_out = NetworkInNetwork(nr_filters, n_params)

    def forward(self, x):
        if not hasattr(self, 'init_padding') or len(self.init_padding) != len(x):
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device)

        if self.background:
            background = get_background(x.shape, x.device)
            x = torch.cat([x, background, self.init_padding], axis=1)
        else:
            x = torch.cat((x, self.init_padding), axis=1)

        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

def down_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :xs[2] - 1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    xs = [int(y) for y in x.size()]
    x = x[:, :, :, :xs[3] - 1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)

class NetworkInNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, permute=True):
        super(NetworkInNetwork, self).__init__()
        self.lin_a = Linear(dim_in, dim_out, channel_linear=True)
        self.dim_out = dim_out
        self.permute = permute

    def forward(self, x):
        """ a network in network layer (1x1 CONV) """
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x)
        return out

class DownShiftedConv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(DownShiftedConv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        # arguments of zeropad2d: padding_left, padding_right, padding_top, padding_bottom
        # therefore, for filter_size = (2, 3), this pads 1 left, 1 right, 1 above, and 0 below
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),
                                  int((filter_size[1] - 1) / 2),
                                  filter_size[0] - 1,
                                  0) )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return down_shift(x) if self.shift_output_down else x


class DownShiftedDeconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(DownShiftedDeconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class DownRightShiftedConv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(DownRightShiftedConv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        # arguments of zeropad2d: padding_left, padding_right, padding_top, padding_bottom
        # therefore, for filter_size = (2, 2), this pads 1 left, 0 right, 1 above, and 0 below
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return right_shift(x) if self.shift_output_right else x


class DownRightShiftedDeconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(DownRightShiftedDeconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


class GatedResNet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(GatedResNet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)

        if skip_connection != 0:
            self.nin_skip = NetworkInNetwork(int(2 * skip_connection * num_filters), num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, orig_x, a=None):
        x = self.conv_input(self.nonlinearity(orig_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * b.sigmoid()
        return orig_x + c3

class CNNLayerUp(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(CNNLayerUp, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream  = nn.ModuleList([GatedResNet(nr_filters, DownShiftedConv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        self.ul_stream = nn.ModuleList([GatedResNet(nr_filters, DownRightShiftedConv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class CNNLayerDown(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(CNNLayerDown, self).__init__()
        self.nr_resnet = nr_resnet

        self.u_stream  = nn.ModuleList([GatedResNet(nr_filters, DownShiftedConv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        self.ul_stream = nn.ModuleList([GatedResNet(nr_filters, DownRightShiftedConv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul

