import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

def ll_to_bpd(ll, dataset='cifar', bits=8):
    if dataset == 'cifar':
        n_pixels = (32 ** 2) * 3
    elif dataset == 'mnist':
        n_pixels = (28 ** 2)
        
    bpd = -((ll / n_pixels) - np.log(2 ** (bits - 1))) / np.log(2)
    return bpd

def cnn_nits_loss(x, params, nits_model, eps=1e-7, discretized=False):
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
        log_probs = (pdf(x, params) + eps).log()

    return -log_probs.sum()

def cnn_nits_sample(params, nits_model):
    params = params.permute(0, 2, 3, 1)
    batch_size, height, width, params_per_pixel = params.shape

    nits_model = nits_model.to(params.device)

    imgs = nits_model.sample(1, params.reshape(-1, nits_model.tot_params)).clamp(min=-1., max=1.)
    imgs = imgs.reshape(batch_size, height, width, nits_model.d).permute(0, 3, 1, 2)

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

class CNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nits_params=200, input_channels=3):
        super(CNN, self).__init__()

        def concat_elu(x):
            axis = len(x.size()) - 3
            return F.elu(torch.cat([x, -x], dim=axis))
        self.resnet_nonlinearity = concat_elu

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nits_params = nits_params
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

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

        self.u_init = DownShiftedConv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([DownShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       DownRightShiftedConv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        self.nin_out = NetworkInNetwork(nr_filters, nits_params)
        self.init_padding = None


    def forward(self, x, sample=False):
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.to(x.device)

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.to(x.device)
            x = torch.cat((x, padding), 1)

        x = x if sample else torch.cat((x, self.init_padding), 1)
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
    def __init__(self, dim_in, dim_out):
        super(NetworkInNetwork, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        """ a network in network layer (1x1 CONV) """
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class DownShiftedConv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(DownShiftedConv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),
                                  int((filter_size[1] - 1) / 2),
                                  filter_size[0] - 1,
                                  0) )

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


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
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


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
    def __init__(self, num_filters, conv_op, nonlinearity, skip_connection=0):
        super(GatedResNet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)

        if skip_connection != 0 :
            self.nin_skip = NetworkInNetwork(2 * skip_connection * num_filters, num_filters)

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
        self.u_stream = nn.ModuleList([GatedResNet(nr_filters, DownShiftedConv2d,
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
