import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from PIL import Image
from nits.model import ConditionalNITS
from nits.cnn_model import *
from nits.discretized_mol import mix_logistic_loss as unstable_pixelcnn_loss
from nits.discretized_mol import sample_from_mix_logistic as unstable_pixelcnn_sample

def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-g', '--gpus', type=str,
                    default='', help='GPU to use')
parser.add_argument('-i', '--data_dir', type=str,
                    default='/data/datasets/', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-t', '--save_interval', type=int, default=5,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_epoch', type=int, default=0,
                    help='Restore training from previous model checkpoint? If so, which epoch?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=1e-3, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=20,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-mc', '--monotonic_const', type=int,
                    default=7, help='Monotonic const parameter')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-ds', '--discretized', type=bool, default=False,
                    help='Discretized model')
parser.add_argument('-ns', '--no_nits', action='store_true',
                    help='nits model')
parser.add_argument('-at', '--attention', type=str, default='',
                    help='Attention-based NITS')
parser.add_argument('-ni', '--normalize_inverse', type=bool, default=False,
                    help='apply normalization?')
parser.add_argument('-es', '--extra_string', type=str, default='',
                    help='extra string to clarify experiment')
parser.add_argument('-be', '--bisection_eps', type=int, default=6,
                    help='epsilon accuracy for bisection search')
parser.add_argument('-dq', '--dequantize', type=bool, default=False,
                    help='do we dequantize the pixels? performs uniform dequantization')
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[10,1]',
                   help='Architecture of NITS PNN')
parser.add_argument('-w', '--step_weights', type=list_str_to_list, default='[1]',
                   help='Weights for each step of multistep NITS')
parser.add_argument('-bg', '--background', type=bool, default=False,
                   help='Shall we add a background?')
parser.add_argument('-ae', '--autoencoder_weight', type=float, default=0.,
                   help='add autoencoding loss?')
parser.add_argument('-pd', '--polyak_decay', type=float, default=0.99995,
                   help='decay factor for EMA')
args = parser.parse_args()

if args.gpus:
    devices = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpus.split(',')]
else:
    devices = ['cpu']

print(devices)

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.extra_string:
    args.extra_string = '_' + args.extra_string

arch_string = str(args.nits_arch).replace(' ', '').replace(',', '_')[1:-1]
stepweight_string = str(args.step_weights).replace(' ', '').replace(',', '_')[1:-1]

model_name = 'nits{}_discretized{}_dequantize{}_attention{}_bseps{}_arch{}_stepweights{}_background{}_ae{:.0E}_ni{}{}'.format(
    not args.no_nits, args.discretized, args.dequantize, args.attention,
    args.bisection_eps, arch_string, stepweight_string, args.background,
    args.autoencoder_weight, args.normalize_inverse, args.extra_string)
print('model_name:', model_name)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

sample_batch_size = 25
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}

if 'mnist' in args.dataset:
    obs = (1, 28, 28)
    dset = datasets.MNIST
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    dropout = 0.5
    if args.no_nits:
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset:
    obs = (3, 32, 32)
    dset = datasets.CIFAR10
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    dropout = 0.5
    if args.no_nits:
        loss_op_t = loss_op = lambda real, fake : unstable_pixelcnn_loss(real, fake, bad_loss=False, discretize=args.discretized)
        sample_op = lambda x : unstable_pixelcnn_sample(x, args.nr_logistic_mix, bad_loss=False, quantize=False)

elif 'imagenet32' in args.dataset:
    obs = (3, 32, 32)
    dset = datasets.ImageNet
    ds_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(32), rescaling])
    dropout = 0.0
    if args.no_nits:
        loss_op_t = loss_op = lambda real, fake : unstable_pixelcnn_loss(real, fake, bad_loss=False, discretize=args.discretized)
        sample_op = lambda x : unstable_pixelcnn_sample(x, args.nr_logistic_mix, bad_loss=False, quantize=False)
else:
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

# build train and test loaders
train_loader = torch.utils.data.DataLoader(dset(args.data_dir, train=True,
    download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader  = torch.utils.data.DataLoader(dset(args.data_dir, train=False,
                transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)


assert np.isclose(1e-7, np.power(10., -7))

if args.no_nits:
    print('DEFAULT DISCRETIZED MoL')
    # n_params = 100
    n_params = 160
else:
    print("NITS")
    nits_bound = 5
    arch = [1] + args.nits_arch
    nits_model = ConditionalNITS(d=3, start=-nits_bound, end=nits_bound, monotonic_const=np.power(10., -args.monotonic_const),
                                    A_constraint='neg_exp', arch=arch, autoregressive=True,
                                    pixelrnn=True, normalize_inverse=args.normalize_inverse,
                                    final_layer_constraint='softmax',
                                    softmax_temperature=False, bisection_eps=np.power(10., -args.bisection_eps)).to(devices[0])
    print("args.discretized", args.discretized)
    if not args.discretized:
        print("CONTINUOUS NITS")
    if args.discretized:
        print("DISCRETIZED NITS")
    loss_op = lambda real, params: cnn_nits_loss(real, params, nits_model, discretized=args.discretized, dequantize=args.dequantize)
    loss_op_t = lambda real, params: cnn_nits_loss(real, params, nits_model, discretized=args.discretized)
    sample_op = lambda params: cnn_nits_sample(params, nits_model)
    n_params = nits_model.tot_params

# normalize step_weights
step_weights = np.array(args.step_weights)
step_weights = step_weights / (np.sum(step_weights) + 1e-7)

# build multistep loss (if applicable)
def compute_loss(model, input, loss_op, test=False):
    loss = torch.tensor(0., device=input.device)
    ae_loss = torch.tensor(0., device=input.device)
    og_input = input
    og_output = output = model(input)
    i = -1

    if test:
        loss = loss_op(og_input, output)
    else:
        for i in range(len(step_weights) - 1):
            loss += loss_op(og_input, output) * step_weights[i]
            input = sample_op(output)
            output = model(input)

        if step_weights[i+1] > 0:
            loss += loss_op(og_input, output) * step_weights[i + 1]

    if test or args.autoencoder_weight > 0:
        input_reconstruction = sample_op(output)
        ae_loss = ((og_input - input_reconstruction) ** 2).sum() * args.autoencoder_weight

    return loss, ae_loss

input_channels = obs[0]
if args.attention == 'full':
    print("USING FACNN")
    model = FACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params, dropout=dropout)
    shadow = FACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params, dropout=dropout)
elif args.attention == 'snail':
    print("USING SNAIL")
    model = ACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params, dropout=dropout)
    shadow = ACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params, dropout=dropout)
elif args.attention == '':
    print("USING PixelCNN")
    model = CNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params,
                background=args.background, dropout=dropout)
    shadow = CNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params,
                background=args.background, dropout=dropout)

def sample(model):
    with torch.no_grad():
        model.train(False)
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(devices[0])
        for i in range(obs[1]):
            for j in range(obs[2]):
                out   = model(data)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample[:, :, i, j].detach().cpu()
    return data

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95,0.9995))
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

if args.load_epoch > 0:
    load_path = '/data/image_model/models/{}_{}.pth'.format(model_name, args.load_epoch)
    print("loading parameters from path =", load_path)
    load_part_of_model(model, load_path, 'cpu')
    print('model parameters loaded')
else:
    # compute initialization
    model = model.to(devices[0])
    init_loader = torch.utils.data.DataLoader(dset(args.data_dir, download=True,
                        train=True, transform=ds_transforms), batch_size=args.batch_size * 3,
                            shuffle=True, **kwargs)
    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(init_loader):
            input = input.to(devices[0])
            out = model(input)
            del input, out
            break

def standardize_loss(loss, batch_idx):
    if args.no_nits or args.discretized:
        loss /= batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    else:
        loss /= batch_idx * args.batch_size
        loss = ll_to_bpd(-loss, dataset=args.dataset)

    return loss

class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)

model = EMA(model, shadow, decay=args.polyak_decay).to(devices[0])
if len(args.gpus) > 1:
    # model = DataParallel(model, device_ids=devices)
    model = nn.DataParallel(model, device_ids=devices)

print('starting training')
for epoch in range(args.load_epoch, args.max_epochs):
    train_loss = 0.
    train_ae_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.to(devices[0])
        loss, ae_loss = compute_loss(model, input, loss_op)
        optimizer.zero_grad()
        (loss + ae_loss).backward()

        optimizer.step()
        scheduler.step()
        model.module.update()

        train_ae_loss += ae_loss.detach().cpu().numpy()
        train_loss += loss.detach().cpu().numpy()

        # if batch_idx > 50:
            # break

    # compute train loss
    train_loss = standardize_loss(train_loss, batch_idx)
    train_ae_loss /= batch_idx * args.batch_size * np.prod(obs)

    # compute test loss
    model.eval()
    test_loss = 0.
    test_ema_loss = 0.
    test_ae_loss = 0.
    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.to(devices[0])
            loss, ae_loss = compute_loss(model, input, loss_op_t, test=True)
            test_loss += loss.detach().cpu().numpy()
            test_ae_loss += ae_loss.detach().cpu().numpy()
            del loss, ae_loss
            model.train()
            loss, _ = compute_loss(model, input, loss_op_t, test=True)
            test_ema_loss += loss.detach().cpu().numpy()
            model.eval()
            del loss, ae_loss

    test_loss = standardize_loss(test_loss, batch_idx)
    test_ae_loss /= batch_idx * args.batch_size * np.prod(obs)

    print('Epoch: {}, time: {:.0f}, train loss: {:.3f}, ae loss: {:.3f} | test loss: {:.3f}, ema loss: {:.3f}, ae loss: {:.3f}, lr: {:.1e}'.format(
        epoch, time.time() - time_, train_loss, train_ae_loss, test_loss, test_ema_loss, test_ae_loss, optimizer.param_groups[0]['lr']))

    if (epoch + 1) % args.save_interval == 0:
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t,'/data/image_model/images/{}_{}.png'.format(model_name, epoch),
                nrow=5, padding=0)

    if (epoch + 1) % 10 == 0:
        save_path = '/data/image_model/models/{}_{}.pth'.format(model_name, epoch)
        print("saving model to {}".format(save_path))
        torch.save(model.state_dict(), save_path)
