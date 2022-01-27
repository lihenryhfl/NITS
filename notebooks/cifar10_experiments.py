import time
import os, shutil
import argparse
import torch
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from nits.cnn_model import *
from nits.model import NITS, ConditionalNITS
from nits.discretized_mol import nits_loss, nits_sample

def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s

parser = argparse.ArgumentParser()

# data
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-i', '--data_dir', type=str, default='/data/pixelcnn/data/')
parser.add_argument('-o', '--save_dir', type=str, default='/data/pixelcnn/models/')
parser.add_argument('-d', '--dataset', type=str, default='cifar')
parser.add_argument('-p', '--print_every', type=int, default=50)
parser.add_argument('-t', '--save_interval', type=int, default=10)
parser.add_argument('-r', '--load_params', type=str, default=None)

# cnn weight model
parser.add_argument('-l', '--lr', type=float, default=2e-4)
parser.add_argument('-e', '--lr_decay', type=float, default=(1 - 5e-6))
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-x', '--max_epochs', type=int, default=5000)
parser.add_argument('-s', '--seed', type=int, default=1)

# nits model
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[8,8,1]',
                   help='Architecture of NITS PNN')
parser.add_argument('-nb', '--nits_bound', type=float, default=5.,
                    help='Upper and lower bound of NITS model')
parser.add_argument('-c', '--constraint', type=str, default='neg_exp',
                    help='Constraint type of NITS')
parser.add_argument('-fc', '--final_constraint', type=str, default='softmax',
                    help='Final constraint of NITS')
parser.add_argument('-st', '--softmax_temp', type=bool, default=True,
                    help='Use of softmax temperature')
parser.add_argument('-ds', '--discretized', type=bool, default=False,
                    help='Discretized NITS')


args = parser.parse_args()

device = 'cuda:' + args.gpu if args.gpu else 'cpu'
print('device:', device)

# HOUSEKEEPING

torch.manual_seed(args.seed)
np.random.seed(args.seed)

sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
                        train=True, transform=ds_transforms), batch_size=args.batch_size,
                            shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
elif 'cifar' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))


# INITIALIZE NITS MODEL
if 'mnist' in args.dataset:
    arch = [1] + args.nits_arch
    nits_model = NITS(d=1, start=-args.nits_bound, end=args.nits_bound, monotonic_const=1e-5,
                      A_constraint=args.constraint, arch=arch,
                      final_layer_constraint=args.final_constraint,
                      softmax_temperature=True).to(device)
elif 'cifar' in args.dataset:
    arch = [1] + args.nits_arch
    nits_model = ConditionalNITS(d=3, start=-args.nits_bound, end=args.nits_bound, monotonic_const=1e-5,
                                 A_constraint=args.constraint, arch=arch, autoregressive=True,
                                 pixelrnn=True, normalize_inverse=True,
                                 final_layer_constraint=args.final_constraint,
                                 softmax_temperature=True).to(device)

tot_params = nits_model.tot_params
loss_op = lambda real, params: nits_loss(real, params, nits_model, discretized=args.discretized)
sample_op = lambda params: nits_sample(params, nits_model)

model = CNN(nr_resnet=5, nr_filters=150,
                 input_channels=input_channels, nits_params=tot_params)
model = model.to(device)

model_name = 'lr_{:.5f}_nits_arch{}_constraint{}_final_constraint{}_softmax_temperature{}'.format(
    args.lr, args.nits_arch, args.constraint, args.final_constraint, args.softmax_temp)

print('model_name:', model_name)

latest_epoch = 0
if args.load_params:
    if args.load_params == 'default':
        for fname in os.listdir(args.save_dir):
            if model_name in fname:
                latest_epoch = max(int(fname.split('_')[-1].split('.')[0]), latest_epoch)

        if latest_epoch == 0:
            raise ValueError('There should have been a saved model!')

        load_fname = '{}/{}_{}.pth'.format(args.save_dir, model_name, latest_epoch)
    else:
        load_fname = args.load_params

    load_part_of_model(model, load_fname)
    print('model parameters loaded')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
test_losses = []

def sample(model):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                out   = model(data, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data


print('starting training')
writes = 0
for epoch in range(latest_epoch, args.max_epochs):
    model.train(True)
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.to(device)
        output = model(input)
        output.retain_grad()
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        if output.grad.isnan().any():
            print('output grad are nan')
            break
        optimizer.step()
        train_loss += loss.detach().cpu().numpy()

        if (batch_idx +1) % args.print_every == 0 :
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss / deno),
                (time.time() - time_)))
            train_loss = 0.
            writes += 1
            time_ = time.time()

    if loss.isnan() or loss.isinf() or output.grad.isnan().any():
        break

    scheduler.step()

    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.to(device)
        input_var = torch.autograd.Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.detach().cpu().numpy()
        del loss, output

    test_losses.append(test_loss / (batch_idx * args.batch_size * np.prod(obs) * np.log(2.)))
    print('test loss : {:4f}, min test loss : {:4f}, lr : {:4e}'.format(
        test_losses[-1],
        np.min(test_losses),
        optimizer.param_groups[0]['lr']
    ))

