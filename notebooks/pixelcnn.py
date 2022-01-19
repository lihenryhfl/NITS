import time
import os, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from nits.pixelcnn_model import *
from nits.model import NITS, ConditionalNITS
from nits.discretized_mol import discretized_nits_loss, nits_sample
from PIL import Image

import matplotlib.pyplot as plt

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
parser.add_argument('-g', '--gpu', type=str,
                    default='', help='Location for the dataset')
parser.add_argument('-i', '--data_dir', type=str,
                    default='/data/pixelcnn/data/', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/data/pixelcnn/models/',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be cifar / mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')

# pixelcnn model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=(1 - 5e-6),
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')

# nits model
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[8, 8, 1]',
                    help='Architecture of NITS model')
parser.add_argument('-nb', '--nits_bound', type=float, default=5.,
                    help='Upper and lower bound of NITS model')
parser.add_argument('-c', '--constraint', type=str, default='neg_exp',
                    help='Upper and lower bound of NITS model')
parser.add_argument('-fc', '--final_constraint', type=str, default='softmax',
                    help='Upper and lower bound of NITS model')


args = parser.parse_args()

device = 'cuda:' + args.gpu if args.gpu else 'cpu'
print('device:', device)

# HOUSEKEEPING

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'lr_{:.5f}_nr_resnet{}_nr_filters{}_nits_arch{}_constraint{}_final_constraint{}'.format(
    args.lr, args.nr_resnet, args.nr_filters, args.nits_arch, args.constraint, args.final_constraint)
if os.path.exists(os.path.join('runs_test', model_name)):
    shutil.rmtree(os.path.join('runs_test', model_name))

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
                      A_constraint=args.constraint, arch=arch, final_layer_constraint=args.final_constraint).to(device)
elif 'cifar' in args.dataset:
    arch = [1] + args.nits_arch
    nits_model = ConditionalNITS(d=3, start=-args.nits_bound, end=args.nits_bound, monotonic_const=1e-5,
                                 A_constraint=args.constraint, arch=arch, autoregressive=True,
                                 pixelrnn=True, normalize_inverse=True, final_layer_constraint=args.final_constraint).to(device)
tot_params = nits_model.tot_params
loss_op = lambda real, params: discretized_nits_loss(real, params, nits_model)
sample_op = lambda params: nits_sample(params, nits_model)

# INITIALIZE PIXELCNN MODEL
model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                 input_channels=input_channels, nr_logistic_mix=tot_params, num_mix=1)
model = model.to(device)

if args.load_params:
    load_part_of_model(model, args.load_params)
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
test_losses = []

def sample(model):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = Variable(data)
                out   = model(data_v, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data


print('starting training')
writes = 0
for epoch in range(args.max_epochs):
    model.train(True)
    torch.cuda.synchronize()
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.to(device)
        input = Variable(input)
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

    # decrease learning rate
    scheduler.step()

    torch.cuda.synchronize()
    model.eval()
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(test_loader):
        input = input.to(device)
        input_var = Variable(input)
        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.detach().cpu().numpy()
        del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    print('test loss : {:4f}, lr : {:4e}'.format(test_loss / deno, optimizer.param_groups[0]['lr']))
    test_losses.append(test_loss / deno)

    if (epoch + 1) % args.save_interval == 0:
        torch.save(model.state_dict(), '{}/{}_{}.pth'.format(args.save_dir, model_name, epoch))
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t,'/data/pixelcnn/images/{}_{}.png'.format(model_name, epoch),
                nrow=5, padding=0)

