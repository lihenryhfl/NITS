import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from nits.model import *
from nits.layer import *
from nits.fc_model import *
from nits.cnn_model import *
from maf.datasets import *
from nits.resmade import ResidualMADE

def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s

def create_batcher(x, batch_size=1):
    idx = 0
    p = torch.randperm(len(x))
    x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx+batch_size], device=device)
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device)


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='gas')
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-b', '--batch_size', type=int, default=512)
parser.add_argument('-hi', '--hidden_dim', type=int, default=512)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=4)
parser.add_argument('-ga', '--gamma', type=float, default=1 - 5e-7)
parser.add_argument('-pd', '--polyak_decay', type=float, default=1 - 5e-4)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-r', '--rotate', type=bool, default=False)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=False)

args = parser.parse_args()

device = 'cuda:' + args.gpu if args.gpu else 'cpu'

print(args)

lr = 5e-4
use_batch_norm = False
zero_initialization = True
weight_norm = False
if args.dataset == 'gas':
    # training set size: 852,174
    data = gas.GAS()
    dropout_probability = 0.1
elif args.dataset == 'power':
    # training set size: 1,659,917
    data = power.POWER()
    dropout_probability = 0.1
elif args.dataset == 'miniboone':
    # training set size: 29,556
    data = miniboone.MINIBOONE()
    dropout_probability = 0.5
elif args.dataset == 'hepmass':
    # training set size: 315,123
    data = hepmass.HEPMASS()
    dropout_probability = 0.5
elif args.dataset == 'bsds300':
    # training set size: 1,000,000
    data = bsds300.BSDS300()
    dropout_probability = 0.2

d = data.trn.x.shape[1]

max_val = max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
min_val = min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

nits_model = NITS(d=d, start=min_val, end=max_val, monotonic_const=1e-5,
                  A_constraint='neg_exp', arch=[1] + args.nits_arch,
                  final_layer_constraint='softmax',
                  add_residual_connections=args.add_residual_connections,
                  softmax_temperature=False).to(device)

model = ResMADEModel(
    d=d,
    rotate=args.rotate,
    nits_model=nits_model,
    n_residual_blocks=args.n_residual_blocks,
    hidden_dim=args.hidden_dim,
    dropout_probability=dropout_probability,
    use_batch_norm=use_batch_norm,
    zero_initialization=zero_initialization,
    weight_norm=weight_norm
).to(device)

shadow = ResMADEModel(
    d=d,
    rotate=args.rotate,
    nits_model=nits_model,
    n_residual_blocks=args.n_residual_blocks,
    hidden_dim=args.hidden_dim,
    dropout_probability=dropout_probability,
    use_batch_norm=use_batch_norm,
    zero_initialization=zero_initialization,
    weight_norm=weight_norm
).to(device)

# initialize weight norm
if weight_norm:
    with torch.no_grad():
        for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
            params = model(x)
            break

model = EMA(model, shadow, decay=args.polyak_decay).to(device)

max_iters = 2000000
print_every = 10
optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

time_ = time.time()
epoch = 0
iters = 0
train_ll = 0.
while iters < max_iters:
    model.train()
    for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
        ll = model(x)
        optim.zero_grad()
        (-ll).backward()
        train_ll += ll.detach().cpu().numpy()

        optim.step()
        scheduler.step()
        model.update()
        iters += 1

    epoch += 1

    if (epoch + 1) % print_every == 0:
        # compute train loss
        train_ll /= len(data.trn.x) * print_every
        lr = optim.param_groups[0]['lr']

        with torch.no_grad():
            model.eval()
            val_ll = 0.
            for i, x in enumerate(create_batcher(data.val.x, batch_size=args.batch_size)):
                x = torch.tensor(x, device=device)
                ll = model(x)
                val_ll += ll.detach().cpu().numpy()

            val_ll /= len(data.val.x)

        with torch.no_grad():
            model.eval()
            test_ll = 0.
            for i, x in enumerate(create_batcher(data.tst.x, batch_size=args.batch_size)):
                x = torch.tensor(x, device=device)
                ll = model(x)
                test_ll += ll.detach().cpu().numpy()

            test_ll /= len(data.tst.x)

        fmt_str1 = 'epoch: {:4d}, time: {:.2f}, train_ll: {:.4f},'
        fmt_str2 = ' val_ll: {:.4f}, test_ll: {:.4f}, lr: {:.4e}'

        print((fmt_str1 + fmt_str2).format(
            epoch + 1,
            time.time() - time_,
            train_ll,
            val_ll,
            test_ll,
            lr))

        time_ = time.time()
        train_ll = 0.

    if (epoch + 1) % (print_every * 10) == 0:
        print(args)
