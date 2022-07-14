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
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('-hi', '--hidden_dim', type=int, default=1024)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=4)
parser.add_argument('-n', '--patience', type=int, default=-1)
parser.add_argument('-ga', '--gamma', type=float, default=1)
parser.add_argument('-pd', '--polyak_decay', type=float, default=1 - 5e-5)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-r', '--rotate', action='store_true')
parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
parser.add_argument('-l', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-p', '--dropout', type=float, default=-1.0)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=False)
parser.add_argument('-bm', '--bound_multiplier', type=float, default=1.0)

args = parser.parse_args()

device = 'cuda:' + args.gpu if args.gpu else 'cpu'

use_batch_norm = False
zero_initialization = True
weight_norm = False
default_patience = 10
if args.dataset == 'gas':
    # training set size: 852,174
    data = gas.GAS()
    default_dropout = 0.1
elif args.dataset == 'power':
    # training set size: 1,659,917
    data = power.POWER()
    default_dropout = 0.4
elif args.dataset == 'miniboone':
    # training set size: 29,556
    data = miniboone.MINIBOONE()
    default_dropout = 0.3
    args.hidden_dim = 128
    args.batch_size = 128
elif args.dataset == 'hepmass':
    # training set size: 315,123
    data = hepmass.HEPMASS()
    default_dropout = 0.3
    default_patience = 3
    args.hidden_dim = 512
    args.batch_size = 1024
elif args.dataset == 'bsds300':
    # training set size: 1,000,000
    data = bsds300.BSDS300()
    default_dropout = 0.2

args.patience = args.patience if args.patience >= 0 else default_patience
args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
print(args)

d = data.trn.x.shape[1]

max_val = max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
min_val = min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

max_val *= args.bound_multiplier
min_val *= args.bound_multiplier

nits_model = NITS(d=d, start=min_val, end=max_val, monotonic_const=1e-5,
                  A_constraint='neg_exp', arch=[1] + args.nits_arch,
                  final_layer_constraint='softmax',
                  add_residual_connections=args.add_residual_connections,
                  normalize_inverse=(not args.dont_normalize_inverse),
                  softmax_temperature=False).to(device)

model = ResMADEModel(
    d=d,
    rotate=args.rotate,
    nits_model=nits_model,
    n_residual_blocks=args.n_residual_blocks,
    hidden_dim=args.hidden_dim,
    dropout_probability=args.dropout,
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
    dropout_probability=args.dropout,
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

print_every = 10 if args.dataset != 'miniboone' else 1
optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

time_ = time.time()
epoch = 0
train_ll = 0.
max_val_ll = -np.inf
patience = args.patience
keep_training = True
while keep_training:
    model.train()
    for i, x in enumerate(create_batcher(data.trn.x, batch_size=args.batch_size)):
        ll = model(x)
        optim.zero_grad()
        (-ll).backward()
        train_ll += ll.detach().cpu().numpy()

        optim.step()
        scheduler.step()
        model.update()

    epoch += 1

    if epoch % print_every == 0:
        # compute train loss
        train_ll /= len(data.trn.x) * print_every
        lr = optim.param_groups[0]['lr']

        with torch.no_grad():
            model.eval()
            val_ll = 0.
            ema_val_ll = 0.
            for i, x in enumerate(create_batcher(data.val.x, batch_size=args.batch_size)):
                x = torch.tensor(x, device=device)
                val_ll += model.model(x).detach().cpu().numpy()
                ema_val_ll += model(x).detach().cpu().numpy()

            val_ll /= len(data.val.x)
            ema_val_ll /= len(data.val.x)

        # early stopping
        if ema_val_ll > max_val_ll + 1e-4:
            patience = args.patience
            max_val_ll = ema_val_ll
        else:
            patience -= 1

        if patience == 0:
            print("Patience reached zero. max_val_ll stayed at {:.3f} for {:d} iterations.".format(max_val_ll, args.patience))
            keep_training = False

        with torch.no_grad():
            model.eval()
            test_ll = 0.
            ema_test_ll = 0.
            for i, x in enumerate(create_batcher(data.tst.x, batch_size=args.batch_size)):
                x = torch.tensor(x, device=device)
                test_ll += model.model(x).detach().cpu().numpy()
                ema_test_ll += model(x).detach().cpu().numpy()

            test_ll /= len(data.tst.x)
            ema_test_ll /= len(data.tst.x)

        fmt_str1 = 'epoch: {:3d}, time: {:3d}s, train_ll: {:.3f},'
        fmt_str2 = ' ema_val_ll: {:.3f}, ema_test_ll: {:.3f},'
        fmt_str3 = ' val_ll: {:.3f}, test_ll: {:.3f}, lr: {:.2e}'

        print((fmt_str1 + fmt_str2 + fmt_str3).format(
            epoch,
            int(time.time() - time_),
            train_ll,
            ema_val_ll,
            ema_test_ll,
            val_ll,
            test_ll,
            lr))

        time_ = time.time()
        train_ll = 0.

    if epoch % (print_every * 10) == 0:
        print(args)
