import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from nits.model import *
from nits.fc_model import *
from maf.datasets import *

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='gas')
parser.add_argument('-g', '--gpu', type=str, default='')

args = parser.parse_args()

device = 'cuda:' + args.gpu if args.gpu else 'cpu'
print('device:', device)

lr = 1e-3
if args.dataset == 'gas':
    data = gas.GAS()
    model_arch = [256, 256]
    nits_arch = [16, 16, 1]
    gamma = 1 - 1e-4
elif args.dataset == 'power':
    data = power.POWER()
    model_arch = [256, 256]
    nits_arch = [16, 16, 1]
    gamma = 1 - 1e-3
elif args.dataset == 'miniboone':
    data = miniboone.MINIBOONE()
    model_arch = [128, 128]
    nits_arch = [16, 16, 1]
    gamma = 1 - 1e-4
elif args.dataset == 'hepmass':
    data = hepmass.HEPMASS()
    model_arch = [256, 256]
    nits_arch = [16, 16, 1]
    gamma = 1 - 1e-4
elif args.dataset == 'bsds300':
    data = bsds300.BSDS300()
    model_arch = [1024, 1024]
    nits_arch = [16, 16, 1]
    gamma = 1 - 1e-3

d = data.trn.x.shape[1]

max_val = max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
min_val = min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

nits_model = NITS(d=d, start=min_val, end=max_val, monotonic_const=1e-4,
                             A_constraint='neg_exp', arch=[1] + nits_arch,
                             final_layer_constraint='softmax',
                             softmax_temperature=True).to(device)

model = ParamModel(arch=[d] + model_arch + [nits_model.n_params]).to(device)

def create_batcher(x, batch_size=1, device=device):
    idx = 0
    p = torch.randperm(len(x))
    x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx+batch_size], device=device)
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device)

max_epochs = 20000
print_every = 500
batch_size = 512
optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

for epoch in range(max_epochs):
    train_ll = 0.
    for i, x in enumerate(create_batcher(data.trn.x, batch_size=batch_size)):
        params = model(x)
        params.retain_grad()
        ll = (nits_model.pdf(x, params) + 1e-5).log().sum()
        loss = -ll

        optim.zero_grad()
        loss.backward()
        train_ll += ll

        assert params.grad.isfinite().all()

        optim.step()

        if i % print_every == print_every - 1:
            train_ll = train_ll / print_every / batch_size
            start = nits_model.forward_(nits_model.start, params)
            end = nits_model.forward_(nits_model.end, params)
            min_end_start = (end - start).min()
            print('train ll: {:.4f}, min end - start: {:.4e}'.format(train_ll, min_end_start))
            train_ll = 0.

    with torch.no_grad():
        val_ll = 0.
        lr = optim.param_groups[0]['lr']
        for x in create_batcher(data.val.x, batch_size=128):
            x = torch.tensor(x, device=device)
            params = model(x)
            ll = nits_model.pdf(x, params).log()
            val_ll += ll.sum()

        print('epoch: {:4d}, val_ll: {:4f}, lr: {:.4e}'.format(epoch, val_ll / data.val.N, lr))

    scheduler.step()

with torch.no_grad():
    test_ll = 0.
    for x in create_batcher(data.tst.x, batch_size=128):
        x = torch.tensor(x, device=device)
        params = model(x)
        ll = nits_model.pdf(x, params).log()
        test_ll += ll.sum()

    print('test_ll: {:4f}'.format(test_ll / data.tst.N))
