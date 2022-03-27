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
    
def create_batcher(x, batch_size=1):
    idx = 0
    p = torch.randperm(len(x))
    x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx+batch_size], device=device)
        idx += batch_size
        
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, default='gas')
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-r', '--rotate', type=bool, default=False)

args = parser.parse_args(['-g', '6', '-d', 'gas'])

device = 'cuda:' + args.gpu if args.gpu else 'cpu'
print('device:', device)

lr = 1e-3
n_residual_blocks = 4
hidden_dim = 512
use_batch_norm = False
zero_initialization = True
if args.dataset == 'gas':
    data = gas.GAS()
    dropout_probability = 0.1
    nits_arch = [16, 16, 1]
    gamma = 1 - 5e-7
elif args.dataset == 'power':
    data = power.POWER()
    dropout_probability = 0.1
    nits_arch = [16, 16, 1]
    gamma = 1 - 5e-7
elif args.dataset == 'miniboone':
    data = miniboone.MINIBOONE()
    dropout_probability = 0.5
    nits_arch = [16, 16, 1]
    gamma = 1 - 5e-7
elif args.dataset == 'hepmass':
    data = hepmass.HEPMASS()
    dropout_probability = 0.2
    nits_arch = [16, 16, 1]
    gamma = 1 - 5e-7
elif args.dataset == 'bsds300':
    data = bsds300.BSDS300()
    dropout_probability = 0.2
    nits_arch = [16, 16, 1]
    gamma = 1 - 5e-7

d = data.trn.x.shape[1]

max_val = max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
min_val = min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()

nits_model = NITS(d=d, start=min_val, end=max_val, monotonic_const=1e-5,
                             A_constraint='neg_exp', arch=[1] + nits_arch,
                             final_layer_constraint='softmax',
                             softmax_temperature=False).to(device)

model = ResMADEModel(
    d=d, 
    rotate=args.rotate, 
    nits_model=nits_model,
    n_residual_blocks=n_residual_blocks,
    hidden_dim=hidden_dim,
    dropout_probability=dropout_probability,
    use_batch_norm=use_batch_norm,
    zero_initialization=zero_initialization
).to(device)

shadow = ResMADEModel(
    d=d, 
    rotate=args.rotate, 
    nits_model=nits_model,
    n_residual_blocks=n_residual_blocks,
    hidden_dim=hidden_dim,
    dropout_probability=dropout_probability,
    use_batch_norm=use_batch_norm,
    zero_initialization=zero_initialization
).to(device)

max_epochs = 20000
batch_size = 512
optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)
    
model = EMA(model, shadow, decay=0.9995).to(device)

time_ = time.time()
for epoch in range(max_epochs):
    model.train()
    train_ll = 0.
    for i, x in enumerate(create_batcher(data.trn.x, batch_size=batch_size)):
        orig_x = x.cpu().detach().clone()
        ll = model(x)
        optim.zero_grad()
        (-ll).backward()
        train_ll += ll.detach().cpu().numpy()

        optim.step()
        scheduler.step()
        model.update()

    if epoch % 10 == 0:
        # compute train loss
        train_ll /= i * batch_size

        with torch.no_grad():
            model.eval()
            val_ll = 0.
            lr = optim.param_groups[0]['lr']
            for i, x in enumerate(create_batcher(data.val.x, batch_size=batch_size)):
                x = torch.tensor(x, device=device)
                ll = model(x)
                val_ll += ll.detach().cpu().numpy()

            val_ll /= i * batch_size
            fmt_str1 = 'epoch: {:4d}, time: {:.2f}, train_ll: {:.4f},'
            fmt_str2 = ' val_ll: {:.4f}, lr: {:.4e}'

            print((fmt_str1 + fmt_str2).format(
                epoch,
                time.time() - time_,
                train_ll,
                val_ll,
                lr))
            
            time_ = time.time()
            
with torch.no_grad():
    model.eval()
    test_ll = 0.
    for i, x in enumerate(create_batcher(data.tst.x, batch_size=batch_size)):
        x = torch.tensor(x, device=device)
        ll = model(x)
        test_ll += ll.detach().cpu().numpy()
        
    test_ll /= i * batch_size

    print('test_ll: {:4f}'.format(test_ll))