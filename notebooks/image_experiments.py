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
from nits.discretized_mol import discretized_mix_logistic_loss as unstable_pixelcnn_loss
from nits.discretized_mol import sample_from_discretized_mix_logistic as unstable_pixelcnn_sample

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
                    default='', help='GPU to use')
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=2,
                    help='Every how many epochs to write checkpoint/samples?')
# parser.add_argument('-r', '--load_params', type=str, default=None,
                    # help='Restore training from previous model checkpoint?')
parser.add_argument('-r', '--load_epoch', type=int, default=-1,
                    help='Restore training from previous model checkpoint? If so, which epoch?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=24,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-mc', '--monotonic_const', type=int,
                    default=7, help='Monotonic const parameter')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('-ds', '--discretized', type=bool, default=False,
                    help='Discretized model')
parser.add_argument('-ns', '--nits', type=bool, default=True,
                    help='nits model')
parser.add_argument('-at', '--attention', type=str, default='',
                    help='Attention-based NITS')
parser.add_argument('-ni', '--normalize_inverse', type=bool, default=True,
                    help='apply normalization?')
parser.add_argument('-es', '--extra_string', type=str, default='',
                    help='extra string to clarify experiment')
parser.add_argument('-be', '--bisection_eps', type=int, default=6,
                    help='epsilon accuracy for bisection search')
parser.add_argument('-dq', '--dequantize', type=bool, default=False,
                    help='do we dequantize the pixels? performs uniform dequantization')
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[10,1]',
                   help='Architecture of NITS PNN')
args = parser.parse_args()

if args.gpu:
    device = 'cuda:{}'.format(args.gpu)
else:
    device = 'cpu'

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.extra_string:
    args.extra_string = '_' + args.extra_string

arch_string = str(args.nits_arch).replace(' ', '').replace(',', '_')[1:-1]

model_name = 'nits{}_discretized{}_dequantize{}_attention{}_bseps{}_arch{}{}'.format(args.nits, args.discretized, args.dequantize,
                                                                                     args.attention,
                                                                                     args.bisection_eps,
                                                                                     arch_string,
                                                                                     args.extra_string)
print('model_name:', model_name)
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)

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

    if not args.nits:
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    if not args.nits:
        # loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
        # loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake, bad_loss=True)
        # sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
        # loss_op   = lambda real, fake : unstable_pixelcnn_loss(real, fake, bad_loss=True)
        # sample_op = lambda x : unstable_pixelcnn_sample(x, args.nr_logistic_mix, bad_loss=True)
        loss_op_t = loss_op   = lambda real, fake : unstable_pixelcnn_loss(real, fake, bad_loss=True)
        sample_op = lambda x : unstable_pixelcnn_sample(x, args.nr_logistic_mix, bad_loss=True, quantize=False)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

assert np.isclose(1e-7, np.power(10., -7))

if args.nits:
    print("NITS")
    nits_arch = [10, 1]
    nits_bound = 5
    arch = [1] + nits_arch
    nits_model = ConditionalNITS(d=3, start=-nits_bound, end=nits_bound, monotonic_const=np.power(10., -args.monotonic_const),
                                    A_constraint='neg_exp', arch=arch, autoregressive=True,
                                    # pixelrnn=True, normalize_inverse=True,
                                    pixelrnn=True, normalize_inverse=args.normalize_inverse,
                                    final_layer_constraint='softmax',
                                    # softmax_temperature=True).to(device)
                                    softmax_temperature=False, bisection_eps=np.power(10., -args.bisection_eps)).to(device)
    print("args.discretized", args.discretized)
    if not args.discretized:
        print("CONTINUOUS NITS")
    if args.discretized:
        print("DISCRETIZED NITS")
    loss_op = lambda real, params: cnn_nits_loss(real, params, nits_model, discretized=args.discretized, dequantize=args.dequantize)
    loss_op_t = lambda real, params: cnn_nits_loss(real, params, nits_model, discretized=args.discretized)
    sample_op = lambda params: cnn_nits_sample(params, nits_model)
    n_params = nits_model.tot_params
else:
    print('DEFAULT DISCRETIZED MoL')
    # n_params = 100
    n_params = 160

if args.attention == 'full':
    print("USING FACNN")
    model = FACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params)
elif args.attention == 'snail':
    print("USING SNAIL")
    model = ACNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params)
elif args.attention == '':
    print("USING PixelCNN")
    model = CNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                input_channels=input_channels, n_params=n_params)
model = model.to(device)

if args.load_epoch != -1:
    load_path = '/data/image_model/models/{}_{}.pth'.format(model_name, args.load_epoch)
    print("loading parameters from path =", load_path)
    load_part_of_model(model, load_path)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    with torch.no_grad():
        model.train(False)
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                out   = model(data)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample[:, :, i, j].detach().cpu()
    return data

print('starting training')
best_loss = np.inf
for epoch in range(args.max_epochs):
    model.train(True)
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(train_loader):
        input = input.to(device)
        input = Variable(input)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().numpy()
        if (batch_idx +1) % args.print_every == 0 :
            if not args.nits or args.discretized:
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                train_loss = train_loss / deno
            else:
                deno = args.print_every * args.batch_size
                train_loss = train_loss / deno
                train_loss = ll_to_bpd(-train_loss, dataset=args.dataset)
            print('loss : {:.4f}, time : {:.4f}'.format(
                (train_loss),
                (time.time() - time_)))
            train_loss = 0.
            time_ = time.time()


    # decrease learning rate
    scheduler.step()

    model.eval()
    test_loss = 0.
    with torch.no_grad():
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.to(device)
            input_var = Variable(input)
            output = model(input_var)
            loss = loss_op_t(input_var, output)
            test_loss += loss.detach().cpu().numpy()
            del loss, output

    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    if not args.nits or args.discretized:
        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        test_loss = test_loss / deno
    else:
        deno = batch_idx * args.batch_size
        test_loss = test_loss / deno
        test_loss = ll_to_bpd(-test_loss, dataset=args.dataset)

    best_loss = min(test_loss, best_loss)
    print('model_name: {},  test loss: {:.5f}, best loss: {:.5f}'.format(model_name, test_loss, best_loss))

    if (epoch + 1) % args.save_interval == 0:
        print('sampling...')
        sample_t = sample(model)
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t,'/data/image_model/images/{}_{}.png'.format(model_name, epoch),
                nrow=5, padding=0)

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), '/data/image_model/models/{}_{}.pth'.format(model_name, epoch))
