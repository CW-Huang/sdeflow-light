import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.plotting import get_grid
from lib.flows.elemwise import LogitTransform
from lib.models.unet import UNet
from lib.helpers import logging, create
from tensorboardX import SummaryWriter
import json


_folder_name_keys = ['dataset', 'real', 'debias', 'batch_size', 'lr', 'num_iterations']


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='mnist')
    parser.add_argument('--dataroot', type=str, default='~/.datasets')
    parser.add_argument('--saveroot', type=str, default='~/.saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0', type=float, default=1.0,
                        help='integration time')
    parser.add_argument('--vtype', type=str, choices=['rademacher', 'gaussian'], default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000)

    # model
    parser.add_argument('--real', type=eval, choices=[True, False], default=True,
                        help='transforming the data from [0,1] to the real space using the logit function')
    parser.add_argument('--debias', type=eval, choices=[True, False], default=False,
                        help='using non-uniform sampling to debias the denoising score matching loss')

    return parser.parse_args()


args = get_args()
folder_tag = 'sde-flow'
folder_name = '-'.join([str(getattr(args, k)) for k in _folder_name_keys])
create(args.saveroot, folder_tag, args.expname, folder_name)
folder_path = os.path.join(args.saveroot, folder_tag, args.expname, folder_name)
print_ = lambda s: logging(s, folder_path)
print_(f'folder path: {folder_path}')
print_(str(args))
with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
    out.write(json.dumps(args.__dict__, indent=4))
writer = SummaryWriter(folder_path)


if args.dataset == 'mnist':
    input_channels = 1
    input_height = 28
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args.dataroot, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args.dataroot, train=False,
                                         download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=True, num_workers=2)

    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )

elif args.dataset == 'cifar':
    input_channels = 3
    input_height = 32
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataroot, 'cifar10'), train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataroot, 'cifar10'), train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=True, num_workers=2)

    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )
else:
    raise NotImplementedError


T = torch.nn.Parameter(torch.FloatTensor([args.T0]), requires_grad=False)

inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=args.vtype, debias=args.debias)


cuda = torch.cuda.is_available()
if cuda:
    gen_sde.cuda()

optim = torch.optim.Adam(gen_sde.parameters(), lr=args.lr)

logit = LogitTransform(alpha=0.05)
if args.real:
    reverse = logit.reverse
else:
    reverse = None


@torch.no_grad()
def evaluate():
    test_bpd = list()
    gen_sde.eval()
    for x_test, _ in testloader:
        if cuda:
            x_test = x_test.cuda()
        x_test = x_test * 255 / 256 + torch.rand_like(x_test) / 256
        if args.real:
            x_test, ldj = logit.forward_transform(x_test, 0)
            elbo_test = gen_sde.elbo_random_t_slice(x_test)
            elbo_test += ldj
        else:
            elbo_test = gen_sde.elbo_random_t_slice(x_test)

        test_bpd.extend(- (elbo_test.data.cpu().numpy() / dimx) / np.log(2) + 8)
    gen_sde.train()
    test_bpd = np.array(test_bpd)
    return test_bpd.mean(), test_bpd.std() / len(testloader.dataset.data) ** 0.5


if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
    gen_sde, optim, not_finished, count = torch.load(os.path.join(folder_path, 'checkpoint.pt'))
else:
    not_finished = True
    count = 0
    writer.add_scalar('T', gen_sde.T.item(), count)
    writer.add_image('samples',
                     get_grid(gen_sde, input_channels, input_height, n=4,
                              num_steps=args.num_steps, transform=reverse),
                     0)
while not_finished:
    for x, _ in trainloader:
        if cuda:
            x = x.cuda()
        x = x * 255 / 256 + torch.rand_like(x) / 256
        if args.real:
            x, _ = logit.forward_transform(x, 0)

        loss = gen_sde.dsm(x).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        count += 1
        if count == 1 or count % args.print_every == 0:
            writer.add_scalar('loss', loss.item(), count)
            writer.add_scalar('T', gen_sde.T.item(), count)

            bpd, std_err = evaluate()
            writer.add_scalar('bpd', bpd, count)
            writer.add_scalar('bpd_std_err', std_err, count)
            print_(f'Iteration {count} \tBPD {bpd}')

        if count >= args.num_iterations:
            not_finished = False
            print_('Finished training')
            break

        if count % args.sample_every == 0:
            gen_sde.eval()
            writer.add_image('samples',
                             get_grid(gen_sde, input_channels, input_height, n=4,
                                      num_steps=args.num_steps, transform=reverse),
                             count)
            gen_sde.train()

        if count % args.checkpoint_every == 0:
            torch.save([gen_sde, optim, not_finished, count], os.path.join(folder_path, 'checkpoint.pt'))
