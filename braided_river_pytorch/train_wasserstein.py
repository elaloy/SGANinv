# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""
from __future__ import print_function
import argparse
import os
import sys
from tqdm import tqdm
from time import time

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from nnmodels import G
from nnmodels import D
import numpy as np
import datetime
import platform
import time
from utils import get_texture2D_iter, zx_to_npx, save_tensor2D


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=3, help='number of non-spatial dimensions in latent space z')
parser.add_argument('--zx', type=int, default=5, help='number of grid elements in every spatial dimension of z (for a square 2D or 3D image)')
parser.add_argument('--zx_sample', type=int, default=5, help='zx for saved image snapshots from G')
parser.add_argument('--nc', type=int, default=1, help='number of channeles in original image space')
parser.add_argument('--ngf', type=int, default=64, help='initial number of filters for dis')
parser.add_argument('--ndf', type=int, default=64, help='initial number of filters for gen')
parser.add_argument('--dfs', type=int, default=3, help='kernel size for dis')
parser.add_argument('--gfs', type=int, default=3, help='kernel size for gen')
parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--niter', type=int, default=100, help='number of iterations per training epoch')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--l2_fac', type=float, default=1e-7, help='factor for l2 regularization of the weights in G and D')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./train_data', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1978,help='manual seed')
parser.add_argument('--data_iter', default='from_TI', help='way to get the training samples')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
print("Random Seed: ", opt.manualSeed)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
np.random.seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
device = torch.device("cuda:0" if opt.cuda else "cpu")

torch.backends.cudnn.enabled = True

if platform.system()=='Windows':
    #pass
    torch.backends.cudnn.enabled = False

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
dfs = int(opt.dfs)
gfs = int(opt.gfs)
nc = int(opt.nc)
zx = int(opt.zx)
zy=zx
zz=zx
zx_sample = int(opt.zx_sample)
depth=5
npx=zx_to_npx(zx,depth)
npy=npx
npz=npx
batch_size = int(opt.batchSize)

print(npx)

#home        = os.path.expanduser("~")
if opt.data_iter=='from_TI':
    texture_dir='./ti/'
    data_iter   = get_texture2D_iter(texture_dir, npx=npx, npy=npx,mirror=False, batch_size=batch_size,n_channel=nc)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
netG = G(nc, nz, ngf, gfs, ngpu)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = D(nc, ndf, dfs, ngpu = 1)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=opt.l2_fac)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=opt.l2_fac)

input_noise = torch.rand(batch_size, nz, zx, zx,device=device)*2-1
fixed_noise = torch.rand(1, nz, zx_sample, zx_sample, device=device)*2-1

input = torch.FloatTensor(batch_size, nc, npx, npy)

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_noise, fixed_noise = input_noise.cuda(), fixed_noise.cuda()
    input = input.cuda()
    one=one.cuda()
    mone=mone.cuda()
    
gen_iterations = 0
for epoch in range(opt.nepoch):

    i=0
    while i < opt.niter:

        ############################
        # (1) Update D
        ###########################
        
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < opt.niter:
            j += 1
 
            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = next(data_iter)
            
            i += 1
            
            # train with real
            netD.zero_grad()

            real_cpu = torch.Tensor(data).to(device)
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real.backward(one)
           
             # train with fake

            noise = torch.rand(batch_size, nz, zx, zy,device=device)*2-1
            with torch.no_grad():
                noisev=Variable(noise)

            fake = Variable(netG(noisev).data) # freezes the weights of netG, so that the weights are not updated for the generator
            inputv = fake
            errD_fake = netD(inputv)	# forward the output of netG (Generator) into netD (Discriminator) 
           
            errD_fake.backward(mone)	# Performs backprop on netD and calculates the gradients
            errD = errD_real - errD_fake
            optimizerD.step()
        

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()

        noise = torch.rand(batch_size, nz, zx, zy, device=device)*2-1
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)

        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1
        
        print(i)
        print(gen_iterations)
        
        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, len(data), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        
        
    # do checkpointing
    print('epoch ',epoch,' done')
#    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
#    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

    fake = netG(fixed_noise)
    save_tensor2D(fake.detach().cpu().numpy()[0], '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))
    if epoch >= 299:
        if np.mod((epoch+1),1) ==0:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))


