
# -*- coding: utf-8 -*-

"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from  scipy.signal import medfilt
import sys
import time
import torch
from PIL import Image, ImageDraw

#%% Import TI
work_case='braided_river_pytorch'
work_dir='./'+ work_case

ti_dir=work_dir
ti_path= work_dir+'/ti/TI_BR_1ch.png'
ti = Image.open(ti_path)
ti=np.array(ti)
if len(ti.shape)==2: # the array is 2d, convert to a 3D array
	ti=ti.reshape((ti.shape[0],ti.shape[1],1))
ti = ti.transpose( (2,0,1) )
ti = ti / 128. - 1.0

#%% Check generated models for a given epoch
DoFiltering=False
DoThreshold=True
TriCat=True
epoch=299

gpath=work_dir+'/generation/netG_epoch_'+str(epoch)+'.pth' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"
if device==torch.device("cuda"):
    cuda=True
else:
    cuda=False
from braided_river_pytorch.generation.generators import G as Generator
netG = Generator(nc=1, nz=3, ngf=64, gfs=3, ngpu=1,cuda=cuda, gpath=gpath).to(device)
netG.eval()

rn_seed=2043
np.random.seed(rn_seed)

nz=3
zx=10
zy=10
znp = np.random.uniform(0,1, (9,nz, zx, zy))*2-1

z = torch.from_numpy(znp).float().to(device)

t0=time.time()

model = netG(z)
model=model.detach().cpu().numpy()
model=0.5*(model+1)

if DoFiltering==True:
    for i in range(0,model.shape[0]):
        model[i,0,:,:,:]=medfilt(model[i,0,:,:], kernel_size=(3,3))

if DoThreshold==True:
    if TriCat:
        model[model<0.334]=0
        model[model>=0.667]=2
        model[np.where((model > 0) & (model < 2))]=1
        model=model/2.0 
    else: # binary image
        threshold=0.5
        model[model<threshold]=0
        model[model>=threshold]=1
    
print(time.time()-t0)


m=np.array(model[:,0,:,:])

# Quick and dirty figure
DoSaveFig=False
fig = plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.title('Patch of TI')
plt.imshow(ti[0,:289,:289],cmap='gray')
plt.subplot(2,2,2)
plt.title('Realz #1')
plt.imshow(m[1,:,:],cmap='gray')
plt.subplot(2,2,3)
plt.title('Realz #2')
plt.imshow(m[2,:,:],cmap='gray')
plt.subplot(2,2,4)
plt.title('Realz #3')
plt.imshow(m[0,:,:],cmap='gray')
plt.show()
if DoSaveFig:
    fig.savefig('SGAN_realz.png',dpi=300)


