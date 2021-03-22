
# -*- coding: utf-8 -*-

"""
Created on Sat May 19 10:04:09 2018

@author: elaloy <elaloy elaloy@sckcen.be>
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from  scipy.signal import medfilt
import sys
import time
import torch
from PIL import Image

work_dir = os.path.dirname(os.path.realpath(__file__))

os.chdir(work_dir)

# Load TI
ti_dir=work_dir
ti_file='ti_2500_1ch.png'
ti_path= ti_dir+'\\'+ti_file
ti = Image.open(ti_path)
ti=np.array(ti)
if len(ti.shape)==2: # the array is 2d, convert to a 3D array
	ti=ti.reshape((ti.shape[0],ti.shape[1],1))
ti = ti.transpose( (2,0,1) )
ti = ti / 128. - 1.0

#%% Check generated models for a given epoch

DoFiltering=False
DoThreshold=False

gpath=work_dir+'/netG.pth' 

cuda=True

from generator import Generator as Generator
netG = Generator(cuda=cuda, gpath=gpath)
netG.eval()

rn_seed=1
np.random.seed(rn_seed)

nz=1
zx=17
zy=17
nrealz=3

znp = np.random.uniform(0,1, (nrealz,nz, zx, zy))*2-1

z = torch.from_numpy(znp).float()

t0=time.time()

if cuda:
    netG.cuda()
    z = z.cuda()
    
model = netG(z)
model=model.detach().cpu().numpy()
model=0.5*(model+1)

if DoFiltering:
    for ii in range(model.shape[0]):
        model[ii, :] = medfilt(model[ii, 0,:,:], kernel_size=(3, 3))

if DoThreshold:
    threshold = 0.5
    model[model < threshold] = 0
    model[model >= threshold] = 1
    
print(time.time()-t0)

m=np.array(model[:,0,:,:])
#%% Plot

DoSaveFig=False
fig = plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.title('Patch of TI')
plt.imshow(ti[0,:513,:513],cmap='gray')
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
    fig.savefig('Benchmark.png',dpi=300)