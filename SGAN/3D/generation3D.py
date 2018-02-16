# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>
"""

import sys
import os
import imp
import scipy, scipy.misc
import numpy as np
from  scipy.signal import medfilt
import h5py
import matplotlib.pyplot as plt
import time

os.chdir('/home/elaloy/SGANinv/SGAN/3D')
sys.path.append('/home/elaloy/SGANinv/SGAN/3D/training')

from sgan3d import SGAN


model_file = 'fold3Dcat_filters64_npx97_5gL_5dL_epoch16.sgan'
gen_model_folder = '/home/elaloy/SGANinv/SGAN/saved_models/3D'

modelpath = '/home/elaloy/SGANinv/SGAN/saved_models/3D/' + model_file
configpath= '/home/elaloy/SGANinv/SGAN/3D/training/config3D.py'

def main_test_local(): # not shown in the paper, just to check the locality of the changes

    sgan        = SGAN(modelpath)
    c = sgan.config
    
    np.random.seed(0)
    z_sample1=  np.random.uniform(-1.,1., (1, c.nz, c.zx_sample, c.zx_sample, c.zx_sample) )    # the sample for which to p
    np.save('z0.npy',z_sample1)
    
    data = sgan.generate(z_sample1)
    np.save('x0.npy',data)
    scipy.misc.imsave('out0.png', np.array(data[0,0,:,:,1]))
    
    z_sample2=np.array(z_sample1)
    z_sample2[0,0,-1,-1,-1]=-1* z_sample2[0,0,-1,-1,-1]
    data = sgan.generate(z_sample2)
    np.save('x2.npy',data)
    scipy.misc.imsave('out2.png', np.array(data[0,0,:,:,1]))

def main_gen(DoFiltering=False,DoThreshold=False,nsample=10):  
    
    config = imp.load_source('config3D', configpath)
    sgan        = SGAN(modelpath)
    c = config.Config
    
    np.random.seed(2046)

    t_start=time.time()

    z_sample1= np.random.uniform(-1.,1., (nsample, c.nz, c.zx_sample, c.zx_sample,c.zx_sample) )    # the sample for which to plot samples
   
    model = sgan.generate(z_sample1)[:,0,:,:,:]
    
    model = (model+1)*0.5 # Convert from [-1,1] to [0,1]
     
    if DoFiltering==True:
        
        for ii in range(model.shape[0]):
            model[ii,:]=medfilt(model[ii,:], kernel_size=(3,3,3))
    if DoThreshold:

        threshold=0.5
        model[model<threshold]=0
        model[model>=threshold]=1
     
    plt.figure(figsize=(8,8)) 
    plt.imshow(model[0,:,:,1],cmap='gray')
    
    h5_filename="3D_GenSet"
    f = h5py.File(gen_model_folder+'/'+h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=model)
    f.flush()
    f.close()

    print('elapsed time was :',time.time()-t_start)

if __name__=="__main__":
    main_gen()
