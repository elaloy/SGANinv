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

os.chdir('/home/elaloy/SGANinv/SGAN/2D')
sys.path.append('/home/elaloy/SGANinv/SGAN/2D/training')

from sgan2d import SGAN

case_study='channelized_aquifer'
#case_study='braided_river'

algo_settings=None
case_study_info=case_study

if case_study=='channelized_aquifer':
    TriCatTI=False
    DoFiltering=True
    DoThreshold=True
    model_file = 'channel_2d_filters64_npx353_5gL_5dL_epoch25.sgan'

if case_study=='braided_river':
    TriCatTI=True
    DoFiltering=False
    DoThreshold=True
    model_file = 'braided_river_filters64_npx129_5gL_5dL_epoch92.sgan'

gen_model_folder = '/home/elaloy/SGANinv/SGAN/saved_models/2D'

modelpath = '/home/elaloy/SGANinv/SGAN/saved_models/2D/'+ model_file
configpath= '/home/elaloy/SGANinv/SGAN/2D/training/'+ 'config.py'

def main_test_local(): # not shown in the paper, just to check the locality of the changes
    config = imp.load_source('config', configpath)
    sgan        = SGAN(modelpath)
    c=config.Config
    
    np.random.seed(0)
    z_sample1= np.random.uniform(-1.,1., (1, c.nz, 10, 10) )
    np.save('z0.npy',z_sample1)
    
    data = sgan.generate(z_sample1)
    np.save('x0.npy',data)
    scipy.misc.imsave('out0.png', np.array(data[0,0,:,:]))
    
    z_sample2=np.array(z_sample1)
    z_sample2[0,0,-1,-1]=-1* z_sample2[0,0,-1,-1]
    data = sgan.generate(z_sample2)
    np.save('x2.npy',data)
    scipy.misc.imsave('out2.png', np.array(data[0,0,:,:]))

def main_gen(DoFiltering=DoFiltering,DoThreshold=DoThreshold,nsample=10,TriCatTI=TriCatTI):  
    config = imp.load_source('config', configpath)
    sgan        = SGAN(modelpath)
    c=config.Config
    
    t_start=time.time()
    np.random.seed(2046) # seed is 2046 for generating the realizations in the paper, seed is 123 for generating the true model in the 2D inversion
    z_sample1= np.random.uniform(-1.,1., (nsample, c.nz, c.zx_sample, c.zx_sample) )   
     
    model = sgan.generate(z_sample1)[:,0,:,:]
    
    model = (model+1)*0.5 # Convert from [-1,1] to [0,1]
   
    if DoFiltering==True:
        for ii in range(model.shape[0]):
            model[ii,:]=medfilt(model[ii,:], kernel_size=(3,3))
            
    if DoThreshold and not(TriCatTI):
        threshold=0.5
        model[model<threshold]=0
        model[model>=threshold]=1
        
    if DoThreshold and TriCatTI:
        model[model<0.334]=0
        model[model>=0.667]=2
        model[np.where((model > 0) & (model < 2))]=1
        model=model/2.0 
    
    print('elapsed_time is: ',time.time()-t_start) 
    plt.figure(figsize=(8,8)) 
    plt.imshow(model[0,:,:],cmap='gray')
    
    h5_filename="2D_Gen_"+case_study_info+'_'+model_file+'_'+str(c.zx_sample)
    f = h5py.File(gen_model_folder+'/'+h5_filename+'.hdf5', mode='w')
    h5dset = f.create_dataset('features', data=model)
    f.flush()
    f.close()
    
if __name__=="__main__":
    main_gen()
