# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>
"""
import sys
import os
#import imp
import numpy as np
from  scipy.signal import medfilt
#import h5py

current_dir=os.getcwd()

sys.path.append('/home/elaloy/SGANinv/SGAN/2D/training')

from sgan2D import SGAN

case_study='channelized_aquifer'

algo_settings=None
case_study_info=case_study

model_file = 'channel_filters64_npx353_5gL_5dL_epoch16.sgan' #case_study='channelized_aquifer'
#model_file = 'braided_river_filters64_npx129_5gL_5dL_epoch92.sgan'        #case_study='braided_river'

modelpath = '/home/elaloy/SGANinv/SGAN/saved_models/2D/' + model_file

sgan = SGAN(modelpath)

def gen_from_noise(z_sample,DoFiltering=True,DoThreshold=True,TriCatTI=False):  
 
    model = sgan.generate(z_sample)[:,0,:,:]
    
    model = (model+1)*0.5
   
    if DoFiltering==True:
        
        for ii in xrange(model.shape[0]):
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
     
    return model

