# -*- coding: utf-8 -*-
"""
A Pytorch implementation in Python 3.6 of the GAN-based probabilistic inversion 
approach by Laloy et al. (2018a). The inversion code is the DREAMzs (ter Braak and Vrugt, 2008; 
Vrugt, 2009; Laloy and Vrugt, 2012) MCMC sampler and the considered toy forward 
problem involves 2D GPR linear tomography in a binary channelized subsurface domain.

@author: Eric Laloy <elaloy@sckcen.be>

Please drop me an email if you have any question and/or if you find a bug in this
program. 

===
Copyright (C) 2018  Eric Laloy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ===                               

References:
    
Laloy, E., Hérault, R., Jacques, D., and Linde, N. 2018a. Training-image based 
    geostatistical inversion using a spatial generative adversarial neural network. 
    Water Resources Research, 54, 381–406. https://doi.org/10.1002/2017WR022148.

Laloy, E., Linde, N., Ruffino, C., Hérault, R., & Jacques, D. 2018b. Gradient-based 
    deterministic inversion of geophysical data with Generative Adversarial Networks: 
    is it feasible? arXiv:1812.09140, https://arxiv.org/abs/1812.09140.                                     
                                                                                                                                                                                                       
"""

import os
import time
import numpy as np

work_dir=os.getcwd()

import mcmc

#% Set random seed and case study

rng_seed=12345

CaseStudy=2
 
if  CaseStudy==0: #100-d correlated gaussian (case study 2 in DREAMzs Matlab code)
    seq=3
    steps=5000
    ndraw=seq*100000
    thin=10
    jr_scale=1.0
    Prior='LHS'

if  CaseStudy==1: #10-d bimodal distribution (case study 3 in DREAMzs Matlab code)
    seq=5
    ndraw=seq*40000
    thin=10
    steps=np.int32(ndraw/(20.0*seq))
    jr_scale=1.0
    Prior='COV'
    
if  CaseStudy==2: # Linear GPR tomography (toy) problem
    seq=8
    ndraw=seq*50000
    thin=1
    steps=200
    jr_scale=0.1
    Prior='LHS'
    
    DoParallel=False # forward model is so quick here that parallel is not necessary
    parallel_jobs=None
 
#% Run the DREAMzs algorithm
if __name__ == '__main__':
    
    start_time = time.time()

    q=mcmc.Sampler(main_dir=work_dir,CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,Prior=Prior,parallel_jobs=seq,steps=steps,
                   parallelUpdate = 0.9,pCR=True,thin=thin,nCR=3,DEpairs=1,pJumpRate_one=0.2,BoundHandling='Reflect',
                   lik_sigma_est=False,DoParallel=DoParallel,jr_scale=jr_scale,rng_seed=rng_seed)
    
    print("Iterating")
    
    tmpFilePath=None # None or: work_dir+'\out_tmp.pkl' for a restart
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar = q.sample(RestartFilePath=tmpFilePath)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
    
