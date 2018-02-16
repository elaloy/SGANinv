# -*- coding: utf-8 -*-
"""
A Python 2.7 implementation of the DREAMzs MCMC sampler (Vrugt et al., 2009, 
Laloy and Vrugt, 2012) tailored to the geostatistical inverse problems considered in 
Laloy et al. (2018) where a generative adversarial network (GAN) is used to construct a
(very) low dimensional parameterization of the subsurface domain. Randomly sampling 
the so-defined low-dimensional (latent) parameters yield consistent subsurface models 
thereby enabling MCMC inversion within this latent space.
 
This DREAMzs implementation is based on the 2013 DREAMzs Matlab code (version 1.5, 
licensed under GPL3) written by Jasper Vrugt. 

Version 0.0 - October 2017.

@author: Eric Laloy <elaloy@sckcen.be>

Please drop me an email if you have any question and/or if you find a bug in this
program. 

Also, if you find this code useful please make sure to cite the paper for which it 
has been developed (Laloy et al., 2018).

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
    
Jetchev, N., Bergmann, U., & Vollgraf, R. (2016, 1 December). Texture synthesis with 
    spatial generative adversarial networks, arXiv:1611.08207v2 [cs.CV].
    
Laloy, E., Hérault, R., Jacques, D., & Linde, N. (2018). Training-image based
    geostatistical inversion using a spatial generative adversarial neural network.
    Water Resources Research, 54. https://doi.org/10.1002/2017WR022148
    
Laloy, E., Vrugt, J.A., High-dimensional posterior exploration of hydrologic models      
    using multiple-try DREAMzs and high-performance computing, Water Resources Research, 
    48, W01526, doi:10.1029/2011WR010608, 2012.
    
ter Braak, C.J.F., Vrugt, J.A., Differential Evolution Markov Chain with snooker updater 
    and fewer chains, Statistics and Computing, 18, 435–446, doi:10.1007/s11222-008-9104-9,
	2008.
    
Vrugt, J. A., C.J.F. ter Braak, C.G.H. Diks, D. Higdon, B.A. Robinson, and J.M. Hyman,
    Accelerating Markov chain Monte Carlo simulation by differential evolution with
    self-adaptive randomized subspace sampling, International Journal of Nonlinear Sciences
    and Numerical Simulation, 10(3), 273-290, 2009.                                         
                                                                                                                                                                                                       
"""

import os
import time
import numpy as np
import shutil

work_dir=r'/home/elaloy/SGANinv/inversion'
os.chdir(work_dir)
import mcmc

#% Set rng_seed and case study

rng_seed=123456789 # np.random.seed(np.floor(time.time()).astype('int'))

CaseStudy=3
 
if  CaseStudy==0: #100-d correlated gaussian (case study 2 in DREAMzs Matlab code)
    seq=3
    steps=5000
    ndraw=seq*100000
    thin=10
    jr_scale=1.0
    Prior='LHS'
    nCR=3
if  CaseStudy==1: #10-d bimodal distribution (case study 3 in DREAMzs Matlab code)
    seq=5
    ndraw=seq*40000
    thin=10
    steps=np.int32(ndraw/(20.0*seq))
    jr_scale=1.0
    Prior='COV'
    nCR=3
if  CaseStudy==2: # 2D SGAN-based inversion
    seq=8
    ndraw=seq*40000
    thin=1
    steps=100
    jr_scale=0.5
    Prior='LHS'
    n_ptrials=seq
    nCR=25
    MakeNewDir=True
if  CaseStudy==3: # 3D SGAN-based inversion
    seq=16
    ndraw=seq*50000
    thin=1
    steps=100
    jr_scale=1
    Prior='LHS'
    n_ptrials=seq
    nCR=54
    MakeNewDir=True
    
    if MakeNewDir==True:
        if CaseStudy==3:
            src_dir=work_dir+'/forward_setup_30x61x61'
        if CaseStudy==2:
            src_dir=work_dir+'/forward_setup_125x125'
        for i in range(1,n_ptrials+1):
            dst_dir=work_dir+'/forward_setup_'+str(i)
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir,dst_dir)


if __name__ == '__main__':
    
    start_time = time.time()

    q=mcmc.Sampler(CaseStudy=CaseStudy,seq=seq,ndraw=ndraw,
                   Prior=Prior,parallel_jobs=seq,steps=steps,parallelUpdate = 0.95,pCR=False,
                   thin=thin,nCR=nCR,DEpairs=3,pJumpRate_one=0.2,BoundHandling='Reflect',
                   lik_sigma_est=False,savemodout=False,DoParallel=True,jr_scale=jr_scale,
                   rng_seed=rng_seed)
    print("Iterating")
    
    tmpFilePath=None   # set tmpFilePath to the following instead if it is a restart: work_dir+'\out_tmp.pkl'
    
    Sequences, Z, OutDiag, fx, MCMCPar, MCMCVar = q.sample(RestartFilePath=tmpFilePath)
    
    end_time = time.time()
    
    print("This sampling run took %5.4f seconds." % (end_time - start_time))
     
    
