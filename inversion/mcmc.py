# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>
"""
from __future__ import print_function

import numpy as np
import numpy.matlib as matlib
try:
    import cPickle as pickle
except:
    import pickle

import time

import h5py

import scipy.io as sio

from mcmc_func import* # This imports both all Dream_zs and inverse problem-related functions

from attrdict import AttrDict

MCMCPar=AttrDict()

MCMCVar=AttrDict()

Measurement=AttrDict()

OutDiag=AttrDict()

Extra=AttrDict()

class Sampler:

    def __init__(self, CaseStudy=3,seq = 3,ndraw=10000,thin = 1,  nCR = 3, 
                 DEpairs = 3, parallelUpdate = 0.9, pCR=True,k=10,pJumpRate_one=0.2,
                 steps=100,savemodout=False, saveout=True,save_tmp_out=True,Prior='LHS',
                 DoParallel=True,eps=5e-2,BoundHandling='Reflect',
                 lik_sigma_est=False,parallel_jobs=4,jr_scale=1.0,rng_seed=1):
                
        self.CaseStudy=CaseStudy
        MCMCPar.seq = seq
        MCMCPar.ndraw=ndraw
        MCMCPar.thin=thin
        MCMCPar.nCR=nCR
        MCMCPar.DEpairs=DEpairs
        MCMCPar.parallelUpdate=parallelUpdate
        MCMCPar.Do_pCR=pCR
        MCMCPar.k=k
        MCMCPar.pJumpRate_one=pJumpRate_one
        MCMCPar.steps=steps
        MCMCPar.savemodout=savemodout
        MCMCPar.saveout=saveout  
        MCMCPar.save_tmp_out=save_tmp_out  
        MCMCPar.Prior=Prior
        MCMCPar.DoParallel=DoParallel
        MCMCPar.eps = eps
        MCMCPar.BoundHandling = BoundHandling
        MCMCPar.jr_scale=jr_scale
        MCMCPar.lik_sigma_est=lik_sigma_est
        Extra.n_jobs=parallel_jobs
        np.random.seed(rng_seed)
        MCMCPar.rng_seed=rng_seed
 
        # Set SGAN-generator, load measurement data and set forward model stuff
        if self.CaseStudy==3: #3D transient hydraulic tomography case study
            self.ndim=27
            MCMCPar.n=self.ndim
            from sgan3D_gen_model_from_noise import gen_from_noise
            DNN=AttrDict()         
            DNN.gen_from_noise=gen_from_noise
            DNN.npx=3
            DNN.nz=1
            DNN.gen_from_noise=gen_from_noise
            self.DNN=DNN
            
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-1
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+1
            
            if MCMCPar.lik_sigma_est==True:
                MCMCPar.n= MCMCPar.n+1
                MCMCPar.lb=np.hstack((np.zeros((1,1))+np.log(0.001),MCMCPar.lb))
                MCMCPar.ub=np.hstack((np.zeros((1,1))+np.log(0.1),MCMCPar.ub))
            ModelName='forward_model_flow'
            MCMCPar.lik=2 #set to 2 if no direct conditioning data are used, set to 4 otherwise
            Extra.SimType=3 # 3D transient hydraulic tomography
            Extra.DomainGeom='3D'
            Extra.crop=True
            # Crop from 65 x 65 x 65 to 30 x 61 x 61
            Extra.crop_istart=17;Extra.crop_iend=47
            Extra.crop_jstart=2;Extra.crop_jend=63
            Extra.crop_kstart=2;Extra.crop_kend=63
            
            tmp_meas = sio.loadmat('MeasData_catfold_3D_noise0.01')
           
            with h5py.File('3D_true_model'+'.hdf5', 'r') as fid:
                Extra.trueK = np.array(fid['features'])
            Extra.idx=tmp_meas['iim'].T-1
            Measurement.MeasData=tmp_meas['simc'].T
            Measurement.MeasData=Measurement.MeasData[0,Extra.idx]
            Measurement.N=1568
            Extra.cdt=tmp_meas['cdt']
            del tmp_meas
            MCMCPar.AdaptJR=True
            MCMCPar.TargetAR=2
            MCMCPar.jr_scale_min=0.1
            MCMCPar.AdaptSigma=True
            if MCMCPar.AdaptSigma==True:
                Measurement.Sigma0=0.1
                Measurement.Sigma1=0.01
                Measurement.Sigma=np.array(Measurement.Sigma0)
                Measurement.Sigma_coeff=-0.00001
            else:
                Measurement.Sigma=0.01 
                
            if MCMCPar.lik==4:
                cobs = sio.loadmat('coord_obs3D')['coord_obs'] 
                cwell= sio.loadmat('coord_wel3D')['coord_wel']  
                Extra.jdx=np.vstack((cwell-1,cobs-1))
                with open('condset3D'+'.pkl','rb') as f:
                    Extra.condset=pickle.load(f)
                    
        elif self.CaseStudy==2:
            self.ndim=25
            MCMCPar.n=self.ndim
            from sgan2D_gen_model_from_noise import gen_from_noise
            DNN=AttrDict()         
            DNN.gen_from_noise=gen_from_noise
            DNN.npx=5
            DNN.nz=1
            DNN.gen_from_noise=gen_from_noise
            self.DNN=DNN
            
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-1
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+1
            Measurement.N=49
            ModelName='forward_model_flow'
            MCMCPar.lik=4 #set to 2 if no direct conditioning data are used, set to 4 otherwise
            Extra.SimType=1 # Steady-state flow
            Extra.DomainGeom='2D'
            Extra.crop=True
            Extra.crop_istart=2;Extra.crop_iend=127
            Extra.idx=(np.array([ [20,20], [35,35], [65,65], [80,80], [20,80], [80,20], [35,65], [65,35], [51,51], 
                [51,65], [51,35], [35,51], [65,51], [20,35], [20,51], [20,65], [35,20], [51,20],
                [65,20], [80,35], [80,51], [80,65], [35,80], [51,80], [65,80], [5,5], [5,20], 
                [5,35], [5,51], [5,65], [5,80], [5,95], [20,5], [35,5], [51,5], [65,5], [80,5],
                [95,5], [95,20], [95,35], [95,51], [95,65], [95,80], [95,95], [20,95], [35,95],
                [51,95], [65,95], [80,95]])*1.25).astype('int')           
            
            with open('MeasData_channel_2D_noise0.01'+'.pkl','rb') as f:
                tmp=pickle.load(f)
            Measurement.MeasData=tmp['MeasData']
            
            if MCMCPar.lik==4:
                with open('condset2D'+'.pkl','rb') as f:
                    Extra.condset=pickle.load(f)
			
            
            MCMCPar.AdaptSigma=True
            if MCMCPar.AdaptSigma==True:
                Measurement.Sigma0=0.20
                Measurement.Sigma1=0.01
                Measurement.Sigma=np.array(Measurement.Sigma0)
                Measurement.Sigma_coeff=-0.00005
            else:
                Measurement.Sigma=tmp['Sigma']

        elif self.CaseStudy==1:  
            self.ndim=10
            MCMCPar.n=self.ndim
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-100
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+100
            MCMCPar.BoundHandling=None  
            Measurement.N=1
            ModelName='theoretical_case_bimodal_mvn'
            MCMCPar.lik=1
            Extra.cov1=np.eye(MCMCPar.n)
            Extra.cov2=np.eye(MCMCPar.n)
            Extra.mu1=np.zeros((MCMCPar.n))-5
            Extra.mu2=np.zeros((MCMCPar.n))+5
            
            
        elif self.CaseStudy==0:
            self.ndim=100
            MCMCPar.n=self.ndim
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-5
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+15
            MCMCPar.BoundHandling=None
            Measurement.N=1
            ModelName='theoretical_case_mvn'
            MCMCPar.lik=0

            A = 0.5 * np.eye(MCMCPar.n) + 0.5 * np.ones(MCMCPar.n)
            cov=np.zeros((MCMCPar.n,MCMCPar.n))
            # Rescale to variance-covariance matrix of interest
            for i in xrange (0,MCMCPar.n):
                for j in xrange (0,MCMCPar.n):
                    cov[i,j] = A[i,j] * np.sqrt((i+1) * (j+1))
            Extra.C=cov
            Extra.invC = np.linalg.inv(cov)
            
        else:
            self.ndim=1
            MCMCPar.n=self.ndim
            MCMCPar.lb=np.zeros((1,MCMCPar.n))
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+1
            MCMCPar.BoundHandling=None
            Measurement.N=1
            ModelName=None
            MCMCPar.lik=1

        MCMCPar.m0=20*MCMCPar.n
        
        self.MCMCPar=MCMCPar
        self.Measurement=Measurement
        self.Extra=Extra
        self.ModelName=ModelName
       
    def _init_sampling(self):
        
        Iter=self.MCMCPar.seq
        iteration=2
        iloc=0
        T=0
        
        if self.MCMCPar.Prior=='StandardNormal':
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
        elif self.MCMCPar.Prior=='Normal':
            Zinit=np.random.multivariate_normal(self.MCMCPar.pmu+np.zeros((MCMCPar.n)),np.eye(self.MCMCPar.n)*self.MCMCPar.psd,n)
        elif self.MCMCPar.Prior=='COV': # generate initial opluation from standard normal distribution but the model returns posterior density directly
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
        else: # Uniform prior, LHS sampling
            Zinit=lhs(self.MCMCPar.lb,self.MCMCPar.ub,self.MCMCPar.m0+self.MCMCPar.seq)
        
        self.MCMCPar.CR=np.cumsum((1.0/self.MCMCPar.nCR)*np.ones((1,self.MCMCPar.nCR)))
        Nelem=np.floor(self.MCMCPar.ndraw/self.MCMCPar.seq)++self.MCMCPar.seq*2
        OutDiag.CR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.nCR+1))
        OutDiag.AR=np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
        OutDiag.AR[0,:] = np.array([self.MCMCPar.seq,-1])
        OutDiag.R_stat = np.zeros((np.int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.n+1))
        pCR = (1.0/self.MCMCPar.nCR) * np.ones((1,self.MCMCPar.nCR))
        
        # Calculate the actual CR values based on pCR
        CR,lCR = GenCR(self.MCMCPar,pCR)       
        
        if self.MCMCPar.savemodout:
            self.fx = np.zeros((self.Measurement.N,np.int(np.floor(self.MCMCPar.ndraw/self.MCMCPar.thin))))
            MCMCVar.m_func = self.MCMCPar.seq     
        
        self.Sequences = np.zeros((np.int(np.floor(Nelem/self.MCMCPar.thin)),self.MCMCPar.n+2,self.MCMCPar.seq))
           
        self.MCMCPar.Table_JumpRate=np.zeros((self.MCMCPar.n,self.MCMCPar.DEpairs))
        for zz in xrange(0,self.MCMCPar.DEpairs):
            self.MCMCPar.Table_JumpRate[:,zz] = 2.38/np.sqrt(2 * (zz+1) * np.linspace(1,self.MCMCPar.n,self.MCMCPar.n).T)
        
        # Change steps to make sure to get nice iteration numbers in first loop
        self.MCMCPar.steps = self.MCMCPar.steps - 1
        
        self.Z = np.zeros((np.floor(self.MCMCPar.m0 + self.MCMCPar.seq * (self.MCMCPar.ndraw - self.MCMCPar.m0) / (self.MCMCPar.seq * self.MCMCPar.k)).astype('int64')+self.MCMCPar.seq*100,self.MCMCPar.n+2))
        self.Z[:self.MCMCPar.m0,:self.MCMCPar.n] = Zinit[:self.MCMCPar.m0,:self.MCMCPar.n]

        X = Zinit[self.MCMCPar.m0:(self.MCMCPar.m0+self.MCMCPar.seq),:self.MCMCPar.n]
        del Zinit

        # Run forward model, if any this is done in parallel
        if  self.CaseStudy > 1:
            if self.MCMCPar.lik_sigma_est==False:
                fx0, Extra.dcp = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
            else:
                fx0, Extra.dcp = RunFoward(X[:,1:],self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
        else:
            fx0, _ = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
        
        # Compute likelihood from simulated data    
        of,log_p = CompDensity(X,fx0,self.MCMCPar,self.Measurement,self.Extra)

        X = np.concatenate((X,of,log_p),axis=1)
        Xfx = fx0
        
        if self.MCMCPar.savemodout==True:
            self.fx=fx0
        else:
            self.fx=None

        self.Sequences[0,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))

        # Save N_CR in memory
       
        OutDiag.CR[0,:MCMCPar.nCR+1] = np.concatenate((np.array([Iter]).reshape((1,1)),pCR),axis=1)
        delta_tot = np.zeros((1,self.MCMCPar.nCR))

        # Compute the R-statistic of Gelman and Rubin
        
        OutDiag.R_stat[0,:self.MCMCPar.n+1] = np.concatenate((np.array([Iter]).reshape((1,1)),GelmanRubin(self.Sequences[:1,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)),axis=1)
      
        self.OutDiag=OutDiag
        
        # Also return the necessary variable parameters
        MCMCVar.m=self.MCMCPar.m0
        MCMCVar.Iter=Iter
        MCMCVar.iteration=iteration
        MCMCVar.iloc=iloc; MCMCVar.T=T; MCMCVar.X=X
        MCMCVar.Xfx=Xfx; MCMCVar.CR=CR; MCMCVar.pCR=pCR
        MCMCVar.lCR=lCR; MCMCVar.delta_tot=delta_tot
        self.MCMCVar=MCMCVar
        
        if self.MCMCPar.save_tmp_out==True:
            with open('out_tmp'+'.pkl','wb') as f:
                 pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                 'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                 'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                 'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

    def sample(self,RestartFilePath=None):
        
        if not(RestartFilePath is None):
            print('This is a restart')
            with open(RestartFilePath, 'rb') as fin:
                tmp_obj = pickle.load(fin)
            self.Sequences=tmp_obj['Sequences']
            self.Z=tmp_obj['Z']
            self.OutDiag=tmp_obj['OutDiag']
            self.fx=tmp_obj['fx']
            self.MCMCPar=tmp_obj['MCMCPar']
            self.MCMCVar=tmp_obj['MCMCVar']
            self.Measurement=tmp_obj['Measurement']
            self.ModelName=tmp_obj['ModelName']
            self.Extra=tmp_obj['Extra']
            del tmp_obj
            
            # The following 12 lines are for case studies > 1 only ((not for the theoretical case studies 0 and 1)
            self.CaseStudy=3
            DNN=AttrDict()     
            if self.CaseStudy==3:
                from sgan3D_gen_model_from_noise import gen_from_noise
                DNN.npx=3
            elif self.CaseStudy==2:
                from sgan2D_gen_model_from_noise import gen_from_noise
                DNN.npx=5    
            DNN.gen_from_noise=gen_from_noise
            DNN.nz=1
            DNN.gen_from_noise=gen_from_noise
            self.DNN=DNN

#            self.MCMCPar.ndraw = 2 * self.MCMCPar.ndraw
            
            # reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
#            #extend Sequences, Z, OutDiag.AR,OutDiag.Rstat and OutDiag.CR
#            self.Sequences=np.concatenate((self.Sequences,np.zeros((self.Sequences.shape))),axis=0)
#            self.Z=np.concatenate((self.Z,np.zeros((self.Z.shape))),axis=0)
#            self.OutDiag.AR=np.concatenate((self.OutDiag.AR,np.zeros((self.OutDiag.AR.shape))),axis=0)
#            self.OutDiag.R_stat=np.concatenate((self.OutDiag.R_stat,np.zeros((self.OutDiag.R_stat.shape))),axis=0)
#            self.OutDiag.CR=np.concatenate((self.OutDiag.CR,np.zeros((self.OutDiag.CR.shape))),axis=0)
#             
             
        else:
            self._init_sampling()
            
        # Main sampling loop  
        print('Iter =',self.MCMCVar.Iter)
        while self.MCMCVar.Iter < self.MCMCPar.ndraw:
            
            # Check that exactly MCMCPar.ndraw are done (uneven numbers this is impossible, but as close as possible)
            if (self.MCMCPar.steps * self.MCMCPar.seq) > self.MCMCPar.ndraw - self.MCMCVar.Iter:
                # Change MCMCPar.steps in last iteration 
                self.MCMCPar.steps = np.ceil((self.MCMCPar.ndraw - self.MCMCVar.Iter)/np.float(self.MCMCPar.seq)).astype('int64')
                
            # Initialize totaccept
            totaccept = 0
            
            # Loop a number of times before calculating convergence diagnostic, etc.
            for gen_number in xrange(0,self.MCMCPar.steps):
                
                # Update T
                self.MCMCVar.T = self.MCMCVar.T + 1
                
                # Define the current locations and associated log-densities
                xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])
                log_p_xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])

                # Without replacement draw rows from Z for proposal creation
                R=np.random.permutation(self.MCMCVar.m)
                R=R[0:2 * self.MCMCPar.DEpairs * self.MCMCPar.seq]
                Zoff = np.array(self.Z[R,:self.MCMCPar.n])
             
        
                # Determine to do parallel direction or snooker update
                if (np.random.rand(1) <= self.MCMCPar.parallelUpdate):
                    Update = 'Parallel_Direction_Update'
                else:
                    Update = 'Snooker_Update'

                # Generate candidate points (proposal) in each chain using either snooker or parallel direction update
                xnew,self.MCMCVar.CR[:,gen_number] ,alfa_s = DreamzsProp(xold,Zoff,self.MCMCVar.CR[:,gen_number],self.MCMCPar,Update)
    
                # Get simulated data (done in parallel)
                if  self.CaseStudy > 1:
                    if self.MCMCPar.lik_sigma_est==False:
                        fx_new, Extra.dcp = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
                    else:
                        fx_new, Extra.dcp = RunFoward(xnew[:,1:],self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
                else:
                    fx_new, _ = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                
                if self.MCMCPar.AdaptSigma==True:
                    self.Measurement.Sigma=np.array((self.Measurement.Sigma0-self.Measurement.Sigma1)*np.exp(self.Measurement.Sigma_coeff*self.MCMCVar.Iter)+self.Measurement.Sigma1)
                 
                # Compute the likelihood of each proposal in each chain
                of_xnew,log_p_xnew = CompDensity(xnew,fx_new,self.MCMCPar,self.Measurement,self.Extra)
 
    
                # Calculate the Metropolis ratio
                accept = Metrop(self.MCMCPar,xnew,log_p_xnew,xold,log_p_xold,alfa_s)

                # And update X and the model simulation

                idx_X= np.argwhere(accept==1);idx_X=idx_X[:,0]
                
                if not(idx_X.size==0):
                     
                    self.MCMCVar.X[idx_X,:] = np.concatenate((xnew[idx_X,:],of_xnew[idx_X,:],log_p_xnew[idx_X,:]),axis=1)
                    self.MCMCVar.Xfx[idx_X,:] = fx_new[idx_X,:]
                                  
                # Check whether to add the current points to the chains or not?
                if self.MCMCVar.T == self.MCMCPar.thin:
                    # Store the current sample in Sequences
                    self.MCMCVar.iloc = self.MCMCVar.iloc + 1
                    self.Sequences[self.MCMCVar.iloc,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(self.MCMCVar.X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))
                   
                   # Check whether to store the simulation results of the function evaluations

                    if self.MCMCPar.savemodout==True:
                        self.fx=np.append(self.fx,self.MCMCVar.Xfx,axis=0)
                        # Update m_func
                        self.MCMCVar.m_func = self.MCMCVar.m_func + self.MCMCPar.seq
                    else:
                        self.MCMCVar.m_func=None
                    # And set the T to 0
                    self.MCMCVar.T = 0

                # Compute squared jumping distance for each CR value
                if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):
                   
                    # Calculate the standard deviation of each dimension of X
                    r = matlib.repmat(np.std(self.MCMCVar.X[:,:self.MCMCPar.n],axis=0),self.MCMCPar.seq,1)
                    # Compute the Euclidean distance between new X and old X
                    delta_normX = np.sum(np.power((xold[:,:self.MCMCPar.n] - self.MCMCVar.X[:,:self.MCMCPar.n])/r,2),axis=1)
                    
                    #print('check :',np.sum(xold[:,:self.MCMCPar.n] - self.MCMCVar.X[:,:self.MCMCPar.n]))
                    
                    # Use this information to update sum_p2 to update N_CR
                    self.MCMCVar.delta_tot = CalcDelta(self.MCMCPar.nCR,self.MCMCVar.delta_tot,delta_normX,self.MCMCVar.CR[:,gen_number])

                # Check whether to append X to Z
                if np.mod((gen_number+1),self.MCMCPar.k) == 0:
                   
                    ## Append X to Z
                    self.Z[self.MCMCVar.m + 0 : self.MCMCVar.m + self.MCMCPar.seq,:self.MCMCPar.n+2] = np.array(self.MCMCVar.X[:,:self.MCMCPar.n+2])
                    # Update MCMCPar.m
                    self.MCMCVar.m = self.MCMCVar.m + self.MCMCPar.seq

                # How many candidate points have been accepted -- for Acceptance Rate
                totaccept = totaccept + np.sum(accept)

                # Update Iteration
                self.MCMCVar.Iter = self.MCMCVar.Iter + self.MCMCPar.seq
                
            print('Iter =',self.MCMCVar.Iter)  
            
            # Reduce MCMCPar.steps to get rounded iteration numbers
            if self.MCMCVar.iteration == 2: 
                self.MCMCPar.steps = self.MCMCPar.steps + 1

            # Store Important Diagnostic information -- Acceptance Rate
            self.OutDiag.AR[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)]).reshape((1,1))),axis=1)
            
            # Store Important Diagnostic information -- Probability of individual crossover values
            self.OutDiag.CR[self.MCMCVar.iteration-1,:self.MCMCPar.nCR+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), self.MCMCVar.pCR),axis=1)
            
            # Check whether to update individual pCR values
            if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):

                # Update pCR values
                self.MCMCVar.pCR = AdaptpCR(self.MCMCPar.seq,self.MCMCVar.delta_tot,self.MCMCVar.lCR,self.MCMCVar.pCR)

            # Check whether to update jump rate
            if (self.MCMCPar.AdaptJR==True and self.MCMCVar.Iter < 0.9 * self.MCMCPar.ndraw): 
                AR=100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)
                if AR < self.MCMCPar.TargetAR:
                    self.MCMCPar.jr_scale=np.maximum(self.MCMCPar.jr_scale*0.9,self.MCMCPar.jr_scale_min)

            # Generate CR values based on current pCR values
            self.MCMCVar.CR,lCRnew = GenCR(MCMCPar,self.MCMCVar.pCR); self.MCMCVar.lCR = self.MCMCVar.lCR + lCRnew

            # Calculate Gelman and Rubin Convergence Diagnostic
            start_idx = np.maximum(1,np.floor(0.5*self.MCMCVar.iloc)).astype('int64')-1; end_idx = self.MCMCVar.iloc
  
            # Compute the R-statistic using 50% burn-in from Sequences
            current_R_stat = GelmanRubin(self.Sequences[start_idx:end_idx,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)
            
            self.OutDiag.R_stat[self.MCMCVar.iteration-1,:self.MCMCPar.n+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)),np.array([current_R_stat]).reshape((1,self.MCMCPar.n))),axis=1)

            # Update the iteration
            self.MCMCVar.iteration = self.MCMCVar.iteration + 1

            if self.MCMCPar.save_tmp_out==True:
                with open('out_tmp'+'.pkl','wb') as f:
                    pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                    'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                    'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                    'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

        # Variables have been pre-allocated --> need to remove zeros at end
        self.Sequences,self.Z,self.OutDiag,self.fx = Dreamzs_finalize(self.MCMCPar,self.Sequences,self.Z,self.OutDiag,self.fx,self.MCMCVar.iteration,self.MCMCVar.iloc,self.MCMCVar.pCR,self.MCMCVar.m,self.MCMCVar.m_func)
        
        if self.MCMCPar.saveout==True:
            with open('dreamzs_out'+'.pkl','wb') as f:
                pickle.dump({'Sequences':self.Sequences,'Z':self.Z,'OutDiag':self.OutDiag,
                'fx':self.fx,'Extra':self.Extra,},f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.Sequences, self.Z, self.OutDiag,  self.fx, self.MCMCPar, self.MCMCVar         