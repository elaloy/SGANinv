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

from scipy.stats import triang

from mcmc_func import* # This imports both all Dream_zs and inverse problem-related functions

import sys

from attrdict import AttrDict

MCMCPar=AttrDict()

MCMCVar=AttrDict()

Measurement=AttrDict()

OutDiag=AttrDict()

Extra=AttrDict()

class Sampler:
    
    def __init__(self, main_dir=None,CaseStudy=0,seq = 3,ndraw=10000,thin = 1,  nCR = 3, 
                 DEpairs = 3, parallelUpdate = 0.9, pCR=True,k=10,pJumpRate_one=0.2,
                 steps=100,savemodout=False, saveout=True,save_tmp_out=True,Prior='LHS',
                 DoParallel=True,eps=5e-2,BoundHandling='Reflect',
                 lik_sigma_est=False,parallel_jobs=4,jr_scale=1.0,rng_seed=123):
        
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
        Extra.main_dir=main_dir
        np.random.seed(rng_seed)
        MCMCPar.rng_seed=rng_seed

        if self.CaseStudy==2:   
   
            ModelName='linear_gpr_tomo'
            MCMCPar.lik=2
           
            MCMCPar.savemodout=False
            MCMCPar.lik_sigma_est=False
            
            MCMCPar.lb=np.ones((1,15))*-1
            MCMCPar.ub=np.ones((1,15))
            MCMCPar.n=MCMCPar.ub.shape[1]

            # Set forward model stuff:
            from tomokernel_straight import tomokernel_straight_2D
            
            nx=60 # Here x is the horizontal axis (number of columns) and not the number of rows
            ny = 125 # Here y is the vertical axis (number of rows) and not the number of columns
        
            # The x-axis is varying the fastest 
            x = np.arange(0,(nx/10)+0.1,0.1)                      
            y = np.arange(0,(ny/10)+0.1,0.1) 
            sourcex = 0.01
            sourcez = np.arange(0.5,ny/10,0.5)                         
            receiverx = nx/10-0.01
            receiverz = np.arange(0.5,ny/10,0.5)   
            nsource = len(sourcez); nreceiver = len(receiverz)
            ndata=nsource*nreceiver
            data=np.zeros((ndata,4))
            # Calculate acquisition geometry (multiple-offset gather)
            for jj in range(0,nsource):
                for ii in range(0,nreceiver):
                    data[ ( jj ) * nreceiver + ii , :] = np.array([sourcex, sourcez[jj], receiverx, receiverz[ii]])
            # Calculate forward modeling kernel (from Matlab code by Dr. James Irving, UNIL)
            G = tomokernel_straight_2D(data,x,y) # Distance of ray-segment in each cell for each ray
            Extra.G=np.array(G.todense())
            
            # DNN stuff
            DNN=AttrDict()
            DNN.nx=nx
            DNN.ny=ny
            DNN.zx=5
            DNN.zy=3
            DNN.nc=1
            DNN.nz=1
            DNN.depth=5
            DNN.threshold=True
            DNN.filtering=False
            DNN.cuda=True
            
            DNN.gpath=Extra.main_dir+'/generation/netG.pth'

            from generator import Generator as Generator
            
            DNN.npx=(DNN.zx-1)*2**DNN.depth + 1
            DNN.npy=(DNN.zy-1)*2**DNN.depth + 1
            DNN.netG = Generator(cuda=DNN.cuda, gpath=DNN.gpath)
            for param in DNN.netG.parameters():
                param.requires_grad = False
            DNN.netG.eval()
            if DNN.cuda:
                DNN.netG.cuda()
            self.DNN=DNN
            
            # Load measurements
            file_name='measurements.pkl'
            Measurement.Sigma=1 # Measurement error is 1 ns
            with open(file_name, 'rb') as fin:
                    tmp=pickle.load(fin)
                    Extra.z_true=tmp['z_true']
                    if DNN.threshold:
                        Extra.m_true=tmp['m_true']
                        Measurement.MeasData=tmp['d']
                    else:
                        Extra.m_true=tmp['m_true_cont']
                        Measurement.MeasData=tmp['d_cont']
                    del tmp
            Measurement.N=len(Measurement.MeasData)
        
        elif self.CaseStudy==1: 
            # A theoretical 10-dimensional bimodal distribution made of 2 Gaussians
            # (example 3 in Matlab DREAMzs code)
            self.ndim=10
            MCMCPar.n=self.ndim
            MCMCPar.Prior='COV'
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
            # A theoretical multivariate normal distribution with 100 correlated dimensions 
            # (example 2 in Matlab DREAM code)
            self.ndim=100
            MCMCPar.n=self.ndim
            MCMCPar.Prior='LHS'
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-5
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+15
            MCMCPar.BoundHandling='Reflect'
			
            Measurement.N=1
            ModelName='theoretical_case_mvn'
            MCMCPar.lik=0
            
            A = 0.5 * np.eye(MCMCPar.n) + 0.5 * np.ones(MCMCPar.n)
            cov=np.zeros((MCMCPar.n,MCMCPar.n))
            # Rescale to variance-covariance matrix of interest
            for i in range (0,MCMCPar.n):
                for j in range (0,MCMCPar.n):
                    cov[i,j] = A[i,j] * np.sqrt((i+1) * (j+1))
            Extra.C=cov
            Extra.invC = np.linalg.inv(cov)
            
        else: # This should not happen and is thus probably not needed
            self.ndim=1
            MCMCPar.n=self.ndim
            MCMCPar.lb=np.zeros((1,MCMCPar.n))
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+1
            MCMCPar.BoundHandling=None
            Measurement.N=1
            ModelName=None
            MCMCPar.lik=1

        MCMCPar.m0=10*MCMCPar.n
        
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
            if self.MCMCPar.lik_sigma_est==True: # Use log-uniform prior for sigma
                Zinit[:,0]=lhs(self.MCMCPar.lb[0][0].reshape((1,1)),self.MCMCPar.ub[0][0].reshape((1,1)),self.MCMCPar.m0+self.MCMCPar.seq).reshape((self.MCMCPar.m0+self.MCMCPar.seq))
                
        elif self.MCMCPar.Prior=='COV': # Generate initial population from multivariate normal distribution but the model returns posterior density directly
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
        for zz in range(0,self.MCMCPar.DEpairs):
            self.MCMCPar.Table_JumpRate[:,zz] = 2.38/np.sqrt(2 * (zz+1) * np.linspace(1,self.MCMCPar.n,self.MCMCPar.n).T)
        
        # Change steps to make sure to get nice iteration numbers in first loop
        self.MCMCPar.steps = self.MCMCPar.steps - 1
        
        self.Z = np.zeros((np.floor(self.MCMCPar.m0 + self.MCMCPar.seq * (self.MCMCPar.ndraw - self.MCMCPar.m0) / (self.MCMCPar.seq * self.MCMCPar.k)).astype('int64')+self.MCMCPar.seq*100,self.MCMCPar.n+2))
        self.Z[:self.MCMCPar.m0,:self.MCMCPar.n] = Zinit[:self.MCMCPar.m0,:self.MCMCPar.n]

        X = Zinit[self.MCMCPar.m0:(self.MCMCPar.m0+self.MCMCPar.seq),:self.MCMCPar.n]
        
        ###X[0,:]=Extra.z_true.cpu().numpy().flatten()
        
        del Zinit
        
        # Run forward model, if any this is done in parallel
        if  self.CaseStudy > 1:
            if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                fx0 = RunFoward(X[:,:-1],self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
            else:
                fx0 = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)    
        else:
            fx0 = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
        
        # Compute likelihood from simulated data    
        of,log_p = CompLikelihood(X,fx0,self.MCMCPar,self.Measurement,self.Extra)

        X = np.concatenate((X,of,log_p),axis=1)
        Xfx = fx0
        
        if self.MCMCPar.savemodout==True:
            self.fx=fx0
        else:
            self.fx=None

        self.Sequences[0,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))

        # Store N_CR
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
            
            self.ndim=self.MCMCPar.n
#                
            self.MCMCPar.ndraw = 2 * self.MCMCPar.ndraw
            
            # Reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
            # Extend Sequences, Z, OutDiag.AR,OutDiag.Rstat and OutDiag.CR
            self.Sequences=np.concatenate((self.Sequences,np.zeros((self.Sequences.shape))),axis=0)
            self.Z=np.concatenate((self.Z,np.zeros((self.Z.shape))),axis=0)
            self.OutDiag.AR=np.concatenate((self.OutDiag.AR,np.zeros((self.OutDiag.AR.shape))),axis=0)
            self.OutDiag.R_stat=np.concatenate((self.OutDiag.R_stat,np.zeros((self.OutDiag.R_stat.shape))),axis=0)
            self.OutDiag.CR=np.concatenate((self.OutDiag.CR,np.zeros((self.OutDiag.CR.shape))),axis=0)
      
            
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

#            start_time = time.time()
            
            # Loop a number of times before calculating convergence diagnostic, etc.
            for gen_number in range(0,self.MCMCPar.steps):
                
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
                    if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                        fx_new = RunFoward(xnew[:,:-1],self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)
                    else:
                        fx_new = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra,DNN=self.DNN)    
                else:
                    fx_new = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                 
                # Compute the likelihood of each proposal in each chain
                of_xnew,log_p_xnew = CompLikelihood(xnew,fx_new,self.MCMCPar,self.Measurement,self.Extra)
    
                # Calculate the Metropolis ratio
                accept = Metrop(self.MCMCPar,xnew,log_p_xnew,xold,log_p_xold,alfa_s,Extra)

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
                                        
                    # Use this information to update delta_tot which will be used to update the pCR values
                    self.MCMCVar.delta_tot = CalcDelta(self.MCMCPar.nCR,self.MCMCVar.delta_tot,delta_normX,self.MCMCVar.CR[:,gen_number])

                # Check whether to append X to Z
                if np.mod((gen_number+1),self.MCMCPar.k) == 0:
                   
                    ## Append X to Z
                    self.Z[self.MCMCVar.m + 0 : self.MCMCVar.m + self.MCMCPar.seq,:self.MCMCPar.n+2] = np.array(self.MCMCVar.X[:,:self.MCMCPar.n+2])
                    # Update MCMCPar.m
                    self.MCMCVar.m = self.MCMCVar.m + self.MCMCPar.seq

                # Compute number of accepted moves
                totaccept = totaccept + np.sum(accept)

                # Update total number of MCMC iterations
                self.MCMCVar.Iter = self.MCMCVar.Iter + self.MCMCPar.seq
                
            print('Iter =',self.MCMCVar.Iter)  
            
            # Reduce MCMCPar.steps to get rounded iteration numbers
            if self.MCMCVar.iteration == 2: 
                self.MCMCPar.steps = self.MCMCPar.steps + 1

            # Store acceptance rate
            self.OutDiag.AR[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)]).reshape((1,1))),axis=1)
            
            # Store probability of individual crossover values
            self.OutDiag.CR[self.MCMCVar.iteration-1,:self.MCMCPar.nCR+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), self.MCMCVar.pCR),axis=1)
            
            # Is pCR updating required?
            if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):

                # Update pCR values
                self.MCMCVar.pCR = AdaptpCR(self.MCMCPar.seq,self.MCMCVar.delta_tot,self.MCMCVar.lCR,self.MCMCVar.pCR)

            # Generate CR values from current pCR values
            self.MCMCVar.CR,lCRnew = GenCR(MCMCPar,self.MCMCVar.pCR); self.MCMCVar.lCR = self.MCMCVar.lCR + lCRnew

            # Calculate Gelman and Rubin Convergence Diagnostic
            start_idx = np.maximum(1,np.floor(0.5*self.MCMCVar.iloc)).astype('int64')-1; end_idx = self.MCMCVar.iloc
            
            current_R_stat = GelmanRubin(self.Sequences[start_idx:end_idx,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)
            
            self.OutDiag.R_stat[self.MCMCVar.iteration-1,:self.MCMCPar.n+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)),np.array([current_R_stat]).reshape((1,self.MCMCPar.n))),axis=1)

            # Update number of complete generation loops
            self.MCMCVar.iteration = self.MCMCVar.iteration + 1

            if self.MCMCPar.save_tmp_out==True:
                with open('out_tmp'+'.pkl','wb') as f:
                    pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                    'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                    'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                    'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove zeros from pre-allocated variavbles if needed
        self.Sequences,self.Z,self.OutDiag,self.fx = Dreamzs_finalize(self.MCMCPar,self.Sequences,self.Z,self.OutDiag,self.fx,self.MCMCVar.iteration,self.MCMCVar.iloc,self.MCMCVar.pCR,self.MCMCVar.m,self.MCMCVar.m_func)
        
        if self.MCMCPar.saveout==True:
            with open('dreamzs_out'+'.pkl','wb') as f:
                pickle.dump({'Sequences':self.Sequences,'Z':self.Z,'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,'Measurement':self.Measurement,'Extra':self.Extra},f
                , protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.Sequences, self.Z, self.OutDiag,  self.fx, self.MCMCPar, self.MCMCVar         
