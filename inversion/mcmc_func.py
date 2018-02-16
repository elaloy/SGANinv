# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>
"""
import numpy as np
from scipy.stats import multivariate_normal
import time
from joblib import Parallel, delayed
import sys
import os

def lhs(minn,maxn,N): # Latin Hypercube sampling
    # Here minn and maxn are assumed to be 1xd arrays 
    x = np.zeros((N,minn.shape[1]))

    for j in xrange (0,minn.shape[1]):
    
        idx = np.random.permutation(N)+0.5
        P =(idx - x[:,j])/N
        x[:,j] = minn[0,j] + P*(maxn[0,j] - minn[0,j])

    return x
    
def GenCR(MCMCPar,pCR):

    if type(pCR) is np.ndarray:
        p=np.ndarray.tolist(pCR)[0]
    else:
        p=pCR
    CR=np.zeros((MCMCPar.seq * MCMCPar.steps),dtype=np.float)
    L =  np.random.multinomial(MCMCPar.seq * MCMCPar.steps, p, size=1)
    L2 = np.concatenate((np.zeros((1),dtype=np.int), np.cumsum(L)),axis=0)

    r = np.random.permutation(MCMCPar.seq * MCMCPar.steps)

    for zz in xrange(0,MCMCPar.nCR):
        
        i_start = L2[zz]
        i_end = L2[zz+1]
        idx = r[i_start:i_end]
        CR[idx] = np.float(zz+1)/MCMCPar.nCR
        
    CR = np.reshape(CR,(MCMCPar.seq,MCMCPar.steps))
    return CR, L

def CalcDelta(nCR,delta_tot,delta_normX,CR):
    # Calculate total normalized Euclidean distance for each crossover value
    
    # Derive sum_p2 for each different CR value 
    for zz in xrange(0,nCR):
    
        # Find which chains are updated with zz/MCMCPar.nCR
        idx = np.argwhere(CR==(1.0+zz)/nCR);idx=idx[:,0]
    
        # Add the normalized squared distance tot the current delta_tot;
        delta_tot[0,zz] = delta_tot[0,zz] + np.sum(delta_normX[idx])
    
    return delta_tot

def AdaptpCR(seq,delta_tot,lCR,pCR_old):
    
    if np.sum(delta_tot) > 0:
        # Updates the probabilities of the various crossover values
        # Adapt pCR using information from averaged normalized jumping distance
        pCR = seq * (delta_tot/lCR) / np.sum(delta_tot)

        # Normalize pCR
        pCR = pCR/np.sum(pCR)
    else:
        pCR=pCR_old
    
    return pCR

def CompDensity(X,fx,MCMCPar,Measurement,Extra):
    
    if MCMCPar.lik==0: # fx contains log-density
        of = np.exp(fx)       
        log_p= fx

    elif MCMCPar.lik==1: # fx contains density
        of = fx       
        log_p= np.log(of)
        
    else: # fx contains  simulated data
#        of=np.zeros((fx.shape[0],1))+999+np.random.randn(fx.shape[0],1)
#        log_p=np.log(of)
        of=np.zeros((fx.shape[0],1))
        log_p=np.zeros((fx.shape[0],1))
        for ii in xrange(0,MCMCPar.seq):
            e=Measurement.MeasData-fx[ii,:]
            of[ii,0]=np.sqrt(np.sum(np.power(e,2.0))/e.shape[1])
            if MCMCPar.lik==2:
                log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(2.0 * np.pi) - Measurement.N * np.log( Measurement.Sigma ) - 0.5 * np.power(Measurement.Sigma,-2.0) * np.sum( np.power(e,2.0) )
            if MCMCPar.lik==3:
                log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(np.sum(np.power(e,2.0)))
            if MCMCPar.lik==4:
                log_p[ii,0]= - ( Measurement.N / 2.0) * np.log(2.0 * np.pi) - Measurement.N * np.log( Measurement.Sigma ) - 0.5 * np.power(Measurement.Sigma,-2.0) * np.sum( np.power(e,2.0) )
                if Extra.DomainGeom=='2D':
                    e2=Extra.dcp[ii,:]-Extra.condset[:,2]
                elif Extra.DomainGeom=='3D':
                    e2=Extra.dcp[ii,:]-Extra.condset[:,3]
                N2=len(e2)
                
                Sigma2=0.5
                log_p2=- ( N2 / 2.0) * np.log(2.0 * np.pi) - N2 * np.log( Sigma2 ) - 0.5 * np.power(Sigma2,-2.0) * np.sum( np.power(e2,2.0) )
                log_p[ii,0]=log_p[ii,0]+log_p2
    return of, log_p

def GelmanRubin(Sequences,MCMCPar):
    """
    See:
    Gelman, A. and D.R. Rubin, 1992. 
    Inference from Iterative Simulation Using Multiple Sequences, 
    Statistical Science, Volume 7, Issue 4, 457-472.
    """
    
    n,nrp,m = Sequences.shape

    if n < 10:
        R_stat = -2 * np.ones((1,MCMCPar.n))
        
    else:
    
        meanSeq = np.mean(Sequences,axis=0)
        meanSeq = meanSeq.T
    
        # Variance between the sequence means 
        B = n * np.var(meanSeq,axis=0)
        
        # Variances of the various sequences
        varSeq=np.zeros((m,nrp))
        for zz in xrange(0,m):
            varSeq[zz,:] = np.var(Sequences[:,:,zz],axis=0)
        
        # Average of the within sequence variances
        W = np.mean(varSeq,axis=0)
        
        # Target variance
        sigma2 = ((n - 1)/np.float(n)) * W + (1.0/n) * B
        
        # R-statistic
        R_stat = np.sqrt((m + 1)/np.float(m) * sigma2 / W - (n-1)/np.float(m)/np.float(n))
    
    return R_stat
    
def DEStrategy(DEpairs,seq):
    
    # Determine which sequences to evolve with what DE strategy

    # Determine probability of selecting a given number of pairs
    p_pair = (1.0/DEpairs) * np.ones((1,DEpairs))
    p_pair = np.cumsum(p_pair)
    p_pair = np.concatenate((np.zeros((1)),p_pair),axis=0)
    
    DEversion=np.zeros((seq),dtype=np.int32)
    Z = np.random.rand(seq)
    # Select number of pairs
    for qq in xrange(0,seq):
        z = np.where(p_pair<=Z[qq])
        DEversion[qq] = z[0][-1]
            
    return DEversion
        
def BoundaryHandling(x,lb,ub,BoundHandling): 
 
    m,n=np.shape(x)
    
    # Replicate lb and ub
    minn = np.tile(lb,(m,1))
    maxn = np.tile(ub,(m,1))
    
    ii_low = np.argwhere(x<minn)
    ii_up = np.argwhere(x>maxn)
    
    if BoundHandling=='Reflect':
         # reflect in minn
         x[ii_low[:,0],ii_low[:,1]]=2 * minn[ii_low[:,0],ii_low[:,1]] - x[ii_low[:,0],ii_low[:,1]]      

         # reflect in maxn
         x[ii_up[:,0],ii_up[:,1]]=2 * maxn[ii_up[:,0],ii_up[:,1]] - x[ii_up[:,0],ii_up[:,1]] 
         
    if BoundHandling=='Bound':
         # set lower values to minn
        x[ii_low[:,0],ii_low[:,1]]= minn[ii_low[:,0],ii_low[:,1]] 
    
        # set upper values to maxn
        x[ii_up[:,0],ii_up[:,1]]= maxn[ii_up[:,0],ii_up[:,1]] 
        
    if BoundHandling=='Fold':
         # Fold parameter space lower values
        x[ii_low[:,0],ii_low[:,1]] = maxn[ii_low[:,0],ii_low[:,1]] - ( minn[ii_low[:,0],ii_low[:,1]] - x[ii_low[:,0],ii_low[:,1]]  )
    
        # Fold parameter space upper values
        x[ii_up[:,0],ii_up[:,1]] = minn[ii_up[:,0],ii_up[:,1]] + ( x[ii_up[:,0],ii_up[:,1]]  - maxn[ii_up[:,0],ii_up[:,1]] )
             
    # Now double check in case elements are still out of bound -- this is
    # theoretically possible if values are very small or large              
    ii_low = np.argwhere(x<minn)
    ii_up = np.argwhere(x>maxn)
    
    if ii_low.size > 0:
       
        x[ii_low[:,0],ii_low[:,1]] = minn[ii_low[:,0],ii_low[:,1]]  + np.random.rand(ii_low.shape[0]) * (maxn[ii_low[:,0],ii_low[:,1]] - minn[ii_low[:,0],ii_low[:,1]])
   
    if ii_up.size > 0:
      
        x[ii_up[:,0],ii_up[:,1]] = minn[ii_up[:,0],ii_up[:,1]] +  + np.random.rand(ii_up.shape[0]) * (maxn[ii_up[:,0],ii_up[:,1]] - minn[ii_up[:,0],ii_up[:,1]])


    return x
    
def DreamzsProp(xold,Zoff,CR,MCMCPar,Update):
    
    # Determine how many pairs to use for each jump in each chain
    DEversion = DEStrategy(MCMCPar.DEpairs,MCMCPar.seq)

    # Generate uniform random numbers for each chain to determine which dimension to update
    D = np.random.rand(MCMCPar.seq,MCMCPar.n)

    # Generate noise to ensure ergodicity for each individual chain
    noise_x = MCMCPar.eps * (2 * np.random.rand(MCMCPar.seq,MCMCPar.n) - 1)

    # Initialize the delta update to zero
    delta_x = np.zeros((MCMCPar.seq,MCMCPar.n))

    if Update=='Parallel_Direction_Update':

        # Define which points of Zoff to use to generate jumps
        rr=np.zeros((MCMCPar.seq,4),dtype=np.int32())
        rr[0,0] = 0; rr[0,1] = rr[0,0] + DEversion[0]
        rr[0,2] = rr[0,1] +1 ; rr[0,3] = rr[0,2] + DEversion[0]
        # Do this for each chain
        for qq in range(1,MCMCPar.seq):
            # Define rr to be used for population evolution
            rr[qq,0] = rr[qq-1,3] + 1; rr[qq,1] = rr[qq,0] + DEversion[qq] 
            rr[qq,2] = rr[qq,1] + 1; rr[qq,3] = rr[qq,2] + DEversion[qq] 
 

        # Each chain evolves using information from other chains to create offspring
        for qq in xrange(0,MCMCPar.seq):

            # ------------ WHICH DIMENSIONS TO UPDATE? USE CROSSOVER ----------
            i = np.where(D[qq,:] > (1-CR[qq]))
            #i=i[0].tolist()

            # Update at least one dimension
            if not i:
                i=np.random.permutation(MCMCPar.n)
                i=np.zeros((1,1),dtype=np.int32)+i[0]
       
              
        # -----------------------------------------------------------------

            # Select the appropriate JumpRate and create a jump
            if (np.random.rand(1) < (1 - MCMCPar.pJumpRate_one)):
                
                # Select the JumpRate (dependent of NrDim and number of pairs)
                NrDim = len(i[0])
                JumpRate = MCMCPar.Table_JumpRate[NrDim-1,DEversion[qq]]*MCMCPar.jr_scale
               
                # Produce the difference of the pairs used for population evolution
                if MCMCPar.DEpairs==1:
                    delta = Zoff[rr[qq,0],:]- Zoff[rr[qq,2],:]
                else:
                    # The number of pairs has been randomly chosen between 1 and DEpairs
                    delta = np.sum(Zoff[rr[qq,0]:rr[qq,1]+1,:]- Zoff[rr[qq,2]:rr[qq,3]+1,:],axis=0)
                
                # Then fill update the dimension
                delta_x[qq,i] = (1 + noise_x[qq,i]) * JumpRate*delta[i]
            else:
                # Set the JumpRate to 1 and overwrite CR and DEversion
                JumpRate = 1; CR[qq] = -1
                # Compute delta from one pair
                delta = Zoff[rr[qq,0],:] - Zoff[rr[qq,3],:]
                # Now jumprate to facilitate jumping from one mode to the other in all dimensions
                delta_x[qq,:] = JumpRate * delta
       

    if Update=='Snooker_Update':
                 
        # Determine the number of rows of Zoff
        NZoff = np.int64(Zoff.shape[0])
        
        # Define rr and z
        rr = np.arange(NZoff)
        rr = rr.reshape((2,rr.shape[0]/2),order="F").T
        z=np.zeros((MCMCPar.seq,MCMCPar.n))
        # Define JumpRate -- uniform rand number between 1.2 and 2.2
        Gamma = 1.2 + np.random.rand(1)
        # Loop over the individual chains
        
        for qq in range(0,MCMCPar.seq):
            # Define which points of Zoff z_r1, z_r2
            zR1 = Zoff[rr[qq,0],:]; zR2 = Zoff[rr[qq,1],:]
            # Now select z from Zoff; z cannot be zR1 and zR2
            ss = np.arange(NZoff)+1; ss[rr[qq,0]] = 0; ss[rr[qq,1]] = 0; ss = ss[ss>0]; ss=ss-1
            t = np.random.permutation(NZoff-2)
            # Assign z
            z[qq,:] = Zoff[ss[t[0]],:]
                            
            # Define projection vector x(qq) - z
            F = xold[qq,0:MCMCPar.n] - z[qq,:]; Ds = np.maximum(np.dot(F,F.T),1e-300)
            # Orthogonally project of zR1 and zR2 onto F
            zP = F*np.sum((zR1-zR2)*F)/Ds
            
            # And define the jump
            delta_x[qq,:] = Gamma * zP
            # Update CR because we only consider full dimensional updates
            CR[qq] = 1
          
    # Now propose new x
    xnew = xold + delta_x
    
    # Define alfa_s
    if Update == 'Snooker_Update':
       
        # Determine Euclidean distance
        ratio=np.sum(np.power((xnew-z),2),axis=1)/np.sum(np.power((xold-z),2),axis=1)
        alfa_s = np.power(ratio,(MCMCPar.n-1)/2.0).reshape((MCMCPar.seq,1))
            
    else:
        alfa_s = np.ones((MCMCPar.seq,1))

    # Do boundary handling -- what to do when points fall outside bound
    if not(MCMCPar.BoundHandling==None):
        
        xnew = BoundaryHandling(xnew,MCMCPar.lb,MCMCPar.ub,MCMCPar.BoundHandling)

    
    return xnew, CR ,alfa_s


def Metrop(MCMCPar,xnew,log_p_xnew,xold,log_p_xold,alfa_s):
    
    accept = np.zeros((MCMCPar.seq,1))
   

    # Calculate the Metropolis ratio based on the log-likelihoods
    alfa = np.exp(log_p_xnew - log_p_xold.reshape((log_p_xold.shape[0],1)))
    
    if MCMCPar.Prior=='StandardNormal':
        log_prior_new=np.zeros((MCMCPar.seq,1))
        log_prior_old=np.zeros((MCMCPar.seq,1))
            
        for zz in xrange(0,MCMCPar.seq):
            # Compute (standard normal) prior log density of proposal
            log_prior_new[zz,0] = -0.5 * reduce(np.dot,[xnew[zz,:],xnew[zz,:].T])     
             
            # Compute (standard normal) prior log density of current location
            log_prior_old[zz,0] = -0.5 * reduce(np.dot,[xold[zz,:],xold[zz,:].T])

        # Take the ratio
        alfa_pr = np.exp(log_prior_new - log_prior_old)
        # Now update alpha value with prior
        alfa = alfa*alfa_pr 
        
    if MCMCPar.Prior=='Normal':
      
        log_prior_new=np.zeros((MCMCPar.seq,1))
        log_prior_old=np.zeros((MCMCPar.seq,1))
            
        for zz in xrange(0,MCMCPar.seq):
            # Compute prior log density of proposal
            log_prior_new[zz,0] = -0.5 * reduce(np.dot,[(xnew[zz,:]-MCMCPar.pmu),(1/np.power(MCMCPar.psd,2)),(xnew[zz,:]-MCMCPar.pmu).T])     
            
            # Compute prior log density of current location
            log_prior_old[zz,0] = -0.5 * reduce(np.dot,[(xold[zz,:]-MCMCPar.pmu),(1/np.power(MCMCPar.psd,2)),(xold[zz,:]-MCMCPar.pmu).T])     
        # Take the ratio
        
        alfa_pr = np.exp(log_prior_new - log_prior_old)
        # Now update alpha value with prior
        alfa = alfa*alfa_pr 
   
    # Modify for snooker update, if any
    alfa = alfa * alfa_s
  
    # Generate random numbers
    Z = np.random.rand(MCMCPar.seq,1)
    
    # Find which alfa's are greater than Z
    idx = np.argwhere(alfa > Z)
    idx=idx[:,0]
    
    # And indicate that these chains have been accepted
    accept[idx,0]=1
    
    return accept

def Dreamzs_finalize(MCMCPar,Sequences,Z,outDiag,fx,iteration,iloc,pCR,m_z,m_func):
    
    # Variables have been pre-allocated --> need to remove zeros at end

    # Start with CR
    outDiag.CR = outDiag.CR[0:iteration-1,0:pCR.shape[1]+1]
    # Then R_stat
    outDiag.R_stat = outDiag.R_stat[0:iteration-1,0:MCMCPar.n+1]
    # Then AR 
    outDiag.AR = outDiag.AR[0:iteration-1,0:2] 
    # Adjust last value (due to possible sudden end of for loop)

    # Then Sequences
    Sequences = Sequences[0:iloc+1,0:MCMCPar.n+2,0:MCMCPar.seq]
    
    # Then the archive Z
    Z = Z[0:m_z,0:MCMCPar.n+2]


    if MCMCPar.savemodout==True:
       # remove zeros
       fx = fx[:,0:m_func]
    
    return Sequences,Z, outDiag, fx
    
def Genparset(Sequences):
    # Generates a 2D matrix ParSet from 3D array Sequences

    # Determine how many elements in Sequences
    NrX,NrY,NrZ = Sequences.shape 

    # Initalize ParSet
    ParSet = np.zeros((NrX*NrZ,NrY))

    # If save in memory -> No -- ParSet is empty
    if not(NrX == 0):
        # ParSet derived from all sequences
        tt=0
        for qq in xrange(0,NrX):
            for kk in xrange(0,NrZ):
                ParSet[tt,:]=Sequences[qq,:,kk]
                tt=tt+1
    return ParSet
        #ParSet = sortrows(ParSet,[MCMCPar.n+3]); ParSet = ParSet(:,1:MCMCPar.n+2);

def forward_parallel(forward_process,X,n,n_jobs,extra_par): 
    
    n_row=X.shape[0]
    
    parallelizer = Parallel(n_jobs=n_jobs)
    
    tasks_iterator = ( delayed(forward_process)(X_row,n,extra_par) 
                      for X_row in np.split(X,n_row))
         
    result = parallelizer( tasks_iterator )
    # merging the output of the jobs
    return np.vstack(result)
      
def forward_model_0(X,n,par):
    
    fx=np.zeros((1,n))
    fx[0,:n] = multivariate_normal.pdf(X, mean=np.zeros((X.shape[1])), cov=np.eye(X.shape[1]))
    
    time.sleep(5)
    
    return fx
    
    
def theoretical_case_mvn(X, n, icov):
    

    fx=np.zeros((1,n))
    # Calculate the log density of zero mean correlated mvn distribution
    fx[0,:n] = -0.5*X.dot(icov).dot(X.T)
    
    #time.sleep(5)
    
    return fx
    
def theoretical_case_bimodal_mvn(X, n, par):
    
    fx=np.zeros((1,n))
    fx[0,:n] = (1.0/3)*multivariate_normal.pdf(X, mean=par[0][0], cov=par[1][0])+(2.0/3)*multivariate_normal.pdf(X, mean=par[2][0], cov=par[3][0])
  
    #time.sleep(5)
    
    return fx
    
def RunFoward(X,MCMCPar,Measurement,ModelName,Extra,DNN=None):
    
    n=Measurement.N
    n_jobs=Extra.n_jobs
    
    if ModelName=='theoretical_case_mvn':
        extra_par = Extra.invC
        
    elif ModelName=='theoretical_case_bimodal_mvn':
        extra_par = []
        extra_par.append([Extra.mu1])
        extra_par.append([Extra.cov1])
        extra_par.append([Extra.mu2])
        extra_par.append([Extra.cov2])
        
    elif ModelName=='forward_model_flow':
        extra_par=[]
        extra_par.append([Extra.SimType])
        extra_par.append([Extra.idx])
        if Extra.DomainGeom=='2D':
            noise_input = X.astype('float32').reshape((X.shape[0],DNN.nz,DNN.npx,DNN.npx))
            del X
            #X=DNN.gen_from_noise(noise_input)
            depth=5
            np_bigx=(DNN.npx - 1)*2**depth + 1
            X=np.empty((noise_input.shape[0],np_bigx,np_bigx),dtype=float)
            for ii in xrange(0,X.shape[0]):
                X[ii,:]=DNN.gen_from_noise(noise_input[ii,:].reshape((1,
                noise_input.shape[1],noise_input.shape[2], noise_input.shape[3])))
        elif Extra.DomainGeom=='3D':
            noise_input = X.astype('float32').reshape((X.shape[0],DNN.nz,DNN.npx,DNN.npx,DNN.npx))
            del X
            #X=DNN.gen_from_noise(noise_input) # this behaves strangely in 3D
            # Do this instead:
            depth=5
            np_bigx=(DNN.npx - 1)*2**depth + 1
            start_time = time.time()
            X=np.empty((noise_input.shape[0],np_bigx,np_bigx,np_bigx),dtype=float)
            for ii in xrange(0,X.shape[0]):
                X[ii,:]=DNN.gen_from_noise(noise_input[ii,:].reshape((1,
                noise_input.shape[1],noise_input.shape[2], noise_input.shape[3]
                ,noise_input.shape[4])))
            
#            forward_process=getattr(sys.modules[__name__], 'gen_mod3D')
#            X=forward_parallel(forward_process,noise_input,n,n_jobs,DNN.gen_from_noise)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Model generation done in %5.4f seconds." % (elapsed_time))
        # Crop if needed
        if Extra.crop:
            if Extra.DomainGeom=='2D':
                X=X[:,Extra.crop_istart:Extra.crop_iend,Extra.crop_istart:Extra.crop_iend]
                if MCMCPar.lik==4:
                    dcp=np.zeros((X.shape[0],Extra.condset.shape[0]))
                    for ii in xrange(0,dcp.shape[0]):
                        for jj in xrange(0,dcp.shape[1]):
                            dcp[ii,jj]=X[ii,Extra.idx[jj,0],Extra.idx[jj,1]] 
                else:
                    dcp=None
            elif Extra.DomainGeom=='3D': 
			    # Turn 3D fields into right position for flow simulation and then crop
                Xturn=np.zeros((X.shape[0],X.shape[3],X.shape[2],X.shape[1]))
                for ii in xrange(0,X.shape[0]):
                    Xturn[ii,:,:,:]=np.rot90(X[ii,:,:,:],1,axes=(0,2))
                del X
                X=Xturn[:,Extra.crop_istart:Extra.crop_iend,Extra.crop_jstart:Extra.crop_jend,Extra.crop_kstart:Extra.crop_kend]
                del Xturn
                if MCMCPar.lik==4:
                    dcp=np.zeros((X.shape[0],Extra.condset.shape[0]))
                    for ii in xrange(0,dcp.shape[0]):
                        for jj in xrange(0,dcp.shape[1]):
                            dcp[ii,jj]=X[ii,Extra.jdx[jj,0],Extra.jdx[jj,1],Extra.jdx[jj,2]] 
                else:
                    dcp=None
                    
        X=X.reshape((X.shape[0],-1))
        idx=np.arange(MCMCPar.seq).reshape((MCMCPar.seq,1))+1    
        X=np.concatenate((idx,X),axis=1)
        X=X.astype('float64')
        
    
        
    else:
        extra_par=None
    
    forward_process=getattr(sys.modules[__name__], ModelName)
    
    if MCMCPar.DoParallel==True:
    
        start_time = time.time()
     
        fx=forward_parallel(forward_process,X,n,n_jobs,extra_par)

        end_time = time.time()
        elapsed_time = end_time - start_time
    
        if not(ModelName[0:4]=='theo'):
            print("Parallel forward calls done in %5.4f seconds." % (elapsed_time))
    else:
        fx=np.zeros((MCMCPar.seq,n))
        
        start_time = time.time()
        
        if not(ModelName[0:4]=='theo'): # X needs to be a 1-dim array instead of a vector
            for qq in xrange(0,X.shape[0]):
                fx[qq,:]=forward_process(X[qq,:].reshape((1,X.shape[1])),n,extra_par)
        else:
            for qq in xrange(0,X.shape[0]):
                fx[qq,:]=forward_process(X[qq,:].reshape((1,X.shape[1])),n,extra_par)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
         
        if not(ModelName[0:4]=='theo'):
            print("Sequential forward calls done in %5.4f seconds." % (elapsed_time))
    
    return fx,dcp
    
def gen_mod3D(z,n,gen_func):
    X=gen_func(z.reshape((1,z.shape[1],z.shape[2], z.shape[3],z.shape[4])))
    return X
    
def forward_model_flow(X,n,par):
    
    idf=X[0,0].astype('uint8')
    X=np.array(X[0,1:])
    main_dir=os.getcwd()
    model_dir=main_dir + "/forward_setup_"+str(idf)
    os.chdir(model_dir)
    if par[0][0]==1:
        npx=125 # todo: make npx a parameter
        X[X==0]=1e-4;X[X==1]=1e-2
    elif par[0][0]==2:
        X[X==0]=1e-8;X[X==1]=1e-4
    elif par[0][0]==3:
        npx=65
        X[X==0]=1e-6
        X[X==1]=1e-4
       
    if  par[0][0]==1 or  par[0][0]==2:
        filename = r'fmod.mlt'
        template='# MF2005 multiplication file \n1 \nmult \nINTERNAL 1.0 (free) 0'
        f = open(filename, 'w') 
        np.savetxt(filename, np.reshape(X,(npx,npx)),delimiter=' ', fmt='%1.4e',header=template,comments='')
        f.close()
    if par[0][0]==3:
        num_layer=30#61 or 30
        X3d=np.reshape(X,(num_layer,61,61)) 
        filename = r'fmod1.mlt'
        template0='# MF2005 multiplication file \n'+str(num_layer)
        template1="""mult{nlayer:d}  \nINTERNAL 1.0 (free) 0"""
        f = open(filename, 'w')
        f.write(template0)
        f.write('\n')
        for i in range(0,num_layer): 
            context1= {"nlayer":i+1}
            f.write(template1.format(**context1))
            f.write('\n')
            np.savetxt(f, np.reshape(X3d[i,:,:],(61,61)), fmt='%1.4e')
        f.close()
    if par[0][0]==1:
        external_code='./mf2005 fmod.nam'  #Linux
        #external_code='mf2005.exe fmod.nam' # Windows
    elif par[0][0]==2:
        pass
    elif par[0][0]==3:
       #external_code='mf2005.exe fmod1.nam' # Windows
		 external_code='./mf2005 fmod1.nam'  #Linux
    os.system(external_code)
    if par[0][0]==1:
        #out=np.loadtxt('fmod.hed')
        with open("fmod.hed") as f:
            out = np.array(f.read().split(), dtype=float).reshape(npx, npx)
        idx=par[1][0]-1
        fx=np.zeros((1,n))
        for i in xrange(idx.shape[0]):
            fx[0,i]=out[idx[i,0],idx[i,1]]
        
    elif par[0][0]==2:
        pass
    
    elif par[0][0]==3:
        idx=par[1][0]
        fx=np.loadtxt('fmod1.sim',skiprows=1,usecols=(0,)).reshape((1,2352))
        fx=fx[0,idx]
    
    os.chdir(main_dir)
    
    return fx