# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:47:18 2018

@author: elaloy
"""

import numpy as np
from scipy.sparse import csr_matrix

def tomokernel_straight_2D(data,x,z):
    
    # This function computes the kernel matrix for a straight ray tomographic 
    # inversion given the data matrix and the x and z position vectors for the 
    # vertical and horizontal cell *boundaries*.
    #
    # translated from matlab code by James Irving (December 2005)

    # check that data are within bounds set by x and z
    xmin = x[0];  xmax = x[-1];  
    zmin = z[0];  zmax = z[-1];  
    
    if xmin > np.min(np.vstack((data[:,0],data[:,2]))) or \
    xmax < np.max(np.vstack((data[:,0],data[:,2]))) or \
    zmin > np.min(np.vstack((data[:,1],data[:,3]))) or \
    zmax < np.max(np.vstack((data[:,1],data[:,3]))):
        print('Error:  Data outside of range of min and max values')
        #return
    
    # determine some initial parameters
    dx = x[1]-x[0]                                  # horizontal discretization
    dz = z[1]-z[0]                                  # vertical discretization
#    xmid=np.arange((xmin+dx/2),(xmax-dx/2),dx)      # x-coordinates of cell midpoints
#    zmid = np.arange((zmin+dz/2),(zmax-dz/2),dz)    # z-coordinates of cell midpoints
    nrays = data.shape[0]                           # number of rays to consider
    nx = len(x)-1                               # number of cells in x-direction
    nz = len(z)-1                               # number of cells in z-direction
    
    # initialize the sparse storage arrays
    maxelem = np.int(np.round(nrays*np.sqrt(nx**2+nz**2)))
    irow = np.zeros((maxelem))
    icol = np.zeros((maxelem))
    jaco = np.zeros((maxelem))
    
    # determine elements of Jacobian matrix
    count = 0
    for i in range (0,nrays):
        xs = data[i,0]                              # x-position of source
        xr = data[i,2]                              # x-position of receiver
        zs = data[i,1]						           # z-position of source
        zr = data[i,3]						           # z-position of receiver
        if xs==xr:                                 # if ray is vertical, add for stability
            xr=xr+1e-10
        
        slope = (zr-zs)/(xr-xs)		                   # slope of raypath
        
        # vector containing x-positions of vertical cell boundaries hit by the ray,
        # and also the ray end points
        xcellb = x.flatten(order='F') # specifiying order='F' is likely not necessary
        idx=np.logical_and(xcellb > np.min([xs,xr]),xcellb < np.max([xs,xr]))
        xcellb = xcellb[idx]
        xcellb = np.append(xcellb,[xs,xr])
        
        # vector containing z-positions of horizontal cell boundaries
        # and also the ray end points
        zcellb = z.flatten(order='F')
        idx=np.logical_and(zcellb > np.min([zs,zr]),zcellb < np.max([zs,zr]))
        zcellb = zcellb[idx]
        zcellb = np.append(zcellb,[zs,zr])
        
        # form matrix containing all intersection points of ray with cell boundaries
        # then sort these points in order of increasing x-coordinate
        ip1 = np.append(xcellb, xs + (zcellb-zs)*1/(slope+1e-20))   # x-coords of all intersection points
        ip2 = np.append(zs + (xcellb-xs)*slope, zcellb)            # z-coords of all intersection points
        ipoint=np.vstack((ip1,ip2)).T
        ipoint = ipoint[ipoint[:,0].argsort()]
        
        # calculate length and midpoint of the ray bits between the intersection points
        xlength = np.abs(ipoint[1:,0]-ipoint[0:-1,0])      # x-component of length
        zlength = np.abs(ipoint[1:,1]-ipoint[0:-1,1])      # z component of length
        clength = np.sqrt(xlength**2 + zlength**2)
        
        cmidpt=0.5*np.vstack(( ipoint[0:-1,0]+ipoint[1:,0], ipoint[0:-1,1]+ipoint[1:,1] )).T
       
        # calculate which slowness cell each ray bit belongs to, and place properly in J matrix
        srow = np.ceil((cmidpt[:,0]-xmin)/dx)
        scol = np.ceil((cmidpt[:,1]-zmin)/dz)
        srow[srow<1] = 1;  srow[srow>nx] = nx
        scol[scol<1] = 1;  scol[scol>nz] = nz
        njaco = len(srow)
        irow[count:(count+njaco)] = (i+1)*np.ones((njaco))
        icol[count:(count+njaco)] = (scol-1)*nx+srow
        jaco[count:(count+njaco)] = clength
        count = count + njaco
        del ipoint   

    
    # convert sparse storage arrays to sparse matrix
    index = np.where(jaco)[0]
    irow = irow[index]-1
    icol = icol[index]-1
    jaco = jaco[index]
    
    J=csr_matrix((jaco, (irow, icol)), (nrays, nx*nz))
    
    return(J)

