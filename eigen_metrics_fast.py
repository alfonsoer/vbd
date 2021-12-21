#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:11:10 2021

@author: John Baxter
"""
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R

def thresholds(mu, sigma, FA, theta, dLdu, dLdo, dLdFA, dLdh, epsilon_mu, epsilon, erase_undetermined=True):
    mask_idx_zeros_mu     = np.where(mu < epsilon_mu) 
    mu[mask_idx_zeros_mu] = 0.0

    mask_idx_zeros_sigma    = np.where(sigma < epsilon) 
    sigma[mask_idx_zeros_sigma] = 0.0
    
    mask_idx_zeros_mu_sigma = np.where(np.logical_and( np.less( mu, epsilon) , np.less( sigma, epsilon) ))
    FA[mask_idx_zeros_mu_sigma] = 0.0    
    mask_idx_zeros_FA    = np.where(FA< epsilon) 
    FA[mask_idx_zeros_FA] = 0.0
        
    # dLdFA[mask_idx_zeros_mu] = 0.0
    # dLdFA[mask_idx_zeros_FA] = 0.0
    
    # dLdu[mask_idx_zeros_mu] = 0.0
    # dLdo[mask_idx_zeros_sigma] = 0.0
    # # mask_idx_zeros_theta    = np.where(np.abs(theta) < epsilon) 
    # dLdh[mask_idx_zeros_sigma]  = 0.0
    
    #Round to zero
    mask_dLdu = np.where(np.abs(dLdu) < epsilon) 
    dLdu[mask_dLdu] = 0.0
    mask_dLdo = np.where(np.abs(dLdo) < epsilon) 
    dLdo[mask_dLdo] = 0.0
    mask_dLdFA = np.where(np.abs(dLdFA) < epsilon) 
    dLdFA[mask_dLdFA] = 0.0
    mask_dLdh = np.where(np.abs(dLdh) < epsilon) 
    dLdh[mask_dLdh] = 0.0  
    
    #Zero undetermined voxels
    dLdFA[mask_idx_zeros_mu] = 0.0
    dLdFA[mask_idx_zeros_FA] = 0.0    
    
    # if erase_undetermined:
    #     dLdu[mu < epsilon] = 0.0
    #     dLdo[sigma < epsilon] = 0.0
    #     dLdFA[mu < epsilon] = 0.0
    #     dLdFA[FA < epsilon] = 0.0
    #     dLdFA[sigma < epsilon] = 0.0
    #     dLdh[sigma < epsilon] = 0.0
    
    
    return mu, sigma, FA, theta, dLdu, dLdo, dLdFA, dLdh

def decomp_3x3_sym_matrix(m,shortcut=None):
    
    a = m[0,0,:]
    b = m[1,1,:]
    c = m[2,2,:]
    d = m[0,1,:]
    e = m[1,2,:]
    f = m[0,2,:]


    return decomp_3x3_sym_matrix_comp(a,b,c,d,e,f)

#matrix is stored as a vector of 6 numbers [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)]
def decomp_3x3_sym_matrix_comp(a,b,c,d,e,f):
    
    n = a.shape[-1]
    eig = np.zeros((3,n))
    x0 = a+b+c
    #print(x0)
    x1 = np.clip(a*a+b*b+c*c-a*b-a*c-b*c+3*(d*d+e*e+f*f),a_min=0,a_max=None)
    #print(x1)
    x2 = -(3*a-x0)*(3*b-x0)*(3*c-x0)+9*((3*c-x0)*d*d+(3*b-x0)*f*f+(3*a-x0)*e*e)-54*d*e*f
    #print(x2)
    phi = np.arctan2(np.sqrt(np.clip(4*x1**3-x2**2,a_min=0,a_max=None)),x2)
    
    eig[0,:] = (x0-2*np.sqrt(x1)*np.cos(phi/3))/3
    eig[1,:] = (x0+2*np.sqrt(x1)*np.cos((phi-np.pi)/3))/3
    eig[2,:] = (x0+2*np.sqrt(x1)*np.cos((phi+np.pi)/3))/3
    
    #bubble sort eigenvalues
    swap = np.maximum(eig[0,:],eig[1,:])
    eig[1,:] = np.minimum(eig[0,:],eig[1,:])
    eig[0,:] = swap
    swap = np.maximum(eig[1,:],eig[2,:])
    eig[2,:] = np.minimum(eig[1,:],eig[2,:])
    eig[1,:] = swap
    swap = np.maximum(eig[0,:],eig[1,:])
    eig[1,:] = np.minimum(eig[0,:],eig[1,:])
    eig[0,:] = swap
    
    #get eigenvectors
    v = np.zeros((3,3,n))
    v[0,0,:] = np.logical_or(f != 0, d != 0)*((eig[0,:]-c)*(f*(b-eig[0,:])-d*e)-e*(d*(c-eig[0,:])-e*f)) + np.logical_and(f == 0, d == 0)
    v[1,0,:] = np.logical_or(f != 0, d != 0)*(eig[0,:] > 0)*f*(d*(c-eig[0,:])-e*f)
    v[2,0,:] = np.logical_or(f != 0, d != 0)*(eig[0,:] > 0)*f*(f*(b-eig[0,:])-d*e)
    vMag = np.sqrt(v[0,0,:]**2+v[1,0,:]**2+v[2,0,:]**2)
    v[:,0,:] /= vMag
    v[0,1,:] = np.logical_or(e != 0, d != 0)*((eig[1]-c)*(f*(b-eig[1])-d*e)-e*(d*(c-eig[1])-e*f))
    v[1,1,:] = np.logical_or(e != 0, d != 0)*f*(d*(c-eig[1])-e*f) + np.logical_and(e == 0, d == 0)
    v[2,1,:] = np.logical_or(e != 0, d != 0)*f*(f*(b-eig[1])-d*e)
    vMag = np.sqrt(v[0,1,:]**2+v[1,1,:]**2+v[2,1,:]**2)
    v[:,1,:] /= vMag
    v[0,2,:] = v[1,0,:]*v[2,1,:]-v[1,1,:]*v[2,0,:]
    v[1,2,:] = v[0,1,:]*v[2,0,:]-v[0,0,:]*v[2,1,:]
    v[2,2,:] = v[0,0,:]*v[1,1,:]-v[0,1,:]*v[1,0,:]
    
    return eig, v

def push_grad_to_eigen(dLdx,e,R):
    dLde = np.zeros_like(e)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                dLde[k,:] += dLdx[i,j,:]*R[i,k,:]*R[j,k,:]
                
    #get Euler angle rep
    dLdR = np.zeros_like(e)
    a2 = -np.arcsin(R[2,0,:])
    a1 = np.arctan2(R[2,1,:]/np.cos(a2),R[2,2,:]/np.cos(a2))
    a3 = np.arctan2(R[1,0,:]/np.cos(a2),R[0,0,:]/np.cos(a2))
    
    R_a1 = np.zeros((e.shape[-1],3,3))
    R_a1[:,0,0] = 1
    R_a1[:,1,1] = np.cos(a1)
    R_a1[:,1,2] = -np.sin(a1)
    R_a1[:,2,1] = np.sin(a1)
    R_a1[:,2,2] = np.cos(a1)
    R_a2 = np.zeros((e.shape[-1],3,3))
    R_a2[:,1,1] = 1
    R_a2[:,0,0] = np.cos(a2)
    R_a2[:,0,2] = np.sin(a2)
    R_a2[:,2,0] = -np.sin(a2)
    R_a2[:,2,2] = np.cos(a2)
    R_a3 = np.zeros((e.shape[-1],3,3))
    R_a3[:,2,2] = 1
    R_a3[:,0,0] = np.cos(a3)
    R_a3[:,0,1] = -np.sin(a3)
    R_a3[:,1,0] = np.sin(a3)
    R_a3[:,1,1] = np.cos(a3)
    
    dR_a1 = np.zeros((e.shape[-1],3,3))
    dR_a1[:,1,1] = -np.sin(a1)
    dR_a1[:,1,2] = -np.cos(a1)
    dR_a1[:,2,1] = np.cos(a1)
    dR_a1[:,2,2] = -np.sin(a1)
    dR_a2 = np.zeros((e.shape[-1],3,3))
    dR_a2[:,0,0] = -np.sin(a2)
    dR_a2[:,0,2] = np.cos(a2)
    dR_a2[:,2,0] = -np.cos(a2)
    dR_a2[:,2,2] = -np.sin(a2)
    dR_a3 = np.zeros((e.shape[-1],3,3))
    dR_a3[:,0,0] = -np.sin(a3)
    dR_a3[:,0,1] = -np.cos(a3)
    dR_a3[:,1,0] = np.cos(a3)
    dR_a3[:,1,1] = -np.sin(a3)
    
    dRda1 = R_a3 @ R_a2 @ dR_a1
    dRda2 = R_a3 @ dR_a2 @ R_a1
    dRda3 = dR_a3 @ R_a2 @ R_a1
    e = np.expand_dims(np.eye(3),0) * np.expand_dims(e.T,1)
    R = R.transpose((2,0,1))
    RT = R.transpose((0,2,1))
    eRT = e@RT
    Re = R@e
    
    dxda1 = dRda1@eRT + Re@dRda1.transpose(0,2,1)
    dxda2 = dRda2@eRT + Re@dRda2.transpose(0,2,1)
    dxda3 = dRda3@eRT + Re@dRda3.transpose(0,2,1)
    
    # print(dLdx.transpose((2,0,1)))
    # print(dxda1)
    # print(dxda2)
    # print(dxda3)
    
    dLdR[0,:] = np.sum(dLdx*dxda1.transpose((1,2,0)),axis=(0,1))
    dLdR[1,:] = np.sum(dLdx*dxda2.transpose((1,2,0)),axis=(0,1))
    dLdR[2,:] = np.sum(dLdx*dxda3.transpose((1,2,0)),axis=(0,1))
    
    return dLde, dLdR

def eigen_grad_to_metric_grad(dLde, e, mu, sigma):
    dLdu = np.sum(dLde,0)
    dLdo = np.sum((e-mu)*dLde,0)/sigma
    dLdh = ((e[1,:]-e[2,:])*dLde[0,:]+(e[2,:]-e[0,:])*dLde[1,:]+(e[0,:]-e[1,:])*dLde[2,:])/(np.sqrt(3)*sigma)
    return dLdu, dLdo, dLdh/sigma

def eigen_to_metric(e):
    mu = np.sum(e,0)/3
    sigma = np.sqrt(np.sum((e-mu)**2,0))
    theta = np.arctan2((e[1,:]-e[2,:]),np.sqrt(3)*(e[0,:]-mu))
    return mu, sigma, theta


def get_metrics(dti, gradients, epsilon = 1e-9, epsilon_mu =  1e-9):
    
    #input data is 9x  sagittal x coronal x axial
    sample_shape = dti.shape
    tensors  = sample_shape[0]
    sagittal = sample_shape[1]
    coronal  = sample_shape[2]
    axial    = sample_shape[3]    

    n_matrices = sagittal * coronal * axial
    dti        = dti.reshape(3,3, n_matrices)    
    dLdx       = gradients.reshape(3,3, n_matrices)    
    
    #Save the dLdx
    
    e, v = decomp_3x3_sym_matrix(dti)
    dLde, dLdR = push_grad_to_eigen(dLdx,e,v)
    
    mu, sigma, theta = eigen_to_metric(e)
    dLdu, dLdo, dLdh = eigen_grad_to_metric_grad(dLde, e, mu, sigma)

    #Compute the FA and dLdFA    
    FA = np.sqrt(3/2.0) * sigma/(np.sqrt(3*mu**2+sigma**2))
   
    dLdFA = ((3*mu**2 + sigma**2)/(3*mu*FA))* ((sigma/mu)*(dLdo) - dLdu)


    #Check if any negative values for MD, A and FA
    
    
    #Make zeroall  values less than epsilon
    mu, sigma, FA, theta, dLdu, dLdo, dLdFA, dLdh = thresholds(mu, sigma, FA, theta,
                                                        dLdu, dLdo, dLdFA, 
                                                        dLdh, epsilon_mu, epsilon)

    #Reshape to sagittal , coronal , axial
    mu_r    = mu.reshape(sagittal , coronal , axial)
    sigma_r = sigma.reshape(sagittal , coronal , axial)
    theta_r = theta.reshape(sagittal , coronal , axial)
    dLdu    = dLdu.reshape(sagittal , coronal , axial)
    dLdo    = dLdo.reshape(sagittal , coronal , axial)
    dLdh    = dLdh.reshape(sagittal , coronal , axial)
    
    FA     = FA.reshape(sagittal , coronal , axial)
    dLdFA  = dLdFA.reshape(sagittal , coronal , axial)  
    
    dLdR_1 = dLdR[0].reshape(sagittal , coronal , axial)    
    dLdR_2 = dLdR[1].reshape(sagittal , coronal , axial)    
    dLdR_3 = dLdR[2].reshape(sagittal , coronal , axial)  

    #Debug
    # print('dLde (eigen_metrics);', dLde.shape)
    dLde1    = dLde[0].reshape(sagittal , coronal , axial)
    dLde2    = dLde[1].reshape(sagittal , coronal , axial)
    dLde3    = dLde[2].reshape(sagittal , coronal , axial)
    
    # print('dLdx (eigen_metrics);', dLdx.shape)
     
    measures = {'MD':mu_r, 'A':sigma_r, 
                'PsPl':theta_r, 'dLdu':dLdu, 
                'dLdo':dLdo, 'dLdh':dLdh, 
                'FA': FA, 'dLdFA': dLdFA, 
                'dLdR1': dLdR_1,
                'dLdR2': dLdR_2,
                'dLdR3': dLdR_3,
                
                #debug
                'dLde1':  dLde1,
                'dLde2':  dLde2,
                'dLde3':  dLde3,
                }
    

    
    
    for key in measures.keys():
        #Mask values
        #measures[key] [mask_idx_zeros_mu] = 0
        measures[key][np.isinf(measures[key])]=0.0
        measures[key] = np.nan_to_num(measures[key], 0.0)       
        idx = np.where(np.abs(measures[key])<epsilon)
        measures[key][idx] = 0.0
       
    return measures

if __name__ == "__main__": 
    time_start=np.datetime64('now')
    
    for i in range(10):
        
        ortho  = R.random().as_matrix()
        ortho2 = R.random().as_matrix()
        eigen = np.zeros((3))
        eigen[2] = abs(np.random.normal())
        eigen[1] = eigen[2]+abs(np.random.normal())
        eigen[0] = eigen[1]+abs(np.random.normal())+2
        matrix = np.matmul(np.matmul(ortho, np.diag(eigen)), ortho.transpose())
        eigen_perturb = np.random.normal(size=3)
        dLdx = np.matmul(np.matmul(ortho,np.diag(eigen_perturb)),ortho.transpose())
        #dLdx = np.diag(eigen_perturb)
        
        e, v = decomp_3x3_sym_matrix(matrix)
        dLde = push_grad_to_eigen(dLdx,v)
        
        #order eigen
        e_copy = np.zeros(3)
        dLde_copy = np.zeros(3)
        if e[0] > e[1]:
            if e[0] > e[2]:
                e_copy[0] = e[0]
                dLde_copy[0] = dLde[0]
                if e[1] > e[2]:
                    e_copy[1] = e[1]
                    dLde_copy[1] = dLde[1]
                    e_copy[2] = e[2]
                    dLde_copy[2] = dLde[2]
                else:  
                    e_copy[1] = e[2]
                    dLde_copy[1] = dLde[2]
                    e_copy[2] = e[1]
                    dLde_copy[2] = dLde[1]
            else:
                e_copy[0] = e[2]
                dLde_copy[0] = dLde[2]
                e_copy[1] = e[0]
                dLde_copy[1] = dLde[0]
                e_copy[2] = e[1]
                dLde_copy[2] = dLde[1]
        else:
            if e[0] > e[2]:
                e_copy[0] = e[1]
                dLde_copy[0] = dLde[1]
                e_copy[1] = e[0]
                dLde_copy[1] = dLde[0]
                e_copy[2] = e[2]
                dLde_copy[2] = dLde[2]
            else:
                e_copy[2] = e[0]
                dLde_copy[2] = dLde[0]
                if e[1] > e[2]:
                    e_copy[0] = e[1]
                    dLde_copy[0] = dLde[1]
                    e_copy[1] = e[2]
                    dLde_copy[1] = dLde[2]
                else:  
                    e_copy[0] = e[2]
                    dLde_copy[0] = dLde[2]
                    e_copy[1] = e[1]
                    dLde_copy[1] = dLde[1]
        e = e_copy
        dLde = dLde_copy
        
        mu, sigma, theta, dLdu, dLdo, dLdh = eigen_grad_to_metric_grad(dLde, -np.sort(-e))
        
        print(matrix)
        print(dLdx)
        
        #print(eigen_perturb)
        print(e)
        print(dLde)
        print(mu, sigma, theta, dLdu, dLdo, dLdh)
        
        print("")
        
        #print(abs(md_true-md_calc))
        #print(abs(a_true-a_calc))
        #print("")
    time_end=np.datetime64('now')
    
    print(time_end-time_start)

