# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:25:13 2020

@author: Alfonso Estudillo Romero
"""
import numpy as np
import torch
from torchvision import datasets, transforms
import os
import nrrd

# import torchio
# from torchio.transforms import RandomAffine
import scipy.stats

from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from utils.dti_data import data_dir
# from datetime  import datetime
from preprocessing.rotate_dti import flipTensors
from preprocessing.rotate_dti import rotate_dti_tensor
from preprocessing.rotate_dti import show_tensor

from sys import exit

"""For DTI data extracted from PPMI subjects, the dimensions are: 72, 116, 116, 3, 3
    Axial, Coronal, Sagittal, 3x3 Tensor
   1. Reshape the 3x3 Tensor as 9 chanels
      a. Remove matrix coefficients: 
          tensor = 
          array([ 7.38755916e-04, -1.01526864e-04,  6.77212447e-05, -1.01526864e-04,
                 8.20228073e-04, -4.52717941e-05,  6.77212447e-05, -4.52717941e-05, 5.11464896e-04], dtype=float32)
              [3,6,7]
          tensor = array([ 7.38755916e-04, -1.01526864e-04,  6.77212447e-05,  8.20228073e-04,
                          -4.52717941e-05,  5.11464896e-04], dtype=float32)
   2. Permute to have B x 6  x 72 x 116 x 116 as required by pytorch (batch, chanels, depth, height, width)
   
"""


# rotate_data = False
# shift_data  = True

da_on_minority = False
# required_shape= [96, 112, 96] #defformed dtis 
# required_shape= [96, 112, 72] #original dtis
# #
# zero_padded =  [96, 112, 96]

# required_shape= [96, 112, 96] #defformed dtis 
# required_shape= [112, 112, 112] #original dtis
# #
# zero_padded =  [112, 112, 112]

#required_shape= [96, 112, 96]

# max_angle = 4
# max_shifts_ct = 4
# max_shifts_pd = 4
# def getScores(scores_df, subject_id, session_id):
    
#     UPDRS3, NHY, SE = list(scores_df.loc[ (scores_df['Sessions'] == session_id) & (scores_df['Subjects'] == subject_id),  ['UPDRS3', 'NHY', 'MSEADLG'] ].iloc[0])
#     #Normalize all
#     return  np.array( [UPDRS3, 
#                        # NHY,
#                        # SE
#                       ]
#                      )

def sample_resize(sample, required_shape):
    shape = sample.shape[1:]
    diff  = np.floor(np.subtract(required_shape, shape)/2).astype('int')
    
    typ = sample.dtype
    X = np.zeros([sample.shape[0]] +  required_shape).astype(typ)
    #Remove slices
    if diff[0]<0:
        x1i = 0 
        x1f = required_shape[0]
        
        s1i = - diff[0]
        s1f = s1i + required_shape[0]
    #Append slices
    else:
        x1i = diff[0]
        x1f = x1i + sample.shape[1]
        
        s1i = 0
        s1f = sample.shape[1]       
        
    if diff[1]<0:
        x2i = 0 
        x2f = required_shape[1]
        
        s2i = - diff[1]
        s2f = s2i + required_shape[1]
    #Append slices
    else:
        x2i = diff[1]
        x2f = x2i + sample.shape[2]
        
        s2i = 0
        s2f = sample.shape[2]   
        
    if diff[2]<0:
        x3i = 0 
        x3f = required_shape[2]
        
        s3i = - diff[2]
        s3f = s3i + required_shape[2]
    #Append slices
    else:
        x3i = diff[2]
        x3f = x3i + sample.shape[3]
        
        s3i = 0
        s3f = sample.shape[3]          
        
        
    X[:, x1i:x1f, x2i:x2f, x3i:x3f] = sample[:, s1i:s1f, s2i:s2f, s3i:s3f]

    return X    


def npy_load(file_path, required_shape, dataset,
              Pat_Domside=None,
              flip_pds=False,
              flip_controls=False, 
              stage='train'):

    prefix = ''
    if dataset=='ppmi_orig_npy' or 'post_registered' in dataset or 'post_reg_pds' in dataset :
        prefix = 'dti_re_' 
    elif 'pre_registered' or dataset=='ppmi_orig_npy_right'  in dataset or 'pre_reg_pds' in dataset :
        prefix = 'dti_MNI_2mm_' 
    
    file_path = os.path.realpath(file_path)
    if flip_pds is True  and Pat_Domside is not None:            
        subject_id = os.path.basename(file_path).split(prefix)[1].split('_')[0]  
        if int(subject_id) in Pat_Domside:
            file_path = file_path.replace('/pd/', '/pdflip/')
    
    if flip_controls is True and '/ct/' in file_path and stage=='train':
        flip_control = np.random.normal(size=(1,)) 
        if flip_control>0:        
            file_path = file_path.replace('/ct/', '/ctflip/')

    #Get scores
    # scores = []
    # if scores_df is not None:
        # subject_id = int(os.path.basename(file_path).split(prefix)[1].split('_')[0]  )
        # session_id = os.path.basename(file_path).split(prefix)[1].split('_')[1].split('.')[0]  
        # print('subject_id', subject_id, 'session_id',session_id)
        # scores = getScores(scores_df, subject_id, session_id)
    
    sample = np.load(file_path, allow_pickle=True)    
    if sample is None:
        print('********    Could not load ', file_path)
        exit(0)
    
    #Remove redundant dimensions pass it to the data loader
    # sample=sample[[0,1,2,4,5,8],:,:,:] #COMMENTED ALREADY REMOVED REDUNDANT DATA
    orig_dims = sample.shape
    # print('Orig dims', orig_dims)

    #Remove or add extra slices (Zero padding)
    sample = sample_resize(sample, required_shape)
        
    return sample, orig_dims



def nhdr_npy_loader_da_john(dataset, flip_data=False, Pat_Domside=None,
                            stage='train', random_flip_on_controls = False,
                            device='cpu', angle_range = 10, shifts_range = 6, required_shape=[96, 112, 96]):
    
    def dti_preprocess(file_path):
        
        sample, orig_dims = npy_load(file_path, required_shape, dataset,
                                      Pat_Domside = Pat_Domside,
                                      flip_pds=flip_data, 
                                      flip_controls=random_flip_on_controls,
                                      stage=stage, )
        
        #Sample is in (tensor x sagittal x coronal x axial) form, need to permute to (tensor x axial x coronal x sagittal)
        # sample = sample.transpose(1, 2, 3, 0)
        sample = sample.transpose(0,3,2,1) #(tensor x axial x coronal x sagittal)
        dims = {}
        
        dims['sagittal_original'] = orig_dims[1]
        dims['coronal_original']  = orig_dims[2]
        dims['axial_original']    = orig_dims[3]
        
        dims['sagittal'] = sample.shape[3]
        dims['coronal']  = sample.shape[2]
        dims['axial']    = sample.shape[1]        
        
        #move to GPU
        # sample = torch.from_numpy(sample)
        
        return sample, file_path, "", dims 
    
    return dti_preprocess



# def npy_load(file_path, required_shape):
#     sample = np.load(os.path.realpath(file_path), allow_pickle=True)    
#     if sample is None:
#         print('********    Could not load ', file_path)
#         exit(0)
    
#     #Remove redundant dimensions pass it to the data loader
#     sample=sample[[0,1,2,4,5,8],:,:,:]
#     orig_dims = sample.shape

#     #Remove or add extra slices (Zero padding)
#     # t = time.time()
#     sample = sample_resize(sample, required_shape)
#     # print('elapsed time: ', time.time()-t)
        
#     return sample, orig_dims



# def nhdr_npy_loader_da_john(dataset, flip_data=False, Pat_Domside=None,
#                             stage='train', random_flip_on_controls = False,
#                             device='cpu', angle_range = 10, shifts_range = 6, required_shape=[96, 112, 96]):
    
#     def dti_preprocess(file_path):
#         sample, orig_dims = npy_load(file_path, required_shape)
        
#         #Sample is in (tensor x sagittal x coronal x axial) form, need to permute to (tensor x axial x coronal x sagittal)
#         # sample = sample.transpose(1, 2, 3, 0)
#         sample = sample.transpose(0,3,2,1) #(tensor x axial x coronal x sagittal)
#         dims = {}
#         dims['sagittal_original'] = orig_dims[3]
#         dims['coronal_original']  = orig_dims[2]
#         dims['axial_original']    = orig_dims[1]
        
#         #move to GPU
#         sample = torch.from_numpy(sample)
        
#         #Flip Parkinson group
#         if Pat_Domside is not None and flip_data is True :
#             if dataset=='ppmi_orig_npy' or 'post_registered' in dataset:
#                 prefix = 'dti_re_' 
#             elif 'pre_registered' or dataset=='ppmi_orig_npy_right'  in dataset:
#                 prefix = 'dti_MNI_2mm_'             

#             subject_id = os.path.basename(file_path).split(prefix)[1].split('_')[0]
#             if int(subject_id) in Pat_Domside:
#                 sample = flipTensors(sample,3) #Flip the tensors

#         if random_flip_on_controls is True and '/ct/' in file_path and stage=='train':
#             flip_control = np.random.normal(size=(1,)) 
#             if flip_control>0 :
#                 sample = flipTensors(sample,3) #Flip the tensors
            
#         #rotate/translate data if in training
#         #if stage == 'train':
#         #    if angle_range > 0 or shifts_range > 0:
#         #        angle = angle_range * (1-2*np.random.uniform(size=(1)))[0] *np.pi/180
#         #        xshift = shifts_range * (1-2*np.random.uniform(size=(1)))[0]
#         #        yshift = shifts_range * (1-2*np.random.uniform(size=(1)))[0]
#         #        sample = rotate_dti_tensor(torch.unsqueeze(sample,0), angle, xshift, yshift).squeeze()

#         # print('x_batch input shqpe: ', sample.shape)
#         # show_tensor(sample[:,:,:,:])        
        
#         return sample, file_path, "", dims  
#     return dti_preprocess


def UPDRS3_balancing(UPDRS3):
    

#0-10: 0
#10-35: 1
#35-45: 2
#45-60: 3
    # if UPDRS3<10:
    #     return 0
    # if UPDRS3>=10 and UPDRS3<35:
    #     return 1
    # if UPDRS3>=35 and UPDRS3<45:
    #     return 2    
    # if UPDRS3>=45 and UPDRS3<=60:
    #     return 3    
    # return 4
    #For each sample, 
    if UPDRS3<16:
        return 0
    if UPDRS3>=16 and UPDRS3<26:
        return 1
    if UPDRS3>=26 and UPDRS3<36:
        return 2    
    if UPDRS3>=36 and UPDRS3<46:
        return 3    
    if UPDRS3>=46:
        return 4
    return 5



def load_custom_dataset(grayScaled=True, batch_size_tr = 4, batch_size_te = 4, 
                        envname='', data_set_name='ppmi',ds_splited='', 
                        gradient_comp=False, k_fold=None, 
                        flip_data=False,
                        Pat_Domside=None, flip_controls=False, device='cpu',
                        angle_range = 0, shifts_range = 0, required_shape=[96, 112, 96], root_path=None,
                        scores_df = False):
    
    if root_path is None:
       # root_path  = os.path.join(data_dir(), ds_splited)
       root_path  = data_dir()
    
    #Both if are the same remove it
    
    if data_set_name=='ppmi_orig_npy_right' or 'post_registered' in data_set_name or 'pre_registered' in data_set_name or 'post_reg_pds' in data_set_name or 'pre_reg_pds' in data_set_name or data_set_name == 'ppmi_orig_npy':
        loader_tr = nhdr_npy_loader_da_john (data_set_name, 
                                             flip_data=flip_data, 
                                             Pat_Domside=Pat_Domside, 
                                             stage='train',
                                             device=device,
                                             required_shape=required_shape,
                                             random_flip_on_controls=flip_controls,
                                             angle_range = angle_range, shifts_range = shifts_range,)
        loader_te = nhdr_npy_loader_da_john (data_set_name,
                                             flip_data=flip_data, 
                                             Pat_Domside=Pat_Domside,
                                             stage='test', device=device,
                                             required_shape=required_shape,
                                             random_flip_on_controls=False,
                                             angle_range = 0, shifts_range = 0, )
        extension = '.npy'


    else:
        print('Check dataset name')
        exit()

    if k_fold is not None:
        data_set_name += '/fold_'  +  str(k_fold) 

    if gradient_comp is True:
        loader_tr = loader_te

        
    trainset = datasets.DatasetFolder(root = os.path.join(root_path, data_set_name, 'train') , 
                                      transform=None, loader=loader_tr, extensions=extension)
         
    testset = datasets.DatasetFolder(root =  os.path.join(root_path, data_set_name, 'test') ,
                                      transform=None, loader=loader_te, extensions=extension)
    
    num_clases = len(trainset.classes)
    if scores_df is not None:
        bins = 25
        num_clases = bins
        id_s = 0
        for samp in trainset.samples:
            session_id = samp[0].split('/')[-1].split('.')[0].split('_')[-1]
            UPDRS3, NHY, SE = scores_df.loc[ (scores_df['Sessions'] == session_id),['UPDRS3', 'NHY', 'MSEADLG'] ].iloc[0].tolist()
            # cla = UPDRS3_balancing(UPDRS3)
            trainset.samples[id_s] = (samp[0], UPDRS3)
            trainset.targets[id_s] = UPDRS3
            id_s += 1
            
        id_s = 0
        for samp in testset.samples:
            session_id = samp[0].split('/')[-1].split('.')[0].split('_')[-1]
            UPDRS3, NHY, SE = scores_df.loc[ (scores_df['Sessions'] == session_id),['UPDRS3', 'NHY', 'MSEADLG'] ].iloc[0].tolist()
            # cla = UPDRS3_balancing(UPDRS3)
            testset.samples[id_s] = (samp[0], UPDRS3)
            testset.targets[id_s] = UPDRS3
            id_s += 1            
            
        hist, edge_bins = np.histogram(trainset.targets, bins=bins)
        centers = edge_bins[0:-1]+np.diff(edge_bins)/2
        hist_d, edge_bins_d = np.histogram(trainset.targets, bins=bins, density=True)

        #Create GT distributions for each bin, centered at the center
        standard_deviation = 2
        gts=[]
        for (i, cent) in enumerate (centers):
            y_values = scipy.stats.norm(cent, standard_deviation)
            gts.append(y_values.pdf(centers)/sum(y_values.pdf(centers))) #Added probabilities
        gts = np.asarray(gts) 
        #Need to store the centers        
        id_s = 0
        for samp in trainset.samples:
            UPDRS3 = samp[1]
            cla = np.digitize(UPDRS3, centers,right=True)
            if cla>=num_clases:
                cla=num_clases-1
            trainset.targets[id_s] = cla
            trainset.samples[id_s] = (samp[0], [UPDRS3, cla]) #File, UPDRS3, BinCenter
            id_s += 1            
        
        id_s = 0
        for samp in testset.samples:
            UPDRS3 = samp[1]
            cla = np.digitize(UPDRS3, centers,right=True)
            if cla>=num_clases:
                cla=num_clases-1
            testset.targets[id_s] = cla
            testset.samples[id_s] = (samp[0], [UPDRS3, cla]) #File, UPDRS3, BinCenter
            id_s += 1      
    
    class_sample_count = []
    x = range(0, num_clases) 
    for n in x:
        class_sample_count.append(trainset.targets.count(n)) 
    
    weights_Cls = 1 / torch.Tensor(class_sample_count)
    weights_Cls = weights_Cls.double()
    weights = weights_Cls[trainset.targets]
    
    ns = int(sum(class_sample_count))#Total samples
    
    orig_nbatches = np.ceil(ns/batch_size_tr)
    Extra_Batches = int(2*orig_nbatches) #We will skip unbalanced batches at training
    if num_clases <=1:
        Extra_Batches = 0
    Extra_Batches = 0
    # Extra_Batches = 50 #We will skip unbalanced batches at training
    nbatches = orig_nbatches + Extra_Batches
    oversampling = int((nbatches)*batch_size_tr)
    num_samples = oversampling
    # num_samples = max(int(sum(class_sample_count)), oversampling)
    
    #Whith replacement = False
    replacement =True
    if replacement is False:
        Extra_Batches  = 0
        nbatches = orig_nbatches
        num_samples = ns
    
    
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples, replacement = replacement)
    
    trainset_loader = None    
    testset_loader = None
    if gradient_comp is True:
        trainset_loader  = torch.utils.data.DataLoader(trainset, batch_size = batch_size_te, shuffle =False)  
        testset_loader   = torch.utils.data.DataLoader(testset, batch_size = batch_size_te,  shuffle = False)     
    else:
        trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size_tr, 
                                                  shuffle=False, sampler = sampler, 
                                                  num_workers=6, 
                                                  worker_init_fn=worker_init_fn)
    
        testset_loader  = torch.utils.data.DataLoader(testset, batch_size = batch_size_te, shuffle = False,
                                                      num_workers=1)        
        
    #Assume all images have the same size
    #0 First image
    #0 First tensor #1 image path
    try:
        # dims = trainset.__getitem__(0)[0][0].shape
        # dims = torch.tensor( np.array([6,96,112,96]))
        dims = torch.Size([6] + required_shape)
        pack = {'trainset_loader': trainset_loader,
            'testset_loader': testset_loader,
            'weights' : torch.DoubleTensor(weights_Cls) ,
            'dims': dims,
            'Extra_Batches': Extra_Batches,
            'data_set_num_samples': num_samples,
            'tr_nbatches': nbatches,
            'orig_nbatches': orig_nbatches,
            'te_nbatches': np.ceil(len(testset.samples)/batch_size_te),
            'dist_gt':gts,
            'centers_gt':centers
            }        
    except:
        print('Review sample required dimensions') 
        pack = None
    
    return pack

def worker_init_fn(worker_id):
    # dt = datetime.now()
    # micros = int(str(dt.microsecond)[-1])
    np.random.seed(np.random.get_state()[1][0] + worker_id )
