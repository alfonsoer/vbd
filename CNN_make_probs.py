#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:30:33 2021

@author: Alfonso Estudillo Romero
"""
import os
import time
import torch
import sys
import nrrd
import copy
import pandas as pd
import torch.nn as nn
from torchsummary import summary


from CNN_fit import evaluate_model
from utils.dti_data import createDir
from cuda import get_device
from CNN_config import  getHyperParameterString
from CNN_datasetsPPMI import load_custom_dataset


def im_info(info, Ntensors = 6):
    orig_shape  = [Ntensors, info['sagittal_original'], 
                    info['coronal_original'], 
                    info['axial_original'] ]
    rems = {}
    rems['remS'] = info['sagittal_original'] - info['sagittal']
    rems['remC'] = info['coronal_original'] - info['coronal']
    rems['remA'] = info['axial_original']- info['axial']
    
    return orig_shape, rems

def ComputeGrads(model, optimizer, dat_loader, loss_function_str, loss_fn, device,
                 subset, subject_split_expr, out_folder_name, save_as, 
                  fold_id, subjs):
    
    t = time.time()
    model.eval() 
    
    model_predict = evaluate_model(model, loss_fn, loss_function_str)
                
    labels_true = []
    labels_pred = []
    
    for batch_num, (imgs, labels) in enumerate (dat_loader, 1): 
        
        
        x_batch   = imgs[0]   #Add [0] when using ppmi dataset 
        file_name = imgs[1][0]
 
        #Check if file exists
        if not os.path.isfile(file_name):
            print('file does not exist: ', file_name)
            continue
        
        y_batch = labels
        target_class = int(y_batch)
        cohort = 'pd'
        if target_class == 0:
            cohort = 'ct'                        

        #Move data to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)        
        
        #Zeroes all gradients
        model.zero_grad()
        
        
        x_batch.requires_grad = True                    
        #Evaluate model
        loss, f1, ps, rs, y, yhat_logit, yhat = model_predict(x_batch, y_batch)                
    
        if loss_function_str == 'BCELoss' or loss_function_str =='BCEWithLogitsLoss':
            one_hot_output = torch.FloatTensor(1).zero_()
            one_hot_output[0] = 1
            if cohort=='ct':
                one_hot_output = -1.0*one_hot_output
            yhat.backward(gradient=one_hot_output, retain_graph=True)
        else:
            one_hot_output = torch.FloatTensor( 1, 2).zero_()
            one_hot_output[0][int(target_class)] = 1
            yhat.backward(gradient=one_hot_output, retain_graph=True)
    

        x_batch.grad.zero_()
        optimizer.zero_grad() 
        model.zero_grad()         
        
        #Store predictions
        labels_true.extend(y_batch.numpy().tolist())
        labels_pred.extend(yhat_logit.tolist())      

        # output_folder_name = os.path.join('/'.join(file_name.split('/')[:-3]), out_folder_name)
        
        
        # createDir(output_folder_name, False)
        # folder = os.path.join(output_folder_name, subset)
        # createDir(folder, False)
        # folder = os.path.join(folder, cohort)
        # createDir(folder, False)
        
        # if not os.path.exists(folder):
        #     os.makedirs(folder)        
            
        subjs.append({ 'subject':file_name.split('/')[-1], 'cohort': cohort , 'label_true':y_batch.numpy()[0] , 'label_pred':yhat_logit[0], 'prob': yhat.detach().numpy()[0], 'fold': fold_id,'filename': file_name})

        print('Computed batch', batch_num)
                 
    print('Finished after: ', str((time.time()-t)/3600), ' hours')   

    return subjs
pre_registered_arch = {
        'num_classes' : 2, 
        'batch_size_tr' :12,
        'batch_size_te' : 1,
        'loss_function' : 'BCELoss',        
        'sel_optimizer' : 'AdamW', #Adam with L2 regularizer
        
        'net_model' : ['config8e',   '64_128_hid_xY_f', '',  [0,]],
                       
        #64_128_hid_xY : Before fixing the transpose                

        'required_shape' : [96, 112, 96],
        
      #PARAMETROS CORRIENDO EL 17 Agosto
        'dropouts'  :      0.1,
        'learnings'       :0.00001,  
        
        'adtional_prefix' :'BN',
       'epochs':160,
        'step_size' :50, 
        'gamma' : 0.96,  
        
      'L2_weighting' : 0.01,
        'min_acc_to_save': 0.8,
       'freeze_lr_at': 400, #43, #900,
       'save_last_iteration_model': True,
        'SEED': 1267,
        'override': True,
        

        # 'data_set_name' :   [  'post_registered10folds2', 'post_registered10folds3', 'post_registered10folds4'],
        # 'data_set_name' :   [  'post_registered10folds0', 'post_registered10folds5', 'post_registered10folds6', 
        #                        'post_registered10folds7'],
        
        #'data_set_name' :   [  'post_registered10folds2', 'post_registered10folds3', 'post_registered10folds4'],
        # 'data_set_name' :   [  'post_registered10folds0', 'post_registered10folds1', 'post_registered10folds8','post_registered10folds9'],
        #'data_set_name' :   [  'post_registered10folds0', 'post_registered10folds1', 'post_registered10folds2','post_registered10folds3',
        #                       'post_registered10folds4', 'post_registered10folds5', 'post_registered10folds6','post_registered10folds7', 
        #                       'post_registered10folds8','post_registered10folds9'],
 
        # 'data_set_name' :   [ 
        #                     # 'pre_registered10folds0','pre_registered10folds1',
        #                    #'pre_registered10folds2','pre_registered10folds3' , 
        #                   #'pre_registered10folds4',
        #                  #'pre_registered10folds5' ,
        #                  # 'pre_registered10folds6',
        #                  # 'pre_registered10folds7', 
        #                   'pre_registered10folds8',
        #                   'pre_registered10folds9'  
 # 
                               # ],

        'data_set_name' :   [  'post_registered10folds0'],

        'flip_data': True,
        'flip_controls':False,
        'shift_data' : 10,
        'rotate_data': 0,
        'device_id':3,       
        
        'train_more': False,
        'epochs_prev'        :1001,
        'learning_prev'      :0.0000000052,
        'batch_size_tr_prev' :   16 ,
        'step_size_prev' :200,  
        'gamma_prev' : 0.95, 
        
    }


if __name__ == "__main__": 
    device_id = 1
    print('--------------------')
    device, device_id, envname = get_device(device_id)
    
    if envname=='envtest':
        device = 'cpu'
    input_dim = None
    
    home = os.getenv("HOME")
 
    #PD subjects to use laterality
    PD_Features_file = os.path.join('Laterality', 'PD_Features.csv')
    PD_Features = pd.read_csv(PD_Features_file)
    Pat_Domside = PD_Features[["PATNO", "DOMSIDE"]]    
    Pat_Domside = Pat_Domside[Pat_Domside["DOMSIDE"]==1]["PATNO"].tolist()
    
    #Open MNI reference to get the header
    mni_file_path = 'references/mni_icbm152_t1_tal_nlin_sym_09c_LAS.nhdr'
    mni_data, header = nrrd.read(mni_file_path)
          
    
    #Model
    arch_config   = pre_registered_arch
    
    lr =  arch_config['learnings']
    dropout_p  = arch_config['dropouts']
    epochs = arch_config['epochs']
    batch_size_tr = arch_config['batch_size_tr']
    batch_size_te = arch_config['batch_size_te']
    sel_optimizer = arch_config['sel_optimizer']
    gamma = arch_config['gamma']
    L2_weighting = arch_config['L2_weighting']
    
    step_size = arch_config['step_size']
    net_model = arch_config['net_model']
    data_set_split_names =arch_config['data_set_name']

    loss_function_str = arch_config['loss_function']
    adtional_prefix = arch_config['adtional_prefix']
    
    shift_data  = arch_config['shift_data']
    rotate_data = arch_config['rotate_data']
    flip_data   = arch_config['flip_data']  
    flip_controls = arch_config['flip_controls']  
    num_classes = arch_config['num_classes']
    num_classes = 1 #Binary classification Change name by Num_ Outputs neurons

    min_acc_to_save = arch_config['min_acc_to_save']
    freeze_lr_at = arch_config['freeze_lr_at']
    save_last_iteration_model = arch_config['save_last_iteration_model']
    
    #Parameter if continue training or loading from past models
    train_more         = arch_config['train_more']
    epochs_prev        = arch_config['epochs_prev']
    learning_prev      = arch_config['learning_prev']
    batch_size_tr_prev = arch_config['batch_size_tr_prev']
    step_size_prev     = arch_config['step_size_prev']
    gamma_prev         = arch_config['gamma_prev']

    SEED = arch_config['SEED']
    override = arch_config['override']
    
    required_shape = arch_config['required_shape']
    
    if envname=='envtest':
        device = 'cpu'    

    models_folder = '/media/alfonso/data/models_Aug/'    

    model = None

    subset = 'train'
    
    k_folds = 10
    #modality = 'post_registered'  
    scratch = ''
    
    modality = 'pre_registered'     
    scratch = 'pre_registered_scratch'
    
    save_as = 'nhdr'
    subject_split_expr = '_re'
    path_orig_dtis = '/media/alfonso/data/dtis_orig/'
    base_folder_name = 'Ensemblings_post'
    samba = '/mnt/nas_ppmi/nas_dir/'
    
    if 'pre_registered' in modality:
        save_as = 'nhdr'
        subject_split_expr = '_MNI_2mm'
        path_orig_dtis = '/media/alfonso/data/dtis_mni_deff/'        
        base_folder_name = 'Ensemblings_pre'
        
    if '_scratch' in scratch:
        base_folder_name = 'Ensemblings_pre_scratch'    
    
    base_folder_name = os.path.join(samba, base_folder_name)
    createDir(base_folder_name, False)
    
    
    models_iterated = {'missing':[], 'processed':[]}

    for data_set_name in data_set_split_names:
        output_data_set_name = os.path.join(base_folder_name, data_set_name)
        createDir(output_data_set_name, False)
        subjs = []
        for k_fold in range(0, k_folds):  

            output_folder_name = os.path.join(output_data_set_name, 'fold_' + str(k_fold))
            createDir(output_folder_name, False)
            prefix     = getHyperParameterString (net_model, adtional_prefix, epochs, 
                                          batch_size_tr, batch_size_te, step_size,
                                          dropout_p, sel_optimizer, lr, data_set_name, 
                                          loss_function_str,  modality, gamma, L2_weighting, k_fold, 
                                          shift_data, rotate_data,flip_data, flip_controls) 
            print ('Parameters of the model: ' + prefix )        
    
            using_dataset = data_set_name
            # if train_more:
            #     using_dataset = new_data_set_name
                
            pack = load_custom_dataset(False, batch_size_tr = batch_size_tr, 
                                        batch_size_te = batch_size_te, 
                                        envname=envname, 
                                        data_set_name=using_dataset, 
                                        k_fold=k_fold,
                                        Pat_Domside=Pat_Domside,flip_data=flip_data,
                                        flip_controls=flip_controls, device=device,
                                        angle_range = rotate_data, shifts_range = shift_data,
                                        required_shape = required_shape, gradient_comp=True)
            
            if pack is None:
                sys.exit('Data loading error---')
            
            input_dim    = pack['dims']
            
            train_loader = pack['trainset_loader']                                                                                                                                                  
            test_loader  = pack['testset_loader']
            
            tr_nbatches  = pack['tr_nbatches']
            te_nbatches  = pack['te_nbatches']
            orig_nbatches = pack['orig_nbatches']
            try:
                model = torch.load(os.path.join(models_folder, prefix+'.mdl') , map_location=device)            
                model.dump_patches=True
                print('-----*--Model Loaded--*------')
                print( os.path.join(models_folder, prefix+'.mdl'))              
                print('Allocated memory (MB)',torch.cuda.memory_allocated()/(1000000))
                summary(model, input_dim, batch_size=-1, device= device)                
            except:
                model = None
                models_iterated['missing'].append(prefix+'.mdl')
                print('Model was not created')
                print(os.path.join(models_folder, prefix+'.mdl'))
                continue
            
            #OPTIMIZER SETUP
            if sel_optimizer == 'Adam':
                optimizer = torch.optim.Adam (model.parameters(), lr=lr)      
            elif sel_optimizer == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)                 
            elif sel_optimizer == 'SGD':
                optimizer = torch.optim.SGD  (model.parameters(), lr=lr)     
            elif sel_optimizer == 'RMSProps':
                optimizer = torch.optim.RMSprop  (model.parameters(), lr=lr) 
                
            #LOSS FUNCTION SETUP
            classweighting = None
            if classweighting:
                classweighting = classweighting.float()
                if device != 'cpu':
                    classweighting = classweighting.cuda()
                
            loss_func = nn.BCELoss()
            if loss_function_str=='NLLLoss':
                loss_func = nn.NLLLoss(weight=classweighting) #Useful for unbalanced data
            elif loss_function_str=='CrossEntropyLoss':
                loss_func = nn.CrossEntropyLoss(weight=classweighting)  
                num_classes = 2
            elif loss_function_str == 'BCELoss':
                loss_func = nn.BCELoss()  
                num_classes = 1
            elif loss_function_str == 'BCELossWithLogits':
                loss_func = nn.BCEWithLogitsLoss()    
                num_classes = 1
            elif loss_function_str == 'MSELoss':
                loss_func = nn.MSELoss()                      
            else:
                print('Missing loss function')
                break               
                    
            model.eval() 
            if subset == 'train':
                test_loader = pack['trainset_loader']   
            
            subjs = ComputeGrads(model, optimizer, test_loader, loss_function_str, loss_func, device,
                  subset, subject_split_expr, output_folder_name, save_as, k_fold, subjs)
            
            models_iterated['processed'].append(prefix+'.mdl')
        
        #Save all subjects
        pd_df = pd.DataFrame.from_dict(subjs)
        pd_df.to_csv(os.path.join(output_data_set_name,   subset + '_csv_probs.csv'), index=False)   
            
        
        
        
