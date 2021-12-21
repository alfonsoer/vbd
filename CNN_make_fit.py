import os
import time
import torch
import sys
import argparse
import configparser
from sys import exit

import torch.nn as nn
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

from cuda import get_device
from utils.ioRes import   data_to_npy
from utils.dti_data import createDir
from utils.githubrepo import upload_plots, upload_json_metrics
from utils.plot_print_results import plot_training_results, plot_grad_flow, plot_last_ROCs

from CNN_config import CNN_settings, getHyperParameterString
from CNN_datasetsPPMI import load_custom_dataset



from CNN_fit import model_fit
import pandas as pd

import SimpleClassificationCNN

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Train CNN using DTI data.')
    parser.add_argument('--config_file', type=str, default="cnn_configs/training_config.config", help='config file for training')
    config = configparser.ConfigParser()
    args = parser.parse_args()
    config.read(args.config_file)
    
    modality = config['CONFIG']['modality']
    k_folds = int(config['CONFIG']['k_folds'])
    arch_config = CNN_settings(modality)
    if arch_config is None:
        print('Undefined architechture')
        exit()
        
    
    input_dim = None
    
    home = os.getenv("HOME")
    
    device_id = arch_config['device_id']
    
    print('--------------------')
    device, device_id, envname = get_device(device_id)    

    #Dirs for plotting and saving metrics 
    plots_dir = 'github/aexp/plots'
    json_dir = 'github/aexp/json'                       
    if envname=='lambda':
        plots_dir = 'experiments/aexp/plots'
        json_dir = 'experiments/aexp/json'
        

    plots_dir = os.path.join(home, plots_dir)
    json_dir = os.path.join(home, json_dir)    
    grads_dir = os.path.join(home, plots_dir, 'grads')
    rocs_dir = os.path.join(home, plots_dir, 'rocs')

    #Create dirs if doesn't exist
    createDir(plots_dir, False)
    createDir(rocs_dir, False)
    createDir(grads_dir, False)
    createDir(json_dir, False)
    
    learnings =  arch_config['learnings']
    dropouts  = arch_config['dropouts']
    orig_epochs = arch_config['epochs']
    batch_size_tr = arch_config['batch_size_tr']
    batch_size_te = arch_config['batch_size_te']
    sel_optimizer = arch_config['sel_optimizer']
    gamma = arch_config['gamma']
    L2_weighting = arch_config['L2_weighting']
    
    step_size = arch_config['step_size']
    net_model_s = arch_config['net_model']
    data_set_names =arch_config['data_set_name']

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
    
    # if envname=='envtest':
    #     device = 'cpu'    
    
    model = None
    
    PD_Features_file = os.path.join('Laterality', 'PD_Features.csv')
    PD_Features = pd.read_csv(PD_Features_file)
    Pat_Domside = PD_Features[["PATNO", "DOMSIDE"]]    
    Pat_Domside = Pat_Domside[Pat_Domside["DOMSIDE"]==1]["PATNO"].tolist()
    #Select all Pad with DOMSIDE = 1 
    
    
    
    
    for data_set_name in data_set_names:
        if  'ppmi_orig_npy' in data_set_name:
            k_folds = 1
        # elif 'folds' in data_set_name:
        #     k_folds = 4
            
        for k_fold in range(0, k_folds):    
            # if k_fold<=4:
            #     print('***Continue trainings')
            #     continue
            if k_folds==1:
                k_fold=None    
            #Point to fold given by the database
            #PENDING TO CHANGE FOLD


            for net_model in net_model_s:
                print("test models")
                for lr in learnings:
                    print("test lr")
                    epochs = orig_epochs
                    for dropout_p in dropouts:
                        t = time.time()
                        
                        print('Starting training with : ', device, device_id)
                        print('--------------------')   
                        
                        if train_more:
                            # prefix_pre = getHyperParameterString (net_model, adtional_prefix, epochs, 
                            #                           batch_size_tr, batch_size_te, step_size,
                            #                           dropout_p, sel_optimizer, lr, data_set_name, 
                            #                           loss_function_str,  modality, gamma, L2_weighting, k_fold, 
                            #                           shift_data, rotate_data,flip_data, flip_controls)            
                            
                            prefix_pre = getHyperParameterString (net_model, adtional_prefix, epochs_prev, 
                                                      batch_size_tr_prev, batch_size_te, step_size_prev,
                                                      dropout_p, sel_optimizer, learning_prev, data_set_name, 
                                                      loss_function_str, modality, gamma_prev, L2_weighting, k_fold, 
                                                      shift_data, rotate_data,flip_data, flip_controls) 

                            
                        prefix     = getHyperParameterString (net_model, adtional_prefix, epochs, 
                                                      batch_size_tr, batch_size_te, step_size,
                                                      dropout_p, sel_optimizer, lr, data_set_name, 
                                                      loss_function_str,  modality, gamma, L2_weighting, k_fold, 
                                                      shift_data, rotate_data,flip_data, flip_controls)    
                        
                        print ('Parameters of the model: ' + prefix )
                        #Check if this model has been trained before
                        if os.path.exists(prefix) and override is False:
                            print(':( Model already trained. Change configuration parameters or check override flag to True')
                            continue    

                        torch.manual_seed(SEED)
                        # np.random.seed(SEED)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(SEED) 
                        
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
                                                   required_shape = required_shape)
                        
                        if pack is None:
                            sys.exit('Data loading error---')
                        
                        input_dim = pack['dims']
                        train_loader = pack['trainset_loader']                                                                                                                                                  
                        test_loader  = pack['testset_loader']
                        tr_nbatches = pack['tr_nbatches']
                        te_nbatches = pack['te_nbatches']
                        orig_nbatches = pack['orig_nbatches']
                        classweighting = None
                        
                        
                        #SETUP LOSS FUNCTION
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

                        if train_more == True:
                            model = torch.load( prefix_pre + '.mdl')
                            print('LOADED old model: ',prefix_pre)
                        else:                            
                            #model = model_setup(net_model, input_dim, num_classes, dropout_p)  
                            # default_conv_layer_widths = [[(8,1),(16,1),(32,3)],[(64,1),(64,3)],[(72,3),],[(72,3),],[(64,3),]]
                            #default_conv_layer_widths = [[(16,1),],[(32,1),(64,3)],[(128,3),],[(256,3),],[(512,3),]]
                            config1 = [[(16,1),(32,1),(36,3)],[(64,1),(72,3)],[(256,3),],[(256,3),(128,1)], [(128,3)]]
                            config2 = [[(16,1),],[(32,1),(64,3)],[(128,3),],[(256,3),],[(512,3),],[(1024,3)]]
                            config3 = [[(16,1),],[(32,1),(64,1)],[(64,3),],[(128,3),],[(256,3),],[(512,3)]]
                            config4 = [[(32,3),],[(32,3),(64,3)],[(64,3),],[(128,3),], [(128,3)], [(256,3)]]
                            config5 = [[(36,3)],[(72,3)], [(144,3)],[(288,3)],[(576,3),],] 
                            config6 = [[(36,3)],[(72,3)], [(144,3)],[(288,3)],[(576,3),],[(576,1)]] 
                            config7 = [[(36,3)],[(36,1),(72,3)], [(72,1),(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config8 = [[(36,1)],[(72,1)], [(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config8b = [[(48,1)],[(96,1)], [(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config8c = [[(18,1)],[(36,1)], [(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config8d = [[(54,1)],[(96,1)], [(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config8f = [[(48,1)],[(96,1)], [(48,3)],[(48,1),(96,3)],[(96,1),(128,3),],[(128,1)]] 
                            config8g = [[(48,1)],[(96,1)], [(64,3),(64,3)],[(64,1),(128,3),(128,3)],[(128,1),(256,3),(256,3)],[(256,1)]] 
                        
                            
                            config9 = [[(12,5), (18,1)],[(72,3)], [(72,1),(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 
                            config10 = [[(12,3), (18,1)],[(72,3)], [(72,1),(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(576,1)]] 

                            config11 = [[(8,3),(16,3),(32,3)],[(16,1),(32,3)], [(16,1),(32,3)],[(16,1),(32,3)],[(16,1),(32,3),],[(16,1)]] #Similar to previous config
                            #default_conv_layer_widths = [[(32,3),(16,1)],[(64,3),(32,1)],[(128,3),(64,1)],[(256,3),(128,1)],[(512,3),(256,1)],[(512,3)]]
                            config8h = [[(48,1)],[(96,1)], [(72,3)],[(72,1),(144,3)],[(144,1),(288,3),],[(288,1)]] 
                            

                            config12 = [[(36,3)],[(72,3)], [(144,3)],[(288,3)],[(576,3),],[(576,1)]] 
                            # default_conv_layer_widths = [[(8,1),(16,1),(32,3)],[(64,1),(64,3)],[(72,3),],[(72,3),],[(72,3),]]


                            config16Aug = [[(48,1)],[(24,3),],[(32,3),(32,3)],[(64,3),(64,3)], [(128,3),(128,3)], [(256,3),(256,3)]]
                            config17Aug = [[(6,3), (12,3),(6,1)],[(12,3), (24,3), (12,1)], [(24,3), (48,3), (24,1)], [(24,3), (48,3), (24,1)],  [(48,3), (96,3), (48,1)]] #Similar to previous config

                            config7B =  [[(36,3)],[(36,1),(72,3)], [(72,1),(144,3)],[(144,1),(288,3)],[(288,1),(576,3),],[(256,1)]]   
                            conf7C = [[(36,3),(18,1)],[(72,3),(36,1)], [(144,3),(72,1)],[(288,3),(144,1)],[(576,3),(288,1)],[(288,3)]] 
                       
                            conf7E = [[(16,3),(16,3)],[(32,3),(32,1)], [(64,3),(64,1)],[(128,3),(128,1)],[(256,3),(256,1)],[(512,3)]] 
                            
                            confMin = [[(36,3),(6,1)], [(72,3),(6,1)], [(144,3),(6,1)], [(256,3), (6,1)], [(512,3),(6,1)]] 
                            confDeep = [[(6,3),(16,3),(32,3)], [(32,3),(64,3),(128,3)], [(128,3),(256,3)], [(256,3)], [(256,3)]] 

                            #default_linear_layer_widths = [32,32, 1]          
                            
                            # default_conv_layer_widths = [[(8,1),(16,1),(32,3)],[(64,1),(64,3)],[(72,3),],[(72,3),],[(72,3),]]
                                                        
                            config8e = [[(48,1)],[(96,1)], [(64,3)],[(64,1),(128,3)],[(128,1),(256,3),],[(256,1)]] 
                            default_linear_layer_widths = [64,128, 1]          
                            
                            
                            # conf9LayU = [[(48,1),(36,3)], [(36,3),(64,3),], [(128,3)],  [(256,3),(512,3),(256,1),(256,3)], [(128,1),(64,3),(32,1)], [(32,3)]] 
                            # default_linear_layer_widths = [512, 1]          
                            
                            config_debug = [[(6,3),(6,1),(6,1)],[(12,3)],[(24,3),],[(48,3),], [(96,3)], [(192,3)]]
                            # config_debug = [[(6,3),(32,1)],[(32,3),(64,1)], [(128,3),(256,1)], [(256,3),(128,1)], [(128,3)], [(128,1)]]
                            default_conv_layer_widths = config_debug
                            
                            dims = torch.Size(required_shape)
                            model = SimpleClassificationCNN.SimpleClassificationCNN(dims, 6, 
                                                                                    default_conv_layer_widths, 
                                                                                    default_linear_layer_widths, 
                                                                                    use_bn=True, prob=dropout_p)
                            
                            print('CREATED new model: ',prefix)
                        
                        if model is None:
                            print('Model was not loaded/created')
                            break
                        
                        #Send to CUDA
                        if device != 'cpu':
                            print('Moving model to GPU')
                            torch.cuda.empty_cache()
                            model.to(device) #Move the model to the available device
                            
                        print('Allocated memory (MB)',torch.cuda.memory_allocated()/(1000000))
                        summary(model, input_dim, batch_size=-1, device= device)
                        
                             
                        #SETUP OPTIMIZER and SCHEDULER
                        if sel_optimizer == 'Adam':
                            optimizer = torch.optim.Adam (filter(lambda p: p.requires_grad, model.parameters()), lr=lr)      
                        elif sel_optimizer == 'AdamW':
                            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                                          weight_decay=L2_weighting,amsgrad=False )                 
                        elif sel_optimizer == 'SGD':
                            optimizer = torch.optim.SGD  (filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=L2_weighting)     
                        elif sel_optimizer == 'RMSProps':
                            optimizer = torch.optim.RMSprop  (filter(lambda p: p.requires_grad, model.parameters()), lr=lr)                      
                        
                        scheduler = None
                        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)            
                        # scheduler = ReduceLROnPlateau(optimizer, 'min',  verbose=True) 
        
                        #If batch size = 12, minimum 7-5, if batch size = 6 minimum 4-2
                        min_im_per_class = int(batch_size_tr/2) - 2   
                        
                        print('Starting training loop:', prefix)
        
                        #MODEL FIT
                        metrics_dict  = model_fit(model, loss_func, optimizer, device,
                            scheduler=scheduler, 
                            n_epochs=epochs, 
                            train_loader=train_loader, 
                            test_loader=test_loader, 
                            scheduler_on='train',
                            nbatches = orig_nbatches,
                            loss_function=loss_function_str,
                            prefix=prefix, early_saving=True, min_im_per_class=min_im_per_class, 
                            min_acc_to_save=min_acc_to_save,
                            freeze_lr_at= freeze_lr_at, angle_range=rotate_data, shifts_range=shift_data, k_fold=k_fold,
                            plots_dir= plots_dir,
                            json_dir = json_dir,
                            rocs_dir = rocs_dir,
                            grads_dir = grads_dir)
                        
                        #Save best model
                        if save_last_iteration_model==True  :                
                            torch.save(model, prefix +'.mdl')
                            print('Saved model at last epoch')
                        
                        data_to_npy(metrics_dict, json_dir, prefix=prefix)
                        upload_json_metrics(prefix)
                        
                        plot_training_results(metrics_dict, plots_dir=plots_dir, prefix=prefix)
                        plot_grad_flow(metrics_dict['flows'], plots_dir=grads_dir, prefix=prefix,  epoch=epochs)     
                        plot_last_ROCs(metrics_dict, plots_dir=rocs_dir, prefix=prefix,  epoch=epochs)     
                        
                        upload_plots(prefix)      
                        
                        if device != 'cpu':
                            torch.cuda.empty_cache()            
                        
                        print('Moving models to NAS')
                        print('Finished after: ', str((time.time()-t)/3600), ' hours')                         
                        #If using several dropouts, it adds an retardant effect.
                        epochs += 20 
