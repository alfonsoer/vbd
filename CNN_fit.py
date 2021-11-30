import torch
import numpy as np
import copy

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics

from utils.plot_print_results import  print_metrics
from utils.grad_flows import compute_avg_max_grad_flow, compute_avg_max_grad_flow_with_layers

from preprocessing.rotate_dti import rotate_dti_tensor

def analyse_grads(model, x_data, y_data, loss_fn, loss_function, optimizer):
    model.cpu()

    x_data = torch.FloatTensor(x_data) 
    y_data = torch.FloatTensor(y_data.reshape(-1,1))     
    eval = evaluate_model(model, loss_fn, loss_function=loss_function)
    
    model.eval()

    amplification_factor = torch.FloatTensor(1).zero_()
    amplification_factor[0] = 100
    grads = {'tg:0':[],
             'tg:1':[],
             }
    for i in range(len(x_data)):
        x_test = x_data[i]
        y_test = y_data[i]    

        target_class = y_test

        optimizer.zero_grad() 
        model.zero_grad()

        x_test.requires_grad = True
        loss, f1, ps, rs, y, yhat_logit, yhat = eval(x_test, y_test)

        yhat.backward(gradient=amplification_factor, retain_graph=True)
        print('tg:', target_class, x_test.grad)

        #Detach the gradients and create a copy
        gradient = copy.deepcopy(x_test.grad.detach().numpy())
        grads['tg:' + str(int(target_class))].append(gradient)

        x_test.grad.zero_()
        optimizer.zero_grad() 
        model.zero_grad()        
    return gradient



def evaluate_model(model, loss_fn, loss_function='', average='macro'):
    
    def eval(x, y):
        yhat = model(x)
        #Computes loss
        if loss_function == 'BCELoss':
            # print(yhat)
            yhat = yhat.squeeze(1)
            yhat = torch.sigmoid(yhat)
            yhat_logit = np.where(yhat.detach().cpu() <0.5, 0, 1) 
            y = y.type(torch.float)
            loss = loss_fn(yhat.type(torch.float), y)
                
        else:
            print('Redefine the loss function')
                        
        #move the labels to cpu        
        y = y.cpu()        

        #Compute other accuracy metrics
        f1 = f1_score(y, yhat_logit, average=average )
        ps = precision_score(y, yhat_logit, average=average  )
        rs = recall_score(y, yhat_logit, average=average ) 
        
        return loss, f1, ps, rs, y, yhat_logit, yhat
    return eval
        

def make_train(model, loss_fn, optimizer, 
               train_loader, loss_function, device, nbatches = 20, min_im_per_class=2, angle_range=0, shifts_range=0):
        
    def train(savelayers=False):
        batch_grad = []
        batch_loss = []
        batch_f1 = []
        batch_ps = []
        batch_rs = []
        
        labels_true = []
        labels_pred = []
        labels_pred_real = []

        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        
        eval = evaluate_model(model, loss_fn, loss_function)
        efective_nBatch = 0
        layers = None
        for batch_num, (imgs, labels) in enumerate (train_loader, 1): 
            
            #Check if batch is balanced
            counts = torch.unique(labels, return_counts=True)
            if counts[0].size()[0]<=1: 
                continue
            NlabMin = torch.min(counts[1]).numpy().item(0) 
            
            if NlabMin<=min_im_per_class : #If batch size = 12, minimum 7-5, if batch size = 6 minimum 4-2
                continue              

            x_batch = imgs[0] #Add [0] when using ppmi dataset 
            y_batch = labels
            #Move data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            #Rotate training data on GPU
            if angle_range > 0 or shifts_range > 0:
                angle   =   angle_range * (1-2*np.random.uniform(size=(1)))[0] *np.pi/180
                xshift  = (shifts_range * (1-2*np.random.uniform(size=(1)))[0]).astype(int) 
                yshift  = (shifts_range * (1-2*np.random.uniform(size=(1)))[0]).astype(int)     
                x_batch = rotate_dti_tensor(x_batch, angle, xshift, yshift)
            
            #Zeroes all gradients
            optimizer.zero_grad()    
            #Evaluate model
            loss, f1, ps, rs, y, yhat_logit, yhat = eval(x_batch, y_batch)
            loss_item = loss.item()
            #Propagate gradients
            loss.backward()
            #Keep gradients to compute an average
            
            if savelayers==True:
                grad_flow, layers = compute_avg_max_grad_flow_with_layers(model.named_parameters())
                savelayers = False
            else:
                grad_flow = compute_avg_max_grad_flow(model.named_parameters())            

                
            #Step optimizer to updates parameters and zeroes gradients
            optimizer.step()        
            optimizer.zero_grad()

            #Store all metrics
            batch_loss.append(loss_item)
            batch_f1.append(f1)
            batch_ps.append(ps)
            batch_rs.append(rs)
            batch_grad.append(grad_flow)
            
            # Also store labels and predictions to compute confusion matrix
            labels_true.extend(y.numpy().tolist())
            labels_pred.extend(yhat_logit.tolist())
            labels_pred_real.extend(yhat.cpu().tolist())
            
            #Count n_batches
            efective_nBatch += 1
            if efective_nBatch > nbatches:
                break            

        losses = np.mean(batch_loss)
        f1s = np.mean(batch_f1)
        pss = np.mean(batch_ps)
        rcs = np.mean(batch_rs)   
        grads = np.mean(batch_grad,axis=0)
        
        cfm_set_tr  = metrics.confusion_matrix(labels_true, labels_pred, labels=None, sample_weight=None)   
        fpr, tpr, thresholds =   metrics.roc_curve(1.0*np.array(labels_true), 1.0*np.array(labels_pred_real))
        auc = metrics.auc(fpr, tpr)        
        return losses, f1s, pss, rcs, cfm_set_tr, [fpr, tpr, thresholds, auc,1.0*np.array(labels_true), 1.0*np.array(labels_pred_real)],  grads, layers
    return train

def make_test(model, loss_fn, optimizer, 
              test_loader, loss_function, device):

    def test():

        batch_loss = []
        batch_f1 = []
        batch_ps = []
        batch_rs = []

        labels_true = []
        labels_pred = []
        labels_pred_real = []

        model.eval()
        optimizer.zero_grad()
        model.zero_grad()
        
        eval = evaluate_model(model, loss_fn, loss_function)
        for batch_num, (imgs, labels) in enumerate (test_loader, 1): 
            x_batch = imgs[0] #Add [0] when using ppmi dataset  
            y_batch = labels
   
            #Move data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            #Evaluate model
            loss, f1, ps, rs, y, yhat_logit, yhat = eval(x_batch, y_batch)
            loss_item = loss.item()

            #Store all metrics
            batch_loss.append(loss_item)
            batch_f1.append(f1)
            batch_ps.append(ps)
            batch_rs.append(rs)

            # Also store labels and predictions to compute confusion matrix
            labels_true.extend(y.numpy().tolist())
            labels_pred.extend(yhat_logit.tolist())            
            labels_pred_real.extend(yhat.cpu().tolist())


        losses = np.mean(batch_loss)
        f1s = np.mean(batch_f1)
        pss = np.mean(batch_ps)
        rcs = np.mean(batch_rs)  

        cfm_set_tr  = metrics.confusion_matrix(labels_true, labels_pred, labels=None, sample_weight=None)   
        fpr, tpr, thresholds =   metrics.roc_curve(1.0*np.array(labels_true), 1.0*np.array(labels_pred_real))
        auc = metrics.auc(fpr, tpr)            
        return losses, f1s, pss, rcs, cfm_set_tr, [fpr, tpr, thresholds, auc, 1.0*np.array(labels_true), 1.0*np.array(labels_pred_real)]
    return test

def model_fit(model, loss_fn, optimizer, device, 
               scheduler=None, 
               n_epochs=10, 
               train_loader=None, 
               test_loader=None, 
               scheduler_on='train',
               nbatches = 20,
               loss_function=None,
               prefix='MODEL_saved', early_saving=True, 
               min_im_per_class=2, 
               min_acc_to_save=0.7,
               freeze_lr_at=100,
               angle_range = 0, 
               shifts_range= 0,
               k_fold=None,
               plots_dir= '',
               json_dir = '',
               rocs_dir = '',
               grads_dir = '',
               ):
    
    metrics_dict = {
               'flows':  [], 
               'train_losses': [], 
               'train_f1':  [],
               'train_ps':  [],
               'train_rc':  [],  
               'tr_roc':  [], 

               
               'test_losses': [],
               'test_f1': [],
               'test_ps': [],
               'test_rc': [],
               'te_roc':  [], 
               
               'learning_rates': [],
               'cfm_set_tr':[],
               'cfm_set_te':[]
                }     
    
    train = make_train(model, loss_fn, optimizer, 
               train_loader, loss_function, device, nbatches = nbatches, min_im_per_class=min_im_per_class,
                       angle_range=angle_range, shifts_range=shifts_range)
                       
    test  = make_test (model, loss_fn, optimizer,
               test_loader, loss_function, device)
    
    last_prec=0
    savelayers = True
    for epoch in range(n_epochs):
        
        lr_start = optimizer.state_dict()['param_groups'][0]['lr']
        
        losses_tr, f1s_tr, pss_tr, rcs_tr, cfm_set_tr, ROC_tr,  grads, layers = train(savelayers)
        losses_te, f1s_te, pss_te, rcs_te, cfm_set_te, ROC_te = test()

        if savelayers == True:
            metrics_dict['flows'].append(layers)
            
        metrics_dict['flows'].append(grads)   
        metrics_dict['train_losses'].append(losses_tr)
        metrics_dict['train_f1'].append(np.mean(f1s_tr))
        metrics_dict['train_ps'].append(np.mean(pss_tr))
        metrics_dict['train_rc'].append(np.mean(rcs_tr))
        metrics_dict['tr_roc'].append(ROC_tr)   
        
 
        metrics_dict['test_losses'].append(losses_te)
        metrics_dict['test_f1'].append(f1s_te)
        metrics_dict['test_ps'].append(pss_te)
        metrics_dict['test_rc'].append(rcs_te)
        metrics_dict['te_roc'].append(ROC_te)   
        
        metrics_dict['cfm_set_tr'].append(cfm_set_tr)
        metrics_dict['cfm_set_te'].append(cfm_set_te)

        metrics_dict['learning_rates'].append(lr_start)            
        
        savelayers = False
        # Schedulet step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if test_loader is not None and scheduler_on!='train':
                    scheduler.step(losses_te)
                else:
                    scheduler.step(losses_tr)

            elif epoch<freeze_lr_at:
                scheduler.step() 

        #Save best model
        if early_saving==True and pss_te >= last_prec and pss_te>=min_acc_to_save :                
            torch.save(model, prefix +'_'+ str(epoch) +'.mdl')
            print('Saved model at epoch: ', epoch)
            last_prec = pss_te   
        #Save the model every 100 epochs
        if epoch%100==0 and epoch>0:
            torch.save(model, prefix +'_'+ str(epoch) +'.mdl')
            print('Regular saving model at epoch: ', epoch)

    return metrics_dict


