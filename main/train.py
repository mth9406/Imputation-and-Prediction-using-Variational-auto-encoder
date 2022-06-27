import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F
import numpy as np
import os
import csv

from torchUtils import get_loss_imp
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

from utils import *

def train(args, 
          model, 
          train_loader, valid_loader, 
          optimizer, criterion, early_stopping,
          device):
    logs = {
        'tr_loss':[],
        'valid_loss':[]
    }

    kl_loss = None
    if args.model_type == 'vibi': 
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)

    num_batches = len(train_loader)
    print('Start training...')
    for epoch in range(args.epoch):
        # to store losses per epoch
        tr_loss, valid_loss = 0, 0
        # a training loop
        for batch_idx, x in enumerate(train_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)

            model.train()
            # feed forward
            with torch.set_grad_enabled(True):
                out = model(x)
                loss_imp = get_loss_imp(x['input'], out['imputation'], x['mask'])
                loss = criterion(out['preds'], x['label'])
                loss += args.imp_loss_penalty * loss_imp
                if out['regularization_loss'] is not None: 
                    loss += args.imp_loss_penalty * out['regularization_loss']
                if kl_loss is not None: 
                    kl = kl_loss(model)[0]
                    # print(kl)
                    # print(loss)
                    loss += args.kl_weight * kl
            
            # backward 
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), args.gradient_max_norm)
            optimizer.step()

            # store the d_tr_loss
            tr_loss += loss.detach().cpu().item()

            if (batch_idx+1) % args.print_log_option == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Batch [{batch_idx+1}/{num_batches}]: \
                    loss = {loss.detach().cpu().item()}')

        # a validation loop 
        for batch_idx, x in enumerate(valid_loader):
            x['input'], x['mask'], x['label'] \
                = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
            
            # model.eval()
            loss = 0
            with torch.no_grad():
                out = model(x)
                loss_imp = get_loss_imp(x['input'], out['imputation'], x['mask'])
                loss = criterion(out['preds'], x['label'])
                loss += args.imp_loss_penalty * loss_imp
                if out['regularization_loss'] is not None: 
                    loss += args.imp_loss_penalty * out['regularization_loss']
                if kl_loss is not None: 
                    kl = kl_loss(model)[0]
                    loss += args.kl_weight * kl
            valid_loss += loss.detach().cpu().item()
        
        # save current loss values
        tr_loss, valid_loss = tr_loss/len(train_loader), valid_loss/len(valid_loader)
        logs['tr_loss'].append(tr_loss)
        logs['valid_loss'].append(valid_loss)

        print(f'Epoch [{epoch+1}/{args.epoch}]: training loss= {tr_loss:.6f}, validation loss= {valid_loss:.6f}')
        early_stopping(valid_loss, model, epoch, optimizer)

        if early_stopping.early_stop:
            break     

    print("Training done! Saving logs...")
    log_path= os.path.join(args.model_path, 'training_logs')
    os.makedirs(log_path, exist_ok= True)
    log_file= os.path.join(log_path, 'training_logs.csv')
    with open(log_file, 'w', newline= '') as f:
        wr = csv.writer(f)
        n = len(logs['tr_loss'])
        rows = np.array(list(logs.values())).T
        wr.writerow(list(logs.keys()))
        for i in range(1, n):
            wr.writerow(rows[i, :])

def test_cls(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    te_pred_loss = 0
    te_imp_loss = 0
    te_imp_pred_loss = 0
    te_tot_loss = 0
    
    labels = np.array([np.arange(args.n_labels)])    
    cm = np.zeros((args.n_labels, args.n_labels))

    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label']\
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        x['complete_input'] = x['complete_input'].to(device) if x['complete_input'] is not None\
            else None 

        model.eval()
        loss = 0
        with torch.no_grad():
            out = model(x, numobs= args.vai_n_samples)
            loss_imp = get_loss_imp(x['input'], out['imputation'], x['mask'])
            preds = torch.argmax(F.softmax(out['preds'], dim=1), dim=1)
            loss = criterion(out['preds'], x['label'])
            loss_reg = 0.
            imp_pred_loss = 0.
            if out['regularization_loss'] is not None: 
                loss_reg += args.imp_loss_penalty * out['regularization_loss']
            tot_loss = loss + args.imp_loss_penalty * loss_imp + loss_reg
            if x['complete_input'] is not None: 
                imp_pred_loss += get_loss_imp(x['complete_input'], out['imputation'], 1-x['mask'])
        # loss
        te_tot_loss += tot_loss.detach().cpu().item()
        te_imp_loss += loss_imp.detach().cpu().item()
        te_pred_loss += loss.detach().cpu().item()
        te_imp_pred_loss += imp_pred_loss.detach().cpu().item()

        # confusion matrix
        preds = preds.detach().cpu().numpy()
        cm += confusion_matrix(x['label'].detach().cpu().numpy(), preds, labels= labels)
    
    acc, rec, prec, f1 = evaluate(cm, weighted= False) 
    te_tot_loss = te_tot_loss/len(test_loader)
    te_imp_loss = te_imp_loss/len(test_loader)
    te_pred_loss = te_pred_loss/len(test_loader)
    te_imp_pred_loss = te_imp_pred_loss/len(test_loader)

    print("Test done!")
    print(f"test total loss: {te_tot_loss:.2f}")
    print(f"test imputation loss: {te_imp_loss:.2f}")
    print(f"test prediction loss: {te_pred_loss:.2f}")
    print(f"test imputation prediction loss {te_imp_pred_loss:.2f}")
    print() 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    cm_file = os.path.join(args.model_path, f"confusion_matrix.png")
    plt.savefig(cm_file)
    plt.show()
    print(f"정확도 (accuracy): {acc:.2f}")
    print(f"재현율 (recall): {rec:.2f}")
    print(f"정밀도 (precision): {prec:.2f}")
    print(f"F1 score: {f1:.2f}")
    print()   

    perf = {
        'acc': acc,
        'rec': rec,
        'prec': prec,
        'f1': f1, 
        'imp_error':te_imp_pred_loss 
    }

    return perf 

def test_regr(args, 
          model, 
          test_loader, 
          criterion, 
          device
          ):
    
    te_loss_imp = 0
    te_loss_preds = 0
    te_loss_tot = 0
    te_imp_pred_loss = 0.
    te_r2 = 0
    te_mae = 0
    te_mse = 0
    
    for batch_idx, x in enumerate(test_loader):
        x['input'], x['mask'], x['label'] \
            = x['input'].to(device), x['mask'].to(device), x['label'].to(device)
        
        x['complete_input'] = x['complete_input'].to(device) if x['complete_input'] is not None\
            else None 
        
        model.eval()
        loss = 0
        with torch.no_grad():
            out = model(x)
            loss_imp = get_loss_imp(x['input'], out['imputation'], x['mask'])
            loss = criterion(out['preds'], x['label'])
            loss_reg = 0. 
            imp_pred_loss = 0.
            if out['regularization_loss'] is not None: 
                loss_reg += args.imp_loss_penalty * out['regularization_loss']
            tot_loss = loss + args.imp_loss_penalty * loss_imp + loss_reg
            if x['complete_input'] is not None: 
                imp_pred_loss += get_loss_imp(x['complete_input'], out['imputation'], 1-x['mask'])        
        te_loss_imp += loss_imp.detach().cpu().numpy()
        te_loss_preds += loss.detach().cpu().numpy()
        te_loss_tot += tot_loss.detach().cpu().numpy()
        te_imp_pred_loss += imp_pred_loss.detach().cpu().item()

        te_r2 += r2_score(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy())
        te_mae += mean_absolute_error(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy()) 
        te_mse += mean_squared_error(out['preds'].detach().cpu().numpy(), x['label'].detach().cpu().numpy()) 

    te_loss_imp = te_loss_imp/len(test_loader)
    te_loss_preds = te_loss_preds/len(test_loader)
    te_loss_tot = te_loss_tot/len(test_loader)
    te_imp_pred_loss = te_imp_pred_loss/len(test_loader)
    te_r2 = te_r2/len(test_loader)
    te_mae = te_mae/len(test_loader)
    te_mse = te_mse/len(test_loader)
    print("Test done!")
    print(f"imputation loss: {te_loss_imp:.2f}")
    print(f"prediction loss: {te_loss_preds:.2f}")
    print(f"total loss: {te_loss_tot:.2f}")
    print(f"test imputation prediction loss {te_imp_pred_loss:.2f}")
    print(f"r2: {te_r2:.2f}")
    print(f"mae: {te_mae:.2f}")
    print(f"mse: {te_mse:.2f}")
    print()    

    perf = {
        'r2': te_r2,
        'mae': te_mae,
        'mse': te_mse,
        'imp_error':te_imp_pred_loss 
    }

    return perf 
# def predict(args, model, x, num_obs):
#     preds, imputations = [], []
#     device= x['input'].device
#     for _ in range(num_obs):
#         out = model(x)
#         preds.append(out['preds'])
#         imputations.append(out['imputation'])
#     preds, imputations = torch.stack(preds, dim= 0).to(device), torch.stack(imputations, dim= 0).to(device)
#     pred = torch.mean(preds, dim= 0)
#     imp_mean = torch.mean(imputations, dim= 0)
#     imp_std = torch.std(imputations, dim= 0)
#     return pred, imp_mean, imp_std
    
