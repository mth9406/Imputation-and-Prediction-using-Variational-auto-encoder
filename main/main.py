import os
import sys
import argparse

import torch
import torchbnn as bnn
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from torchUtils import * 
from models import *
from train import *
from dataloader import * 

from sklearn.utils.class_weight import compute_class_weight

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--data_type', type= str, default= 'gestures', 
                    help= 'one of: gestures, elec, wind')
parser.add_argument('--data_path', type= str, default= './data/gesture')
parser.add_argument('--tr', type= float, default= 0.7, 
                help= 'the ratio of training data to the original data')
parser.add_argument('--val', type= float, default= 0.2, 
                help= 'the ratio of validation data to the original data')
parser.add_argument('--standardize', action= 'store_true', 
                help= 'standardize the inputs if it is true.')
parser.add_argument('--prob', type= float, default= 0.1, 
                help= 'the ratio of missing data to make in the original data')
parser.add_argument('--test_all_missing', action= 'store_true', 
                help= 'force every observation in the test data to have missing values.')
parser.add_argument('--test_n_missing', type= int, default= 1, 
                help= 'the number of missing values to generate by row. (default= 1)')
parser.add_argument("--cat_features", nargs="+", default= None, 
                help= 'the indices of categorical features (list type, default= None)')

# Training options
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--patience', type=int, default=30, help='patience of early stopping condition')
parser.add_argument('--delta', type= float, default=0., help='significant improvement to update a model')
parser.add_argument('--print_log_option', type= int, default= 10, help= 'print training loss every print_log_option')
parser.add_argument('--imp_loss_penalty', type= float, default= 1.0, 
                    help= 'the penalty term of imputation loss')
parser.add_argument('--kl_weight', type= float, default= 0.1, help= 'kl-loss of vibi (default = 0.1)')
# parser.add_argument('--gradient_max_norm', type= float, default= 5,
#                     help= 'clips gradient norm of an iterable of parameters by \"gradient_max_norm\"')

# model options
parser.add_argument('--model_path', type= str, default= './data/gesture/model',
                    help= 'a path to (save) the model')
parser.add_argument('--input_size', type= int, default= 18, 
                    help= 'the number of time series (dimension)')
parser.add_argument('--n_labels', type= int, default= 5, 
                    help= 'the number of labels (default= 5)')
parser.add_argument('--drop_p', type= float, default= 0.5,
                    help= 'dropout ratio (default= 0.5)')
parser.add_argument('--stack_ae_lyrs', action= 'store_true', 
                    help= 'auto-encoder layer will have 2 linear layers respectively if it is set true.')
parser.add_argument('--stack_fc_lyrs', action= 'store_true',
                    help= 'fc_out layer will have 2 Linear layers if it is set true.')

# To test
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--model_file', type= str, default= 'latest_checkpoint.pth.tar'
                    ,help= 'model file', required= False)
parser.add_argument('--model_type', type= str, default= 'si', 
                    help= 'si: SoftImpute, ai: AutoImpute, vai: VarationalAutoImpute, vibi: VariationalAutoBayesImpute')
parser.add_argument('--vai_n_samples', type= int, default= 100, 
                    help= 'sampling size of VAI model')

parser.add_argument('--num_folds', type= int, default= 1, 
                    help = 'the number of folds')

args = parser.parse_args()
print(args)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# make a path to save a model 
if not os.path.exists(args.model_path):
    print("Making a path to save the model...")
    os.makedirs(args.model_path, exist_ok= True)
else:
    print("The path already exists, skip making the path...")

def main(args):
    # read data
    # one of: gestures, physionet, mimic3
    print("Loading data...")
    if args.data_type == 'gesture':
        # load gestures-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_gestures(args)
        task_type = 'cls'
    elif args.data_type == 'elec':
        # load elec-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_elec(args)
        task_type = 'cls'
    elif args.data_type == 'wind':
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_wind_turbin_power(args)
        task_type = 'regr'
    elif args.data_type == 'mobile': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_mobile(args)
        task_type= 'cls'     
    elif args.data_type == 'wine': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_wine(args)
        task_type= 'cls'  
    elif args.data_type == 'appliances': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_appliances(args)
        task_type= 'regr'
    elif args.data_type == 'pulsar': 
        # load pulsar data 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_pulsar(args)
        task_type = 'cls'     
    elif args.data_type == 'faults': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_faults(args)        
        task_type = 'cls'         
    elif args.data_type == 'abalone': 
        # load abalone data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_abalone(args)              
        task_type = 'regr'
    elif args.data_type == 'spam': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_spam(args)        
        task_type = 'cls'    
    elif args.data_type == 'breast': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_breast(args)        
        task_type = 'cls'       
    elif args.data_type == 'letter': 
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_letter(args)        
        task_type = 'cls'  
    elif args.data_type == 'eeg':
        # load faults data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_eeg(args)        
        task_type = 'cls'         
    elif args.data_type == 'recipes': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_recipes(args)
        task_type= 'regr'
    elif args.data_type == 'stroke': 
        # load wind-data
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_stroke(args)        
        task_type= 'cls'
    elif args.data_type == 'simul': 
        X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde\
            = load_simul(args)        
        task_type= 'cls'        
    else: 
        print("Unkown data type, data type should be one of the followings...")
        print("gesture, elec, wind, mobile, wine, appliances, pulsar, faults, abalone, spam, letter")
        sys.exit()

    # find masks
    M_train, M_valid, M_test = make_mask(X_train_tilde), make_mask(X_valid_tilde), make_mask(X_test_tilde)
    # define training, validation, test datasets and their dataloaders respectively 
    train_data, valid_data, test_data \
        = TableDataset(X_train_tilde, M_train, y_train, X_comp= X_train),\
            TableDataset(X_valid_tilde, M_valid, y_valid, X_comp= X_valid),\
            TableDataset(X_test_tilde, M_test, y_test, X_comp= X_test)
    train_loader, valid_loader, test_loader \
        = DataLoader(train_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(valid_data, batch_size = args.batch_size, shuffle = True),\
            DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    print("Loading data done!")

    # model
    if args.model_type == 'si':
        model = NaiveSoftImpute(args.input_size, args.n_labels, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs).to(device)
        args.cat_features = None
    elif args.model_type == 'ai':
        model = AutoImpute(args.input_size, args.n_labels, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs, stack_ae_lyrs= args.stack_ae_lyrs).to(device)
    elif args.model_type == 'vai': 
        model= VariationalAutoImpute(args.input_size, args.n_labels, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs, stack_ae_lyrs= args.stack_ae_lyrs).to(device)
    elif args.model_type == 'vibi': 
        model = VariationalAutoBayesImpute(args.input_size, args.n_labels, args.drop_p, stack_fc_lyrs=args.stack_fc_lyrs).to(device)
    else:
        print("The model is yet to be implemented.")
        sys.exit()
    
    # setting training args...
    if task_type == 'cls':
        w = compute_class_weight(class_weight='balanced', classes= np.arange(args.n_labels), y= y_train.numpy())
        w = torch.FloatTensor(w).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else: 
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.lr)    
    early_stopping = EarlyStopping(
        patience= args.patience,
        verbose= True,
        delta = args.delta,
        path= args.model_path
    ) 
       
    if args.test: 
        model_file = os.path.join(args.model_path, args.model_file)
        ckpt = torch.load(model_file)
        model.load_state_dict(ckpt['state_dict'])
    else: 
        train(args, model, train_loader, valid_loader, optimizer, criterion, early_stopping, device)
    
    print("==============================================")
    print("Testing the model...")
    if task_type == 'cls': 
        perf = test_cls(args, model, test_loader, criterion, device)
    else: 
        perf = test_regr(args, model, test_loader, criterion, device)
    return perf 

if __name__ =='__main__':

    if args.num_folds == 1:
        main(args)
    else: 
        perf = main(args)
        perfs = dict().fromkeys(perf, None)
        for k in perfs.keys():
            perfs[k] = [perf[k]]

        for i in range(1, args.num_folds): 
            perf = main(args)
            for k in perfs.keys():
                perfs[k].append(perf[k])
        
        for k, v in perfs.items():
            perfs[k] = [np.mean(perfs[k]), np.std(perfs[k])]
        
        print("==============================================")
        print(f"Model type: {args.model_type}")
        print(f"Data: {args.data_type}")
        print(f"Column size: {args.input_size}")
        print(f"Number of classes: {args.n_labels}") 
        if args.test_all_missing:
            print(f"The number of missing values per row: {args.test_n_missing}")
        else:
            print(f"Missing rate: {args.prob}")
        print("==============================================")

        for k, v in perfs.items(): 
            print(f"{k}: mean= {v[0]:.3f}, std= {v[1]:.3f}")