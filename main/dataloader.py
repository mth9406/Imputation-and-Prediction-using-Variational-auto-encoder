import torch
from torch.utils.data import Dataset
from utils import *
import numpy as np
import pandas as pd 
import os

from imblearn.under_sampling import RandomUnderSampler 

class TableDataset(Dataset):
    """
    Table dataset

    # Parameters
    X: input tableau data with missing values (float tensor type)
    M: mask (float tensor type)
    y: independent variable (target variable: long or float type)
    X_comp: complete matrix of X (true label of input)
    """ 
    def __init__(self, X, M, y, X_comp= None):
        super().__init__()
        self.X, self.y = X, y
        self.M = M
        self.X_comp= X_comp
        
    def __getitem__(self, index):
        if self.X_comp is None:
            return {"input":self.X[index], 
                    "mask":self.M[index],
                    "label": self.y[index],
                    "complete_input": None
                    }
        else: 
            return {"input":self.X[index], 
                    "mask":self.M[index],
                    "label": self.y[index],
                    "complete_input": self.X_comp[index]
                    }            

    def __len__(self):
        return len(self.X)

def make_mask(x_batch):
    """
    A fucntion to make a mask matrix

    # Parameter
    x_batch: input data (float torch type)

    # Returns
    mask: mask matrix which indicates the indices of not missing values (float torch type)
    """
    mask = ~torch.isnan(x_batch) * 1.0
    return mask

def train_valid_test_split(args, X, y, task_type= "cls"):
    """
    A fuction to train-validation-test split

    # Parameter
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.
    X: independent variables
    y: dependent variables
    task_type: regression if "regr", classification if "cls"

    # Return
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde (torch tensor type)

    """
    tr, val = args.tr, args.val
    # split the data into training, validation and test data
    n, p = X.shape
    si_val = int(n*tr)  # starting index of each validation and test data
    si_te = si_val + int(n*val)
    idx = np.random.permutation(np.arange(n)) # shuffle the index

    X_train, y_train = X.values[idx[:si_val], :], y.values[idx[:si_val], ]
    X_valid, y_valid = X.values[idx[si_val:si_te],:], y.values[idx[si_val:si_te], ]
    X_test, y_test = X.values[idx[si_te:], :], y.values[idx[si_te:], ]

    if args.standardize: 
        if args.cat_features is None:
            X_train, cache = min_max_scaler(X_train)
            X_valid, X_test = min_max_scaler_test(X_valid, cache), min_max_scaler_test(X_test, cache)
        else: 
            tot_features = list(range(p))
            num_features = list(set(tot_features)-set(args.cat_features))
            X_train[:, num_features], cache = min_max_scaler(X_train[:, num_features])
            X_valid[:, num_features], X_test[:, num_features]\
                 = min_max_scaler_test(X_valid[:, num_features], cache), min_max_scaler_test(X_test[:, num_features], cache)
        if task_type == 'regr': 
            y_train, cache = min_max_scaler(y_train) 
            y_test = min_max_scaler_test(y_test, cache)
            y_valid = min_max_scaler_test(y_valid, cache)

    X_train, X_valid, X_test\
        = torch.FloatTensor(X_train), torch.FloatTensor(X_valid), torch.FloatTensor(X_test)
    
    X_train_tilde, X_valid_tilde, X_test_tilde = None, None, None
    
    if args.prob > 0.:
        X_train_tilde, _ = make_missing(X_train, args.prob)
        X_valid_tilde, _ = make_missing(X_valid, args.prob)
        if args.test_all_missing:
            X_test_tilde, _ = make_missing_by_row(X_test, args.test_n_missing)
        else:
            X_test_tilde, _ = make_missing(X_test, args.prob)

    if task_type == 'cls':
        y_train, y_valid, y_test\
            = torch.LongTensor(y_train), torch.LongTensor(y_valid), torch.LongTensor(y_test)
    else: 
        y_train, y_valid, y_test\
            = torch.FloatTensor(y_train), torch.FloatTensor(y_valid), torch.FloatTensor(y_test)
    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_gestures(args):
    """
    A function to load gestures-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.LongTensor for "y")    
    """
    path = os.listdir(args.data_path)
    gesture_files = []
    for g in path:
        if '_raw' in g:
            gesture_files.append(os.path.join(args.data_path, g))

    # to convert y to a numeric variable
    mapping = {'Rest': 0, 'Preparation': 1, 'Stroke': 2, 'Hold': 3, 'Retraction': 4}

    data = pd.DataFrame([])
    for i, gesture in enumerate(gesture_files):
        g = pd.read_csv(gesture)
        g = g.dropna(axis= 0)
        g = g.drop(['timestamp'], axis= 1)
        # Prepara????o --> Preparation (anomaly correction)
        anom_idx = g.iloc[:,-1] == 'Prepara????o'
        if sum(anom_idx) >= 1:
            g.loc[anom_idx, 'phase'] = 'Preparation'
        # convert y to a numeric variable
        g.iloc[:, -1] = g.iloc[:, -1].map(mapping).astype('int64')
        data = pd.concat([data, g], axis= 0)
    print(data.info())
    print('-'*20)    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_elec(args):
    """
    A function to load elec-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.LongTensor for "y")    
    """
    f = os.path.join(args.data_path, 'elec_data.csv') # file
    # data = pd.read_csv(f, encoding= 'cp949')
    data = pd.read_csv(f)
    data = data.dropna(axis= 0)
    print(data.info())
    print('-'*20)

    X, y = data.iloc[:, :8], data.iloc[:, -1] # voltage high-frequency average 
    # to convert y to a numeric variable
    mapping = {'??????':0, '??????':1, '??????':2}
    y = y.map(mapping)

    args.n_labels = 3
    args.input_size= 8

    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde 

def load_wind_turbin_power(args):
    """
    A function to load wind-turbin-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    f = os.path.join(args.data_path, 'features.csv') # file
    t = os.path.join(args.data_path, 'power.csv') # target file
    X = pd.read_csv(f)
    y = pd.read_csv(t)
    data = pd.merge(left = X, right= y, on= 'Timestamp' ,how= 'inner')
    data = data.dropna(axis= 0)
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde       

def load_mobile(args):
    """
    A function to load mobile-price-prediction-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'train.csv')
    data = pd.read_csv(data_file)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    cat_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
    X_cat = X[cat_features]
    num_features = list(set(X.columns)-set(cat_features))
    X_num = X[num_features]
    X = pd.concat([X_cat,X_num], axis= 1)

    args.cat_features = list(range(X_cat.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 4

    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_wine(args):
    """
    A function to load wine-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'WineQT.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis= 0)
    X, y = data.iloc[:, :-2], data.iloc[:, -2]
    y.loc[y <= 5] = 0
    y.loc[y==6] = 1
    y.loc[(y == 7)|(y==8)] = 2
    args.input_size = X.shape[1] 
    args.n_labels = 3

    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_appliances(args):
    """
    A function to load appliances-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'KAG_energydata_complete.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    X, y = data.iloc[:, 2:-2], data.iloc[:, 1]
    args.input_size = X.shape[1]
    args.n_labels = 1
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_pulsar(args):
    """
    A function to load pulsar-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    x_file = os.path.join(args.data_path, 'pulsar_x.csv')
    y_file = os.path.join(args.data_path, 'pulsar_y.csv')
    X, y = pd.read_csv(x_file), pd.read_csv(y_file).iloc[:, -1]
    data = pd.concat((X,y), axis= 1)
    data = data.dropna(axis= 0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    args.input_size = X.shape[1]
    args.n_labels = 2
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_faults(args):
    """
    A function to load faults-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'faults.csv')
    targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
    
    data = pd.read_csv(data_file)
    n, p = data.shape
    classes = np.zeros((n,))
    for i, target in enumerate(targets):
        idx = (data[target] == 1)
        classes[idx] = i
    data = data.drop(targets, axis=1)
    data['faults'] = classes

    data = data.dropna(axis= 0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    args.input_size = X.shape[1] 
    args.n_labels = 7

    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_abalone(args):
    """
    A function to load abalone-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'abalone_csv.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    dummies = pd.get_dummies(data.iloc[:, 0], drop_first= True)
    X = pd.concat([dummies, X], axis= 1)

    args.cat_features = list(range(dummies.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_spam(args):
    """
    A function to load spam-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'spambase.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_breast(args):
    """
    A function to load breast-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to breast-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'wpbc.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)
    
    idx = data.iloc[:,-1] == '?'
    data = data.loc[~idx, :]
    data.iloc[:,-1] = data.iloc[:,-1].astype(float)

    cols = np.arange(data.shape[1])
    X, y = data.iloc[:, np.argwhere(cols!=1).flatten()], data.iloc[:, 1]

    y = y.map({'N':0, 'R':1})
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_letter(args):
    """
    A function to load letter-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'letter-recognition.data')
    data = pd.read_csv(data_file, header= None)
    data = data.dropna(axis=0)

    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z']
    mapping = {alp:idx for idx, alp in enumerate(alphabets)}

    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    y = y.map(mapping)
    
    args.input_size = X.shape[1]
    args.n_labels = len(alphabets)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_eeg(args):
    """
    A function to load eeg-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'EEG_Eye_State_Classification.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    args.input_size = X.shape[1]
    args.n_labels = 2
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_recipes(args):
    """
    A function to load recipes-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'epi_r.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 2:-1], data.iloc[:, 1]
    # dummies = pd.get_dummies(data.iloc[:, 0], drop_first= True)
    # X = pd.concat([dummies, X], axis= 1)

    args.cat_features = list(range(4, X.shape[1]))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_stroke(args):
    """
    A function to load stroke-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    data_file = os.path.join(args.data_path, 'healthcare-dataset-stroke-data.csv')
    data = pd.read_csv(data_file)
    data = data.dropna(axis=0)
    
    X, y = data.iloc[:, 1:-1], data.iloc[:, -1]
    X_num = X[['hypertension','heart_disease','age','avg_glucose_level','bmi']]
    col_cat = X.columns.drop(['age','avg_glucose_level','bmi','hypertension','heart_disease'])
    X_cat = X[col_cat]
    X_cat = pd.get_dummies(X_cat, drop_first= True)
    X = pd.concat([X_cat, X_num], axis= 1)

    args.cat_features = list(range(13))
    args.input_size = X.shape[1]
    args.n_labels = 2

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_simul(args):
    """
    A function to load simul-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    target_file = os.path.join(args.data_path, 'target.csv')
    var_file = os.path.join(args.data_path, 'var.csv')
    y = pd.read_csv(target_file).iloc[:, -1]
    X = pd.read_csv(var_file)

    args.cat_features = None
    args.input_size = X.shape[1]
    args.n_labels = 2

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'cls')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde

def load_bench(args):
    """
    A function to load bench-data.
    
    # Parameters
    args contains the followings...
    * data_path: a path to gesture-data
    * tr: the ratio of training data to the original data
    * val: the ratio of validation data to the original data
    remaining is the test data so, tr+val < 1.

    # Returns
    X_train, X_valid, X_test, y_train, y_valid, y_test (torch.FloatTensor for "X", torch.FloatTensor for "y")    
    """
    X_file = os.path.join(args.data_path, 'X_train.csv')
    y_file = os.path.join(args.data_path, 'y_train.csv')
    X = pd.read_csv(X_file).drop(['BestSquatKg'], axis= 1)
    y = pd.read_csv(y_file).iloc[:, 1]
    data = pd.concat([X,y], axis= 1)
    data = data.dropna(axis=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_num = X.iloc[:, 4:]
    X_cat = X.iloc[:, 2:4]
    X_cat = pd.get_dummies(X_cat, drop_first= True)
    X = pd.concat([X_cat, X_num], axis= 1)

    args.cat_features = list(range(4))
    args.input_size = X.shape[1]
    args.n_labels = 1

    data = pd.concat([X,y], axis= 1)
    print(data.info())
    print('-'*20)
    X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde \
        = train_valid_test_split(args, X, y, task_type= 'regr')

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_tilde, X_valid_tilde, X_test_tilde
