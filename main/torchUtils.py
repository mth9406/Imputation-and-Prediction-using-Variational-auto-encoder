import torch
import numpy as np
import os

class EarlyStopping:
    """
    Applies early stopping condition... 
    """
    def __init__(self, 
                patience: int= 10,
                verbose: bool= False,
                delta: float= 0,
                path= './',
                model_name= 'latest_checkpoint.pth.tar'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_name = model_name
        self.path = os.path.join(path, model_name)
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model, epoch, optimizer):
        ckpt_dict = {
            'epoch':epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            print(f'Early stopping counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, ckpt_dict)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, ckpt_dict):
        if self.verbose:
            print(f'Validation loss decreased: {self.val_loss_min:.4f} --> {val_loss:.4f}. Saving model...')
        torch.save(ckpt_dict, self.path)
        self.val_loss_min = val_loss

def get_loss_imp(x, x_hat, m):
    """
    # Parameters
    x: original value
    m: mask
    x_hat: imputation

    # Returns
    imputation loss:
    m * Loss(x_hat, x)
    """
    a = torch.masked_select(x_hat, m==1.)
    b = torch.masked_select(x, m==1.)
    return torch.mean((a-b)**2)