import torch 
from torch import nn
import torchbnn as bnn
from torch.nn import functional as F
from torch.autograd import Variable

class EncoderLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (integer)
    out_features: output dimension (integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features),
                nn.Dropout(p=drop_p, inplace=False)
        )

class DecoderLayer(nn.Sequential):
    """
    Decoder layer
    in_features: input dimension (integer)
    out_features: output dimension (integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.ReLU()
        )

class EncoderDoubleLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (integer)
    out_features: output dimension (integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, middle_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=middle_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.BatchNorm1d(middle_features),
                nn.ReLU(),
                nn.Linear(middle_features, out_features),
                nn.Dropout(p= drop_p, inplace= False),
                nn.BatchNorm1d(out_features)
        )

class DecoderDoubleLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (integer)
    out_features: output dimension (integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, middle_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                nn.Linear(in_features=in_features, out_features=middle_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.BatchNorm1d(middle_features),
                nn.ReLU(),
                nn.Linear(middle_features, out_features),
                nn.Dropout(p= drop_p, inplace= False),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
        )

class BayesDecoderDoubleLayer(nn.Sequential):
    """
    Encoder layer
    in_features: input dimension (integer)
    out_features: output dimension (integer)
    drop_p: drop out rate (default: 0.5)
    """
    def __init__(self, in_features:int, middle_features:int, out_features:int, drop_p:float= 0.5):
        super().__init__(
                bnn.BayesLinear(prior_mu= 0, prior_sigma= 0.1, in_features=in_features, out_features= middle_features),
                nn.Dropout(p=drop_p, inplace=False),
                nn.BatchNorm1d(middle_features),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu= 0, prior_sigma= 0.1, in_features=middle_features, out_features= out_features),
                nn.Dropout(p= drop_p, inplace= False),
                nn.BatchNorm1d(out_features),
                nn.ReLU()
        )