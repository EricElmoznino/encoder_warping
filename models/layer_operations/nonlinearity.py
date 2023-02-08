import torch
from torch import nn
import numpy as np

class nonlinearity(nn.Module):
    
    def __init__(self,operation,dim=None):
        super().__init__()
    
        self.operation = operation
        self.dim = dim

    #def extra_repr(self) -> str:
    #    return 'operation={operation}'.format(**self.__dict__)
    
    def forward(self,x):

        if self.operation == 'Zscore':
            x = x.data.cpu().numpy()
            std = (np.std(x, axis=self.dim, keepdims=True))
            mean = np.mean(x, axis=self.dim, keepdims=True)
            x_norm = (x - mean)/std
            return torch.Tensor(x_norm)

        if self.operation == 'Norm':
            x = x.data.cpu().numpy()
            std = 1
            mean = 0
            x_norm = (x - mean)/std
            return torch.Tensor(x_norm)

        if self.operation == 'Sum2one':
            x = x.data.cpu().numpy()
            m = 1/(np.sum(x, axis=self.dim,keepdims=True))
            x_norm = x*m
            return torch.Tensor(x_norm)
        
        if self.operation == 'ReLU': 
            nl = nn.ReLU()
            return nl(x)

        if self.operation == 'GELU': 
            nl = nn.GELU()
            return nl(x)

        if self.operation == 'Abs': 
            return x.abs()