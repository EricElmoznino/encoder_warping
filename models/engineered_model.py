from .layer_operations.convolution import *
from .layer_operations.output import Output
import torch
from torch import nn
                         

class InteractionsModel(nn.Module):
    
    def __init__(self,
                c1: nn.Module,
                c2: nn.Module,
                c2_batch: int,
                last : nn.Module,
                representation_size : int
                ):
        
        super(InteractionsModel, self).__init__()
        

        
        self.c1 = c1 
        self.c2 = c2
        self.c2_batch = c2_batch
        self.last = last
        self.representation_size = representation_size
        # print("representation size", self.representation_size)
        
        
    def forward(self, x_orig:nn.Module):
        
        
        #conv layer 1
        x_c1 = self.c1(x_orig)
        # print('conv1', x_c1.shape)
    
        
        #conv layer 2
        conv_2 = []
        for i in range(self.c2_batch):
            conv_2.append(self.c2(x_c1)) 
        x_c2 = torch.cat(conv_2,dim=1)
        
        # print('conv2', x_c2.shape)
    
        
        x = self.last(x_c2)
        # print('output', x.shape)
        return x    



  