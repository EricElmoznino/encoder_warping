from .filters import filters
from .nonlinearity import nonlinearity
from torch import nn
from torch.nn import functional as F
import math
import torch



class StandardConvolution(nn.Module):
    def __init__(self, filter_type,
                 filter_size=None,
                 out_channels=None,
                 pooling=None,
                 curv_params=None,
                 ints_size=None,
                 skip_connection=False):
        
        super().__init__()
        
        
        self.out_channels = out_channels
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.ints_size = ints_size
        self.curv_params = curv_params
        self.pooling = pooling
        self.skip_connection = skip_connection
    

    
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels}, kernel_size={filter_size}, filter_type:{filter_type},pooling={pooling},curv_params:{curv_params}'.format(**self.__dict__)
    
    
    
    def forward(self,x):
            
        
        in_channels = x.shape[1]

        weight = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=in_channels,
                         kernel_size=self.filter_size,curv_params=self.curv_params)
        
        weight = weight.cuda()
        x =  x.cuda()
        #torch.cuda.init()
        x = F.conv2d(x,weight=weight,padding=math.floor(weight.shape[-1] / 2))

        
         
        if self.pooling == None:
            pass
        else:
            assert self.pooling[0] in ['max','avg'], "pooling operation should be one of max or avg"
            if self.pooling[0] == 'max':
                mp = nn.MaxPool2d(self.pooling[1])
                x = mp(x)
            else:
                mp = nn.AvgPool2d(self.pooling[1])
                x = mp(x)   

        return x
        

           
      
        
class ScalingConvolution(StandardConvolution):
    
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels}, kernel_size={filter_size}, filter_type:{filter_type},pooling={pooling},curv_params:{curv_params}'.format(**self.__dict__)
    
    
    def forward(self,x):
        weight = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=1,
                         kernel_size=self.filter_size,curv_params=self.curv_params)        

        if x.shape[1] == 1: # for first layer
            x = F.conv2d(x,weight=weight,padding=math.floor(weight.shape[-1] / 2))
        
        elif x.shape[1] == 3:
            out_channels = weight.shape[0]
            #weight = weight.reshape(int(out_channels/3),3,self.filter_size,self.filter_size)
            x = F.conv2d(x,weight=weight,padding=math.floor(weight.shape[-1] / 2),groups=3)
        
        else:
            x = F.conv2d(x,weight=weight.repeat(x.shape[1],1,1,1),padding=math.floor(weight.shape[-1] / 2),groups=x.shape[1])
         
        
        
        if self.pooling == None:
            pass
        else:
            assert self.pooling[0] in ['max','avg'], "pooling operation should be one of max or avg"
            if self.pooling[0] == 'max':
                mp = nn.MaxPool2d(self.pooling[1])
                x = mp(x)
            else:
                mp = nn.AvgPool2d(self.pooling[1])
                x = mp(x)  
        return x
        
    
    
    
    
class RandomProjections(nn.Module): 
    
    def __init__(self, out_channels,max_pool=None):
        super().__init__()

        self.out_channels = out_channels
        self.max_pool = max_pool
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels},max_pool={max_pool}'.format(**self.__dict__)
    
    
    def forward(self, x):
        
        in_channels = x.shape[1] 
        weight = filters(filter_type='1x1',out_channels=self.out_channels,in_channels=in_channels)
 
        
        x = F.conv2d(x,weight=weight,padding=0)
        
         
        if self.pooling == None:
            pass
        else:
            assert self.pooling[0] in ['max','avg'], "pooling operation should be one of max or avg"
            if self.pooling[0] == 'max':
                mp = nn.MaxPool2d(self.pooling[1])
                x = mp(x)
            else:
                mp = nn.AvgPool2d(self.pooling[1])
                x = mp(x)  
        return x
        
        
        
               
    

