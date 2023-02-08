from .engineered_model import *
from .layer_operations.convolution import *
from .layer_operations.output import Output

                              
    
    
class EngineeredModel:
    
    def __init__(self, curv_params = {'n_ories':3,'n_curves':8,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=10,reps=1):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.reps = reps
        
    
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=45,filter_type='Curvature',pooling=('max',6),curv_params=self.curv_params)       
        c2 = StandardConvolution(out_channels=self.filters_2,filter_size=9,filter_type='Random',pooling=('max',8))
        last = Output()

        return InteractionsModel(c1,c2,self.reps,last)  
    
    
    #
    
    
    
    

    
    
    

