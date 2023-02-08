import torch
from torch import nn
import itertools
import numpy as np
from random import sample, seed

class Interactions(nn.Module):
    
    def __init__(self, skip_connection = False, out_channels=None, max_pool=None, batched=False):
        super().__init__()
        
        self.out_channels = out_channels
        self.max_pool = max_pool
        self.batched = batched
    
    def extra_repr(self) -> str:
        return 'operation={operation}, out_channels={out_channels}'.format(**self.__dict__)
        
    def forward(self,x):

            seed(27)
            if x.dim() == 4:
                l = np.arange(0,x.shape[1],1) # all numbers
            elif x.dim() == 5:
                l = np.arange(0,x.shape[2],1)
                
            all_pairs = [[a, b] for idx, a in enumerate(l) for b in l[idx + 1:]] # all pairs
            print(f'number of all possible pairs of interactions: {len(all_pairs)}')
            rand_pairs = np.array(sample(all_pairs,self.out_channels)) # sample of random numbers
            m, n = rand_pairs[:,0], rand_pairs[:,1] 
            
            if self.batched:
                start, batch = 0, 100
                batches = []

                while start < len(m):
                    m_batched, n_batched = m[start:start+batch], n[start:start+batch]

                    # for each image, multiply the channels represented by m and n
                    if x.dim() == 4:
                        mul = torch.mul(x[:,m_batched,:,:],x[:,n_batched,:,:])

                    elif x.dim() == 5:
                        mul = torch.mul(x[:,:,m_batched,:,:],x[:,:,n_batched,:,:])
                    
                    batches.append(mul)
                    start += batch

                axis=1 if x.dim() == 4 else 2
                ints = torch.cat(batches,axis=axis)    

            
               
            else:
                
                if x.dim() == 4:
                    ints = torch.mul(x[:,m,:,:],x[:,n,:,:])

                elif x.dim() == 5:
                    ints = torch.mul(x[:,:,m,:,:],x[:,:,n,:,:])

                
            if self.max_pool == None:
                pass
            else:
                mp = nn.MaxPool2d(self.max_pool)
                ints = mp(ints)
            
            return ints