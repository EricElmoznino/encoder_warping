import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from .v1_random_conv import V1_init
from scipy.ndimage import gaussian_filter
import random
# from kymatio.scattering2d.filter_bank import filter_bank

# # random filter generator
# def make_random_filters(out_channels,in_channels,kernel_size,gau_smooth=True):
#     torch.manual_seed(27)
#     w = torch.rand(out_channels,in_channels,kernel_size,kernel_size)
#     w -= w.mean(dim = [2,3],keepdim=True) # mean centering
#     if gau_smooth:
#         num_smoothed = round(out_channels*0.5)
#         idx_smoothed = random.sample(list(np.arange(0,out_channels)), num_smoothed)
#         channels_smoothed = torch.Tensor(gaussian_filter(w[idx_smoothed,:,:,:], sigma=5))
#         w[idx_smoothed,:,:,:] = channels_smoothed
        
#     return w


def make_low_pass_filters(out_channels,in_channels,kernel_size):
    low_pass = filter_bank(M=kernel_size, N=kernel_size, J=4, L=8)
    w = np.zeros((out_channels,kernel_size,kernel_size))
    for i in range(out_channels):
        w[i,:,:] = low_pass['psi'][i]['levels'][0]
    w = torch.Tensor(w)
    w = torch.unsqueeze(w,1)
    w = w.repeat(1,in_channels,1,1)
    return w



# random filter generator
def make_random_filters(out_channels,in_channels,kernel_size,gau_smooth=False):
    torch.manual_seed(27)
    w = torch.rand(out_channels,in_channels,kernel_size,kernel_size)
    w -= w.mean(dim = [2,3],keepdim=True) # mean centering
    if gau_smooth:
        num_smoothed = round(out_channels*0.5)
        idx_smoothed = random.sample(list(np.arange(0,out_channels)), num_smoothed)
        for i in idx_smoothed:
            channel_smoothed = torch.Tensor(gaussian_filter(w[i,:,:,:], sigma=1))
            w[i,:,:,:] = channel_smoothed
        
    return w



class EdgeModel(nn.Module):
    def __init__(self, n_ories=8, gau_sizes=(1,), filt_size=9, fre=1.2, gamma=1, sigx=1, sigy=1):
        super().__init__()

        self.n_ories = n_ories
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy


                
    def forward(self):
        # Construct filters
        i = 0
        ories = np.arange(0, np.pi, np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.gau_sizes), 1, self.filt_size, self.filt_size))
        for gau_size in self.gau_sizes:
            for orie in ories:
                w[i, 0, :, :] = banana_filter(gau_size, self.fre, orie, 0, self.gamma, self.sigx, self.sigy, self.filt_size)
                i += 1
        return w


# curvature filter generator
class CurvatureModel(nn.Module):
    
    def __init__(self,
                 n_ories=16,
                 in_channels=1,
                 curves=np.logspace(-2, -0.1, 5),
                 gau_sizes=(5,), filt_size=9, fre=[1.2], gamma=1, sigx=1, sigy=1):
        super().__init__()

        self.n_ories = n_ories
        self.curves = curves
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy
        self.in_channels = in_channels

    def forward(self):
        i = 0
        ories = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.curves) * len(self.gau_sizes) * len(self.fre), self.in_channels, self.filt_size, self.filt_size))
        for curve in self.curves:
            for gau_size in self.gau_sizes:
                for orie in ories:
                    for f in self.fre:
                        w[i, 0, :, :] = banana_filter(gau_size, f, orie, curve, self.gamma, self.sigx, self.sigy, self.filt_size)
                        i += 1
        return w        


def banana_filter(s, fre, theta, cur, gamma, sigx, sigy, sz):
    # Define a matrix that used as a filter
    xv, yv = np.meshgrid(np.arange(np.fix(-sz/2).item(), np.fix(sz/2).item() + sz % 2),
                         np.arange(np.fix(sz/2).item(), np.fix(-sz/2).item() - sz % 2, -1))
    xv = xv.T
    yv = yv.T

    # Define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # Define the bias term
    bias = np.exp(-sigx / 2)
    k = xc + cur * (xs ** 2)

    # Define the rotated Guassian rotated and curved function
    k2 = (k / sigx) ** 2 + (xs / (sigy * s)) ** 2
    G = np.exp(-k2 * fre ** 2 / 2)

    # Define the rotated and curved complex wave function
    F = np.exp(fre * k * 1j)

    # Multiply the complex wave function with the Gaussian function with a constant and bias
    filt = gamma * G * (F - bias)
    filt = np.real(filt)
    filt -= filt.mean()

    filt = torch.from_numpy(filt).float()
    return filt





# random projections
def make_onebyone_filters(out_channels,in_channels):
    torch.manual_seed(27)
    w = torch.randn(out_channels,in_channels, 1, 1) 
    w = w/np.sqrt(out_channels)
    #w = w/(torch.sqrt(torch.sum(torch.square(w))))
    return w









# filters function
def filters(filter_type,out_channels=None,in_channels=None,curv_params = None,kernel_size=None):

        if filter_type == 'Random':
            return make_random_filters(out_channels,in_channels,kernel_size)

        elif filter_type == '1x1':
            return make_onebyone_filters(out_channels,in_channels)

        elif filter_type == 'low_pass':
            return make_low_pass_filters(out_channels,in_channels,kernel_size)
        
        elif filter_type == 'Curvature':
            
            curve = CurvatureModel(
                in_channels=in_channels,
                n_ories=curv_params['n_ories'],
                gau_sizes=curv_params['gau_sizes'],
                curves=np.logspace(-2, -0.1, curv_params['n_curves']),
                fre = curv_params['spatial_fre'],
                filt_size=kernel_size)
            return curve()
        
        elif filter_type == 'gabor':
            gabor = EdgeModel(n_ories=8,
                              gau_sizes=(1,), 
                              filt_size=9, 
                              fre=1.2, 
                              gamma=1, 
                              sigx=1, 
                              sigy=1)
            return gabor()            
        
        elif filter_type == 'V1_Inspired':
            return V1_init(out_channels=out_channels, in_channels=in_channels, kernel_size=kernel_size, size=kernel_size, spatial_freq=1.2, center=None, scale=1., bias=False, seed=None, tied=False)
