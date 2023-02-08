import numpy as np
import torch
import numpy.linalg as la
from scipy.spatial.distance import pdist, squareform

def V1_covariance_matrix(kernel_size, size, spatial_freq, center, scale=1):
    """
    Generates the covariance matrix for Gaussian Process with non-stationary 
    covariance. This matrix will be used to generate random 
    features inspired from the receptive-fields of V1 neurons.
    C(x, y) = exp(-|x - y|/(2 * spatial_freq))^2 * exp(-|x - m| / (2 * size))^2 * exp(-|y - m| / (2 * size))^2
    Parameters
    ----------
    dim : tuple of shape (2, 1)
        Dimension of random features.
    size : float
        Determines the size of the random weights 
    spatial_freq : float
        Determines the spatial frequency of the random weights  
    
    center : tuple of shape (2, 1)
        Location of the center of the random weights.
    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
    Returns
    -------
    C : array-like of shape (dim[0] * dim[1], dim[0] * dim[1])
        covariance matrix w/ Tr norm = scale * dim[0] * dim[1]
    """

    x = np.arange(kernel_size)
    y = np.arange(kernel_size)
    size = kernel_size
    yy, xx = np.meshgrid(y, x)
    grid = np.column_stack((xx.flatten(), yy.flatten()))

    a = squareform(pdist(grid, 'sqeuclidean')) # pairwise distances between all entries of grid
    b = la.norm(grid - center, axis=1) ** 2
    c = b.reshape(-1, 1)
    C = np.exp(-a / (2 * spatial_freq ** 2)) * np.exp(-b / (2 * size ** 2)) * np.exp(-c / (2 * size ** 2)) \
        + 1e-5 * np.eye(kernel_size **2 )
    C *= scale * kernel_size * kernel_size / np.trace(C)
    return C




def shift_pad(X, y_shift, x_shift):
    '''
    Given an image, we shift every pixel by x_shift and y_shift. We zero pad the portion
    that ends up outside the original frame. We think of the origin of the image
    as its top left. The co-ordinate frame is the matrix kind, where (a, b) means
    ath row and bth column.
    
    Parameters
    ----------
    img: array-like
        image to shift
        
    y_shift: int
        Pixel shift in the vertical direction
        
    x_shift: int
        Pixel shift in the horizontal direction
    
    Returns
    -------
    img_shifted: array-like with the same shape as img
        Shifted and zero padded image
    '''
    
    img_shifted = np.roll(X, x_shift, axis=1)
    img_shifted = np.roll(img_shifted, y_shift, axis=0)
    
    if y_shift > 0:
        img_shifted[:y_shift, :] = 0
    if y_shift < 0:
        img_shifted[y_shift:, :] = 0
    if x_shift > 0:
        img_shifted[:, :x_shift] = 0
    if x_shift < 0:
        img_shifted[:, x_shift:] = 0
    return img_shifted
    

def V1_weights(num_weights, kernel_size, size, spatial_freq, center=None, scale=1, seed=27):
    """
    Generate random weights inspired by the tuning properties of the 
    neurons in Primary Visual Cortex (V1).
    If a value is given for the center, all generated weights have the same center
    If value is set to None, the centers randomly cover the RF space
    Parameters
    ----------
    num_weights : int
        Number of random weights
    dim : tuple of shape (2,1)
        dim of each random weights
    
    size : float
        Determines the size of the random weights
    spatial_freq : float
        Determines the spatial frequency of the random weights 
    center: tuple of shape (2, 1), default = None
        Location of the center of the random weights
        With default value, the centers uniformly cover the RF space
    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
    seed : int, default=None
        Used to set the seed when generating random weights.
    Returns
    -------
    W : array-like of shape (num_weights, dim[0] * dim[1])
        Matrix of random weights
    """
    np.random.seed(seed)
    if center == None: # centers uniformly cover the visual field
        # first generate centered weights
        c = (int(kernel_size/ 2), int(kernel_size/2)) # center of the visual field
        C = V1_covariance_matrix(kernel_size, size, spatial_freq, c, scale) 
        W_centered = np.random.multivariate_normal(mean=np.zeros(kernel_size **2), cov=C, size=num_weights)
        W_centered = W_centered.reshape(-1, kernel_size, kernel_size)
        
        # shift around to uniformly cover the visual field
        centers = np.random.randint((kernel_size, kernel_size), size=(num_weights, 2))
        shifts = centers - c
        W = np.zeros_like(W_centered)
        for i, [y_shift, x_shift] in enumerate(shifts):
            W[i] = shift_pad(W_centered[i], y_shift, x_shift)
        W = W.reshape(-1, kernel_size * kernel_size)

    elif center is not None:
        C = V1_covariance_matrix(kernel_size, size, spatial_freq, center, scale)
        W = np.random.multivariate_normal(mean=np.zeros(kernel_size **2), cov=C, size=out_channel)
        
    return W


def V1_init(out_channels, in_channels, kernel_size, size, spatial_freq, center=None, scale=1., bias=False, seed=27, tied=False):
    """
    Initialize weights of a Conv2d layer according to receptive fields of V1.
    The bias is turned off by default.
    
    Parameters
    ----------
    layer: torch.nn.Conv2d layer
        Layer that will be initialized
        
    size : float
        Determines the size of the random weights
    spatial_freq : float
        Determines the spatial frequency of the random weights 
    center: tuple of shape (2, 1), default = None
        Location of the center of the random weights
        With default value, the centers uniformly cover the RF space
    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
        
    bias: Bool, default=False
        The bias of the convolutional layer
    seed : int, default=None
        Used to set the seed when generating random weights.
    """

    data = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
    # same weights for each channel
    if tied:
        W =  V1_weights(out_channels, kernel_size,
                        size, spatial_freq, center, scale, seed=seed)
    for chan in range(in_channels):
        if not tied:
            W =  V1_weights(out_channels, kernel_size,
                            size, spatial_freq, center, scale, seed=seed)
        W = torch.Tensor(W)
        data[:, chan, :, :] = W.reshape(out_channels, kernel_size, kernel_size)
    
    return data
