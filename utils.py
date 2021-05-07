#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft 

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
from unet.unet_model import UNetHalf
import torch_optimizer as optim

# from torch.utils.tensorboard import SummaryWriter
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.float64)

"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')


#%% Neural EM algorithm
def loss_func(x, s, vhat, Rsshat, Rb):
    pass
    return 0

def calc_ll_cpx2(x, vhat, Rj, Rb):
    """ Rj shape of [J, M, M]
        vhat shape of [N, F, J]
        Rb shape of [M, M]
        x shape of [N, F, M]
    """
    _, M, M = Rj.shape
    N, F, J = vhat.shape
    Rcj = vhat.reshape(N*F, J) @ Rj.reshape(J, M*M)
    Rcj = Rcj.reshape(N, F, M, M)
    Rx = Rcj + Rb 
    # l = -(np.pi*Rx.det()).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
    l = -(np.pi*mydet(Rx)).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
    return l.sum()

def mydet(x):
    """calc determinant of tensor for the last 2 dimensions,
    suppose x is postive definite hermitian matrix

    Args:
        x ([pytorch tensor]): [shape of [..., N, N]]
    """
    s = x.shape[:-2]
    N = x.shape[-1]
    l = torch.linalg.cholesky(x)
    ll = l.diagonal(dim1=-1, dim2=-2)
    res = torch.ones(s).to(x.device)
    for i in range(N):
        res = res * ll[..., i]**2
    return res
