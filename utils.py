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
from unet.unet_model import UNetHalf4to50 as UNetHalf
import torch_optimizer as optim

# from torch.utils.tensorboard import SummaryWriter
"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')


#%% functions
def loss_func(vhat, Rsshatnf, lamb=0):
    """This function is only the Q1 part, which is related to vj
        Q= Q1 + Q2, Q1 = \sum_nf -log(|Rs(n,f)|) - tr{Rsshat_old(n,f)Rs^-1(n,f))}
        loss = -Q1

    Args:
        vhat ([tensor]): [shape of [batch, N, F, J]]
        Rsshatnf ([tensor]): [shape of [batch, N, F, J, J]]

    Returns:
        [scalar]: [-Q1]
    """
    J = vhat.shape[-1]
    shape = vhat.shape[:-1]
    det_Rs = torch.ones(shape).to(vhat.device)
    for j in range(J):
        det_Rs = det_Rs * vhat[..., j]
    p1 = det_Rs.log().sum() 
    p2 = Rsshatnf.diagonal(dim1=-1, dim2=-2)/vhat
    loss = p1 + p2.sum() + lamb*vhat.abs().sum()
    return loss.sum()

def log_likelihood(x, vhat, Hhat, Rb, lamb=0):
    """ Hhat shape of [I, M, J]
        vhat shape of [I, N, F, J]
        Rb shape of [I, M, M]
        x shape of [I, N, F, M]
    """
    Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
    Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
    Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    l = lamb*vhat.abs().sum() -(np.pi*mydet(Rx)).log() - \
        (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze() 
    return l.sum().real, Rs, Rxperm


def calc_ll_cpx2(x, vhat, Rj, Rb):
    """ Rj shape of [I, J, M, M]
        vhat shape of [I, N, F, J]
        Rb shape of [I, M, M]
        x shape of [I, N, F, M]
    """
    _, _, M, M = Rj.shape
    I, N, F, J = vhat.shape
    Rcj = vhat.reshape(I, N*F, J) @ Rj.reshape(I, J, M*M)
    Rcj = Rcj.reshape(I, N, F, M, M).permute(1,2,0,3,4)
    Rx = (Rcj + Rb).permute(2,0,1,3,4) # shape of [I, N, F, M, M]
    l = -(np.pi*mydet(Rx)).log() - (x[..., None, :].conj()@Rx.inverse()@x[..., None]).squeeze()
    return l.sum().real


def mydet(x):
    """calc determinant of tensor for the last 2 dimensions,
    suppose x is postive definite hermitian matrix

    Args:
        x ([pytorch tensor]): [shape of [..., N, N]]
    """
    s = x.shape[:-2]
    N = x.shape[-1]
    try:
        l = torch.linalg.cholesky(x)
    except:
        eps = eps = x.abs().max().requires_grad_(False)
        l = torch.linalg.cholesky(x + eps*1e-5*torch.ones(x.shape[:-1], device=x.device).diag_embed())
        print('low rank happend')
    ll = l.diagonal(dim1=-1, dim2=-2)
    res = torch.ones(s).to(x.device)
    for i in range(N):
        res = res * ll[..., i]**2
    return res


def threshold(x, floor=1e-20, ceiling=1e3):
    y = torch.min(torch.max(x, torch.tensor(floor)), torch.tensor(ceiling))
    return y


#%%
if __name__ == '__main__':
    a, b = torch.rand(3,1).double(), torch.rand(3,1).double()
    x = (a@a.t() + b@b.t()).cuda()

    eps = x.abs().max().requires_grad_(False)
    ll = torch.linalg.cholesky(x + eps*1e-5*torch.ones(x.shape[:-1], device=x.device).diag_embed())
    l = torch.linalg.cholesky(x)
# %%
