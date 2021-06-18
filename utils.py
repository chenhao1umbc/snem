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



def em_func(x, J=3, Hscal=1, Rbscale=100, max_iter=201, lamb=0, show_plot=False):
    max_iter = max_iter

    #  EM algorithm for one complex sample
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

    N, F, M = x.shape
    NF= N*F
    vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
    Hhat = torch.randn(M, J, dtype=torch.cdouble)*Hscal
    Rb = torch.eye(M).to(torch.cdouble)*Rbscale
    Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
    Rj = torch.zeros(J, M, M).to(torch.cdouble)
    ll_traj = []

    for i in range(max_iter):
        "E-step"
        Rs = vhat.diag_embed()
        Rx = Hhat @ Rs @ Hhat.t().conj() + Rb
        W = Rs @ Hhat.t().conj() @ Rx.inverse()
        shat = W @ x[...,None]
        Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - W@Hhat@Rs

        Rsshat = Rsshatnf.sum([0,1])/NF
        Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((0,1))/NF

        "M-step"
        if lamb<0:
            raise ValueError('lambda should be negative')
        elif lamb >0:
            y = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            vhat = -0.5/lamb + 0.5*(1+4*lamb*y)**0.5/lamb
        else:
            vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
        vhat.imag = vhat.imag - vhat.imag
        # print(vhat[:,:,0].flatten()[1424], vhat[:,:,0].real.argmax())
        Hhat = Rxshat @ Rsshat.inverse()
        Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
            Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
        Rb = Rb.diag().diag()
        Rb.imag = Rb.imag - Rb.imag

        "compute log-likelyhood"
        for j in range(J):
            Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
        ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())

    if show_plot:
        plt.figure(100)
        plt.plot(ll_traj,'o-')
        plt.show()
        "display results"
        for j in range(J):
            plt.figure(j)
            plt.subplot(1,2,1)
            plt.imshow(vhat[:,:,j].real)
            plt.colorbar()

    return shat, Hhat, vhat, Rb


#%%
if __name__ == '__main__':
    a, b = torch.rand(3,1).double(), torch.rand(3,1).double()
    x = (a@a.t() + b@b.t()).cuda()

    eps = x.abs().max().requires_grad_(False)
    ll = torch.linalg.cholesky(x + eps*1e-5*torch.ones(x.shape[:-1], device=x.device).diag_embed())
    l = torch.linalg.cholesky(x)
# %%
