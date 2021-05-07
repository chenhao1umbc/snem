# this file the python version of rank1 model 

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
# import torch_optimizer as optim

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


data = h5py.File('data/x5000M5.mat', 'r')
x = torch.tensor(data['x'], dtype=torch.float) # [sample, N, F, channel]
xtr, xcv, xte = x[:4000], x[4000:4500], x[4500:]
gamma = torch.rand(5000, 4, 4)
M, N, F, J = 3, 150, 150, 3
NF = N*F



# for epoch in range(1):    
#     for i, (gamma, v) in enumerate(tr): # gamma [n_batch, n_f, n_t]
#         pass



"initial"
# vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
# Hhat = torch.randn(M, J).to(torch.cdouble)
d = sio.loadmat('data/vhat_Hhat.mat')
vhat, Hhat = torch.tensor(d['vhat']).to(torch.cdouble), torch.tensor(d['Hhat'])
Rb = torch.eye(M).to(torch.cdouble)*100
Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
Rj = torch.zeros(J, M, M).to(torch.cdouble)
ll_traj = []

for i in range(100):
    "E-step"
    Rs = vhat.diag_embed()
    Rx = Hhat @ Rs @ Hhat.t().conj() + Rb
    W = Rs @ Hhat.t().conj() @ Rx.inverse()
    shat = W @ x[...,None]
    Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - W@Hhat@Rs

    Rsshat = Rsshatnf.sum([0,1])/NF
    Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((0,1))/NF

    "M-step"
    vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
    vhat.imag = vhat.imag - vhat.imag
    Hhat = Rxshat @ Rsshat.inverse()
    Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
        Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
    Rb = Rb.diag().diag()
    Rb.imag = Rb.imag - Rb.imag
    
    "compute log-likelyhood"
    for j in range(J):
        Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
    ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
    





        # to do
        # x = gamma[:,None].cuda().requires_grad_()
        # v = v[:,None].cuda()

        # "update gamma"
        # optim_gamma = torch.optim.SGD([x], lr= opts['lr'])  # every iter the gamma grad is reset
        # out = model(x.diag_embed())
        # loss = ((out - v)**2).sum()/opts['n_batch']
        # optim_gamma.zero_grad()   
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_([x], max_norm=500)
        # optim_gamma.step()
        # torch.cuda.empty_cache()

        # "update model"
        # for param in model.parameters():
        #     param.requires_grad_(True)
        # x.requires_grad_(False)
        # out = model(x.diag_embed())
        # loss = ((out - v)**2).sum()/opts['n_batch']
        # optimizer.zero_grad()   
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=500)
        # optimizer.step()
        # torch.cuda.empty_cache()

