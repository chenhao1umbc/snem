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
# from unet.unet_model import UNet

# from torch.utils.tensorboard import SummaryWriter
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.float64)

"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')


# %% EM algorithm for one complex sample
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
    Rx = (Rx + Rx.transpose(-1, -2).conj())/2
    l = -(np.pi*Rx.det()).log() - (x[..., None, :]@Rx.inverse()@x[..., None]).squeeze()
    return l.sum()

"reproduce the Matlab result"
d = sio.loadmat('data/x1M3_cpx.mat')
x, c = torch.tensor(d['x'], dtype=torch.cdouble), \
    torch.tensor(d['c'], dtype=torch.cdouble)
M, N, F, J = c.shape
NF = N*F
x = x.permute(1,2,0)  # shape of [N, F, M]
c = c.permute(1,2,3,0) # shape of [N, F, J, M]

"loade data"
d = sio.loadmat('data/v.mat')
vj = torch.tensor(d['v'])
pwr = torch.ones(1, 3)  # signal powers
max_iter = 200

"initial"
vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
Rb = torch.eye(M).to(torch.cdouble)
Hhat = torch.randn(M, J).to(torch.cdouble)
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
    
    if i%50 == 0:
        plt.figure(100)
        plt.plot(ll_traj,'o-')
        plt.show()
        
"display results"
for j in range(J):
    plt.figure(j)
    plt.subplot(1,2,1)
    plt.imshow(vhat[:,:,j].real)
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.imshow(vj[:,:,j])
    plt.title('Ground-truth')
    plt.colorbar()
    plt.show()


#%% Neural EM algorithm

# data = h5py.File('data/x5000M5.mat', 'r')
# x = torch.tensor(data['x'], dtype=torch.float) # [sample, N, F, channel]
# xtr, xcv, xte = x[:4000], x[4000:4500], x[4500:]

