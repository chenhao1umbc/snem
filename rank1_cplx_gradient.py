# this file the python version of rank1 model 

#%% loading dependency
import os
import time 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft 

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data

plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')

t = time.time()
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

def loss_func(vhat, Rsshatnf):
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
    loss = p1 + p2.sum()
    loss.imag = loss.imag - loss.imag
    return loss.sum()

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
max_iter = 81

"initial"
# vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
# Hhat = torch.randn(M, J).to(torch.cdouble)
d = sio.loadmat('data/vhat_Hhat.mat')
vhat, Hhat = torch.tensor(d['vhat']).to(torch.cdouble), torch.tensor(d['Hhat'])
Rb = torch.eye(M).to(torch.cdouble)*100
Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
Rj = torch.zeros(J, M, M).to(torch.cdouble)
ll_traj = []

gamma = vhat.real.clone().requires_grad_()
optim_gamma = torch.optim.SGD([gamma], lr=0.01)

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
    Hhat = Rxshat @ Rsshat.inverse()
    Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
        Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
    Rb = Rb.diag().diag()
    Rb.imag = Rb.imag - Rb.imag
    vj = Rsshatnf.diagonal(dim1=-1, dim2=-2).real
    # vhat.imag = vhat.imag - vhat.imag
    loss_rec = []
    for ii in range(100):
        out = gamma.exp()
        out.retain_grad()
        vhat.real = torch.max(out, torch.tensor(1e-30))
        loss = loss_func(vhat, Rsshatnf)
        optim_gamma.zero_grad()   
        loss.backward(retain_graph=True)
        # print(gamma.grad.max(), 'before clip')
        torch.nn.utils.clip_grad_norm_([gamma], max_norm=500)
        # print(gamma.grad.max(), 'after clip')
        # print(out.grad.max(), 'out.grad max')
        temp = gamma.clone()
        optim_gamma.step()
        torch.cuda.empty_cache()
        loss_rec.append(loss.detach().item())
        if ii > 5 and abs((loss_rec[-1]-loss_rec[-2])/loss_rec[-2])<1e-3:
            print(ii)
            break
    # plt.plot(loss_rec)
    # plt.show()

    "compute log-likelyhood"
    vhat = vhat.detach()
    for j in range(J):
        Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
    ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
    
    if i%40 == 0:
        plt.figure(100)
        plt.plot(ll_traj,'o-')
        plt.show()
        
"display results"
for j in range(J):
    plt.figure(j)
    plt.subplot(1,2,1)
    plt.imshow(vhat[:,:,j].real.cpu())
    plt.title('GD')
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.imshow(vj[:,:,j].cpu())
    plt.title('Math')
    plt.colorbar()
    plt.show()

print('spent time: ',time.time() -t)
# %%
