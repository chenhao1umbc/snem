#%% load dependency 
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%% load data
I = 300 # how many samples
M, N, F, J = 3, 50, 50, 3
NF = N*F
opts = {}
opts['batch_size'] = 64
opts['EM_iter'] = 150
opts['n_epochs'] = 100
opts['lr'] = 0.01
opts['d_gamma'] = 4 # gamma dimesion 16*16 to 200*200
opts['n_ch'] = 1  

# x = torch.rand(I, N, F, M, dtype=torch.cdouble)
data = sio.loadmat('data/x3000M3_shift.mat')
x = torch.tensor(data['x'], dtype=torch.cdouble).permute(0,2,3,1) # [sample, N, F, channel]
gamma = torch.rand(I, J, 1, opts['d_gamma'], opts['d_gamma'])
xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

#%% neural EM
model, optimizer = {}, {}
loss_iter, loss_tr = [], []
for j in range(J):
    model[j] = UNetHalf(opts['n_ch'], 1).cuda()
    optimizer[j] = optim.RAdam(model[j].parameters(),
                    lr= opts['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)

for epoch in range(opts['n_epochs']):    
    for j in range(J):
        for param in model[j].parameters():
            param.requires_grad_(False)
        model[j].eval()

    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        #%% EM part
        "initial"
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.05) 

        x = x.cuda()
        vhat = torch.randn(opts['batch_size'], N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(opts['batch_size'], M, J).to(torch.cdouble).cuda()
        Rb = torch.ones(opts['batch_size'], M).diag_embed().cuda().to(torch.cdouble)*100
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [I, N, F, J, J]
        ll_traj = []

        for ii in range(opts['EM_iter']):
            "E-step"
            W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
            shat = W.permute(2,0,1,3,4) @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
            Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

            "M-step"
            Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
            Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
            Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
            Rb.imag = Rb.imag - Rb.imag

            # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vj.imag = vj.imag - vj.imag
            out = torch.randn(opts['batch_size'], N, F, J, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = model[j](g[:,j]).exp().squeeze()
            vhat.real = threshold(out)
            print((out -vhat.real).norm().detach(), 'if changed')
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=10)
            optim_gamma.step()
            torch.cuda.empty_cache()
            for j in range(J):
                out[..., j] = model[j](g[:,j].detach()).exp().squeeze()
            loss_after = loss_func(threshold(out.detach()), Rsshatnf.cuda())
            print(loss.detach().real - loss_after.real, ' loss diff')

            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            # if i == 1:
            #     temp = loss_func2(x, vhat, Rj, Rb, Hhat)
            if ii > 3 and ll_traj[-1] < ll_traj[-2]  and  abs((ll_traj[-2] - ll_traj[-1])/ll_traj[-2])>0.1 :
                input('large descreasing happened')
            if ii > 10 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}, batch {i}, epoch {epoch}')
                break
        
        if i == 0 :
            plt.plot(ll_traj, '-x')
            plt.title(f'the log-likelihood of the first batch at epoch {epoch}')
            plt.show()

            plt.imshow(vhat[0,...,0].real.cpu())
            plt.colorbar()
            plt.title(f'1st channel of vj in first sample from the first batch at epoch {epoch}')
            plt.show()
            plt.imshow(vhat[0,...,1].real.cpu())
            plt.colorbar()
            plt.title(f'2nd channel of vj in first sample from the first batch at epoch {epoch}')
            plt.show()
            plt.imshow(vhat[0,...,2].real.cpu())
            plt.colorbar()
            plt.title(f'3rd channel of vj in first sample from the first batch at epoch {epoch}')
            plt.show()
        #%% update neural network
        g.requires_grad_(False)
        gtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = g.cpu()
        out = torch.randn(opts['batch_size'], N, F, J, device='cuda', dtype=torch.double)
        for j in range(J):
            model[j].train()
            for param in model[j].parameters():
                param.requires_grad_(True)
            out[..., j] = model[j](g[:,j]).exp().squeeze()
            optimizer[j].zero_grad() 
        vhat.real = threshold(out)
        ll, *_ = log_likelihood(x, vhat, Hhat, Rb)
        loss = -ll
        loss.backward()
        for j in range(J):
            torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=100)
            optimizer[j].step()
            torch.cuda.empty_cache()
        loss_iter.append(loss.detach().cpu().item())

    print(f'done with epoch{epoch}')
    plt.plot(loss_iter, '-xr')
    plt.title(f'Loss fuction of all the iterations at epoch{epoch}')
    plt.show()

    loss_tr.append(loss.detach().cpu().item())
    plt.plot(loss_tr, '-or')
    plt.title(f'Loss fuction at epoch{epoch}')
    plt.show()



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
max_iter = 201

"initial"
vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
Hhat = torch.randn(M, J).to(torch.cdouble)
# d = sio.loadmat('data/vhat_Hhat.mat')
# vhat, Hhat = torch.tensor(d['vhat']).to(torch.cdouble), torch.tensor(d['Hhat'])
Rb = torch.eye(M).to(torch.cdouble)*1e2
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
    # print('max, mean, median', vhat.real.max(), vhat.real.mean(), vhat.real.median())
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


# %% 
