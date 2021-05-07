#%% load dependency 
from utils import *

#%% load data
# data = h5py.File('data/x5000M5.mat', 'r')
# x = torch.tensor(data['x'], dtype=torch.float) # [sample, N, F, channel]
I = 500 # how many samples
M, N, F, J = 3, 150, 150, 3
NF = N*F
opt = {}
opt.batch_size = 64

x = torch.rand(I, 150, 150, M, dtype=torch.cdouble)
gamma = torch.rand(I, 4, 4)
xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]



#%% neural EM
# for epoch in range(1):    
#     for i, (gamma, v) in enumerate(tr): # gamma [n_batch, n_f, n_t]
#         pass



"initial"
vhat = torch.randn(opt.batch_size, N, F, J).abs().to(torch.cdouble)
Hhat = torch.randn(opt.batch_size, M, J).to(torch.cdouble)
Rb = torch.eye(opt.batch_size, M).to(torch.cdouble)*100
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

