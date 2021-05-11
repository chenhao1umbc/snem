#%% load dependency 
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%% load data
I = 200 # how many samples
M, N, F, J = 3, 50, 50, 3
NF = N*F
opts = {}
opts['batch_size'] = 64
opts['EM_iter'] = 150
opts['lr'] = 0.01
opts['n_epochs'] = 200
opts['d_gamma'] = 4 # gamma dimesion 16*16 to 200*200
opts['n_ch'] = 1  

# x = torch.rand(I, 150, 150, M, dtype=torch.cdouble)
data = sio.loadmat('../data/x3000M3.mat')
x = torch.tensor(data['x'], dtype=torch.cdouble).permute(0,2,3,1) # [sample, N, F, channel]
gamma = torch.rand(I, J, 1, opts['d_gamma'], opts['d_gamma'])
xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

#%% neural EM
model, optimizer = {}, {}
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
        optim_gamma = torch.optim.SGD([g], lr= opts['lr']) 

        x = x.cuda()
        vhat = torch.randn(opts['batch_size'], N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(opts['batch_size'], M, J).to(torch.cdouble).cuda()
        Rb = torch.ones(opts['batch_size'], M).diag_embed().cuda().to(torch.cdouble)*100
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rj = torch.zeros(opts['batch_size'], J, M, M).to(torch.cdouble).cuda()
        ll_traj = []

        for ii in range(opts['EM_iter']):
            "E-step"
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [I, N, F, J, J]
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

            # vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vhat.imag = vhat.imag - vhat.imag
            for j in range(J):
                vhat[..., j] = model[j](g[:,j]).exp().squeeze()
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=500)
            optim_gamma.step()
            torch.cuda.empty_cache()
            vhat = vhat.detach()
            
            "compute log-likelyhood"
            for j in range(J):
                Rj[:, j] = Hhat[..., j][..., None] @ Hhat[..., j][..., None].transpose(-1,-2).conj()
            ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}')
                break

        plt.plot(ll_traj, '-x')
        plt.show()
        #%% update neural network
        g.requires_grad_(False)
        gtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = g.cpu()
        for j in range(J):
            model[j].train()
            for param in model[j].parameters():
                param.requires_grad_(True)
            vhat[..., j] = model[j](g[:,j]).exp().squeeze()
            optimizer[j].zero_grad() 

        loss = loss_func(vhat, Rsshatnf.cuda())
        loss.backward()
        for j in range(J):
            torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=500)
            optimizer[j].step()
            torch.cuda.empty_cache()
    print(f'done with epoch{epoch}')

# %%
