#%% load dependency 
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)

#%% load data
I = 3000 # how many samples
M, N, F, J = 3, 50, 50, 3
NF = N*F
opts = {}
opts['batch_size'] = 32
opts['EM_iter'] = 50
opts['gamma_iter'] = 50
opts['n_epochs'] = 200
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

            # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vj.imag = vj.imag - vj.imag
            
            loss_rec = []
            for iii in range(opts['gamma_iter']):
                out = torch.randn(opts['batch_size'], N, F, J, device='cuda', dtype=torch.double)
                for j in range(J):
                    out[..., j] = model[j](g[:,j]).exp().squeeze()
                vhat.real = torch.min(torch.max(out, torch.tensor(1e-20)), torch.tensor(1e3))
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_([g], max_norm=100)
                optim_gamma.step()
                torch.cuda.empty_cache()
                loss_rec.append(loss.detach().item())
                if iii > 3 and abs((loss_rec[-1]-loss_rec[-2])/loss_rec[-2])<1e-3:
                    break
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            for j in range(J):
                Rj[:, j] = Hhat[..., j][..., None] @ Hhat[..., j][..., None].transpose(-1,-2).conj()
            ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
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
        vhat.real = torch.min(torch.max(out, torch.tensor(1e-20)), torch.tensor(1e3))
        loss = loss_func(vhat, Rsshatnf.cuda())
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

# %%
