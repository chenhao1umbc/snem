#%% temp rum
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
# from unet.unet_model import UNetHalf8to50 as UNetHalf
torch.manual_seed(1)

rid = 1000 # running id
I = 2 # how many samples
M, N, F, J = 3, 50, 50, 3
NF = N*F
opts = {}
opts['n_ch'] = 1  
opts['batch_size'] = 64
opts['EM_iter'] = 150
opts['lr'] = 0.001
opts['n_epochs'] = 31
opts['d_gamma'] = 4 # gamma dimesion 16*16 to 200*200
# opts['d_gamma'] = 8 # gamma dimesion 16*16 to 200*200

# x = torch.rand(I, N, F, M, dtype=torch.cdouble)
data = sio.loadmat('../data/nem_ss/x5000M3_shift.mat')
x = torch.tensor(data['x'], dtype=torch.cdouble).permute(0,2,3,1) # [sample, N, F, channel]
# gamma = torch.rand(I, J, 1, opts['d_gamma'], opts['d_gamma'])
gamma = torch.rand(J, 1, opts['d_gamma'], opts['d_gamma']).repeat(I,1,1,1,1)
xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]
data = Data.TensorDataset(xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

#@title shift, EM 150, 31epoch, clip to 1, lr1e-3, 1e-2, 1Hhat, 100Rb, same init
model, optimizer = {}, {}
loss_iter, loss_tr = [], []
for j in range(J):
    model[j] = UNetHalf(opts['n_ch'], 1).cuda()
    optimizer[j] = optim.RAdam(model[j].parameters(),
                    lr= opts['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)
"initial"
vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
Htr = torch.randn(M, J).to(torch.cdouble).repeat(I, 1, 1)
Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

for epoch in range(opts['n_epochs']):    
    for j in range(J):
        for param in model[j].parameters():
            param.requires_grad_(False)
        model[j].eval()

    for i, (x,) in enumerate(tr): # gamma [n_batch, 4, 4]
        #%% EM part
        vhat = vtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        Hhat = Htr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        Rb = Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda()
        g = gtr[i*opts['batch_size']:(i+1)*opts['batch_size']].cuda().requires_grad_()

        x = x.cuda()
        optim_gamma = torch.optim.SGD([g], lr= 0.01)
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
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
                out[..., j] = torch.sigmoid(model[j](g[:,j]).squeeze())
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=10)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}, batch {i}, epoch {epoch}')
                break
        print('one batch is done')
        if i == 0 :
            plt.plot(ll_traj, '-x')
            plt.title(f'the log-likelihood of the first batch at epoch {epoch}')
            plt.savefig(f'./figures/id_{rid}_log-likelihood at epoch {epoch}')

            plt.imshow(vhat[0,...,0].real.cpu())
            plt.colorbar()
            plt.title(f'1st source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(f'../figures/id_{rid}_vj1 at epoch {epoch}')

            plt.imshow(vhat[0,...,1].real.cpu())
            plt.colorbar()
            plt.title(f'2nd source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(f'../figures/id_{rid}_vj2 at epoch {epoch}')

            plt.imshow(vhat[0,...,2].real.cpu())
            plt.colorbar()
            plt.title(f'3rd source of vj in first sample from the first batch at epoch {epoch}')
            plt.savefig(f'./figures/id_{rid}_vj3 at epoch {epoch}')

        # #%% update neural network
        # with torch.no_grad():
        #     gtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = g.cpu()
        #     vtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = vhat.cpu()
        #     Htr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Hhat.cpu()
        #     Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Rb.cpu()
        g.requires_grad_(False)
        out = torch.randn(opts['batch_size'], N, F, J, device='cuda', dtype=torch.double)
        for j in range(J):
            model[j].train()
            for param in model[j].parameters():
                param.requires_grad_(True)
            out[..., j] = torch.sigmoid(model[j](g[:,j]).squeeze())
            optimizer[j].zero_grad() 
        vhat.real = threshold(out)
        ll, *_ = log_likelihood(x, vhat, Hhat, Rb)
        loss = -ll
        loss.backward()
        for j in range(J):
            torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=10)
            optimizer[j].step()
            torch.cuda.empty_cache()
        loss_iter.append(loss.detach().cpu().item())

    print(f'done with epoch{epoch}')
    plt.plot(loss_iter, '-xr')
    plt.title(f'Loss fuction of all the iterations at epoch{epoch}')
    plt.savefig(f'./figures/id_{rid}_Loss fuction of all at epoch{epoch}')

    loss_tr.append(loss.detach().cpu().item())
    plt.plot(loss_tr, '-or')
    plt.title(f'Loss fuction at epoch{epoch}')
    plt.savefig(f'./figures/id_{rid}_Loss fuction at epoch{epoch}')

    torch.save(model, 'model_4to50_31epoch_1H_100Rb_cold_same_M3_shift_v1.pt')
#%%
