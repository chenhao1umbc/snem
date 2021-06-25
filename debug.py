"""This is file is coded based on cell mode, 
if True gives each cell an indent, so that each cell could be folded in vs code
"""
#%% load dependency 
if True:
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)

#%% load data
    I = 3000 # how many samples
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
    data = sio.loadmat('../../Hpython/data/nem_ss/x3000M3.mat')
    x = torch.tensor(data['x'], dtype=torch.cdouble).permute(0,2,3,1) # [sample, N, F, channel]
    gamma = torch.rand(I, J, 1, opts['d_gamma'], opts['d_gamma'])
    xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
    gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]
    data = Data.TensorDataset(xtr)
    tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

#%% NEM train and debug
    lamb = 0
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
            optim_gamma = torch.optim.SGD([g], lr= 0.01) 

            x = x.cuda()
            vhat = torch.randn(opts['batch_size'], N, F, J).abs().to(torch.cdouble).cuda()
            Hhat = torch.randn(opts['batch_size'], M, J).to(torch.cdouble).cuda()*100
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
                # print((out -vhat.real).norm().detach(), 'if changed')
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_([g], max_norm=10)
                optim_gamma.step()
                torch.cuda.empty_cache()
                for j in range(J):
                    out[..., j] = model[j](g[:,j].detach()).exp().squeeze()
                # loss_after = loss_func(threshold(out.detach()), Rsshatnf.cuda())
                # print(loss.detach().real - loss_after.real, ' loss diff')

                "compute log-likelyhood"
                vhat = vhat.detach()
                ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 3 and ll_traj[-1] < ll_traj[-2]  and  abs((ll_traj[-2] - ll_traj[-1])/ll_traj[-2])>0.1 :
                    input('large descreasing happened')
                if ii > 10 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                    print(f'EM early stop at iter {ii}, batch {i}, epoch {epoch}')
                    break
            print('one batch is done')
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
                torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=1)
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

#%% test part of NEM
    opts['EM_iter'] = 300
    Hscale, Rbscale = 1, 1e2
    lamb = 0
    # models = torch.load('../../Hpython/data/nem_ss/models/model_3000data_3epoch_1Hscale_1e-9lamb.pt')
    models = torch.load('../../Hpython/data/nem_ss/models/model_100H_1e3Rb_31epoch.pt')
    optimizer = {}
    for j in range(J):
        models[j].eval()
        for param in models[j].parameters():
                param.requires_grad_(False)
            
    for i, x in enumerate(xcv[:1]): # gamma [n_batch, 4, 4]
        #%% EM part
        "initial"
        g = gcv[i].cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.05) 

        x = x.cuda()
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
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
            out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
            for j in range(J):
                out[..., j] = models[j](g[None,j]).exp().squeeze()
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda(), lamb=lamb)
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')

        # plot result
        plt.figure()
        plt.plot(ll_traj, '-x')
        plt.title(f'the log-likelihood of validation sample {i+1}')
        plt.show()

        plt.figure()
        plt.imshow(vhat[0,...,0].real.cpu())
        plt.colorbar()
        plt.title(f'1st source of vj of validation sample {i+1}')
        plt.savefig(f'v1.png')
        plt.show()

        plt.figure()
        plt.imshow(vhat[0,...,1].real.cpu())
        plt.colorbar()
        plt.title(f'2nd source of vj of validation sample {i+1}')
        plt.savefig(f'v2.png')
        plt.show()

        plt.figure()
        plt.imshow(vhat[0,...,2].real.cpu())
        plt.colorbar()
        plt.title(f'3rd source of vj of validation sample {i+1}')
        plt.savefig(f'v3.png')
        plt.show()

        cj = Hhat.squeeze() * shat.squeeze().unsqueeze(-2) #[N,F,M,J]
        for j in range(J):
            plt.figure()
            plt.imshow(cj[...,0,j].abs().cpu())
            plt.colorbar()
            plt.savefig(f'c{j}.png')
            plt.show()
            
            plt.figure()
            plt.imshow(cj[...,0,j].abs().log().cpu())
            plt.colorbar()
            plt.savefig(f'log_c{j}.png')
            plt.show()

# %% reguler EM test
    for i, x in enumerate(xcv[:1]):
        shat, Hhat, vhat, Rb = em_func(x,Hscal=1, Rbscale=1e2)
        plt.figure()
        plt.imshow(vhat[...,0].real.cpu())
        plt.colorbar()
        plt.title(f'1st source of vj of validation sample {i+1}')
        plt.savefig(f'v1.png')
        plt.show()

        plt.imshow(vhat[...,1].real.cpu())
        plt.colorbar()
        plt.title(f'2nd source of vj of validation sample {i+1}')
        plt.savefig(f'v2.png')
        plt.show()

        plt.imshow(vhat[...,2].real.cpu())
        plt.colorbar()
        plt.title(f'3rd source of vj of validation sample {i+1}')
        plt.savefig(f'v3.png')
        plt.show()

        cj2 = Hhat.squeeze() * shat.squeeze().unsqueeze(-2) #[N,F,M,J]
        for j in range(J):
            plt.figure()
            plt.imshow(cj2[...,0,j].abs())
            plt.colorbar()
            plt.savefig(f'c{j}.png')
            plt.show()

            plt.figure()
            plt.imshow(cj2[...,0,j].abs().log())
            plt.colorbar()
            plt.savefig(f'log_c{j}.png')
            plt.show()

# %% compare EM vs EM_l1
    import itertools
    d = sio.loadmat('../data/nem_ss/100_test_all.mat') 
    "x shape of [I,M,N,F], c [I,M,N,F,J], h [I,M,J]"
    x_all, c_all, h_all = d['x'], d['c_all'], d['h_all']
    d = sio.loadmat('../data/nem_ss/v.mat')
    v = torch.tensor(d['v'], dtype=torch.cdouble) # shape of [N,F,J]

    def mse(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + (v[...,j] -temp[j]).norm()**2
            r.append(s.item())
        r = sorted(r)
        return r[0]/J

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0]
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def myfun(x_all, v, lamb=0):
        I = x_all.shape[0]
        res_mse, res_corr = [], []
        for i in range(I):
            x = torch.from_numpy(x_all[i]).permute(1,2,0)
            MSE, CORR = [], []
            for ii in range(20):  # for diff initializations
                shat, Hhat, vhat, Rb = em_func(x, seed=ii, lamb=lamb, show_plot=False)
                MSE.append(mse(vhat, v))
                CORR.append(corr(vhat.real, v.real))
            res_mse.append(MSE)
            res_corr.append(CORR)
            print(f'finished {i} samples')
        torch.save((res_mse, res_corr), f'lamb_{lamb}.pt')

    for lamb in [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        myfun(x_all, v, lamb=lamb)

# %% check the EM vs EM_l1 results
    all_lamb = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for lamb in all_lamb:
        mse, corr = torch.load(f'../data/nem_ss/lamb/lamb_{lamb}.pt')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(1, 101), torch.tensor(mse).mean(dim=1))
        plt.boxplot(mse, showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([340, 360])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'MSE result for lambda={lamb}')

        plt.subplot(2,1,2)
        plt.plot(range(1, 101), torch.tensor(corr).mean(dim=1))
        plt.boxplot(corr, showfliers=False)        
        plt.legend(['Mean is blue'])
        plt.ylim([0.5, 0.8])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'Correlation result for lambda={lamb}')

        plt.subplots_adjust(hspace=0.7)
        plt.savefig(f'lambda{lamb}.png')
        plt.show()

    m, c = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}
    for lamb in all_lamb:
        mse, corr = torch.load(f'../data/nem_ss/lamb/lamb_{lamb}.pt')
        m['mean'].append(torch.tensor(mse).mean())
        m['std'].append(torch.tensor(mse).var()**0.5)
        c['mean'].append(torch.tensor(corr).mean())
        c['std'].append(torch.tensor(corr).var()**0.5)
    plt.figure()
    plt.plot(range(len(all_lamb)),np.log(m['mean']), '-x')
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of MSE')
    plt.savefig('Mean of MSE.png')

    plt.figure()
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.plot(np.log(m['std']), '-x')
    plt.title('STD of MSE')
    plt.savefig('STD of MSE.png')

    plt.figure()
    plt.errorbar(range(len(all_lamb)),np.log(m['mean']), abs(np.log(m['std'])), capsize=4)
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of MSE with std')
    plt.savefig('Mean of MSE with std.png')


    plt.figure()
    plt.plot(c['mean'],'-x')
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of Corr.')
    plt.savefig('Mean of Corr.png')

    plt.figure()
    plt.plot(c['std'], '-x')
    plt.xticks(ticks=range(len(all_lamb)), \
    labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('STD of Corr.')
    plt.savefig('STD of Corr.png')

    plt.figure()
    plt.errorbar(range(len(all_lamb)),c['mean'], c['std'], capsize=4)
    plt.xticks(ticks=range(len(all_lamb)), \
        labels=('0','1e-3','0.01','0.1','1','10','100','1e3'))
    plt.xlabel('Lambda')
    plt.title('Mean of Corr with std')
    plt.savefig('Mean of Corr with std.png')

# %%
