"""This is file is coded based on cell mode, 
if True gives each cell an indent, so that each cell could be folded in vs code
"""
#%% load dependency 
if True:
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
    vtr = torch.randn(I, N, F, J).abs().to(torch.cdouble)
    Htr = torch.randn(I, M, J).to(torch.cdouble)
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
            optim_gamma = torch.optim.SGD([g], lr= 0.05)
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
                    out[..., j] = model[j](g[:,j]).exp().squeeze()
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
                if ii > 3 and abs((ll_traj[ii] - ll_traj[ii-1])/ll_traj[ii-1]) <1e-3:
                    print(f'EM early stop at iter {ii}, batch {i}, epoch {epoch}')
                    break
            print('one batch is done')
            if i == 0 :
                plt.plot(ll_traj, '-x')
                plt.title(f'the log-likelihood of the first batch at epoch {epoch}')
                plt.show()

                plt.imshow(vhat[0,...,0].real.cpu())
                plt.colorbar()
                plt.title(f'1st source of vj in first sample from the first batch at epoch {epoch}')
                plt.show()
                plt.imshow(vhat[0,...,1].real.cpu())
                plt.colorbar()
                plt.title(f'2nd source of vj in first sample from the first batch at epoch {epoch}')
                plt.show()
                plt.imshow(vhat[0,...,2].real.cpu())
                plt.colorbar()
                plt.title(f'3rd source of vj in first sample from the first batch at epoch {epoch}')
                plt.show()
            #%% update neural network
            with torch.no_grad():
                gtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = g.cpu()
                vtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = vhat.cpu()
                Htr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Hhat.cpu()
                Rbtr[i*opts['batch_size']:(i+1)*opts['batch_size']] = Rb.cpu()
            g.requires_grad_(False)
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
                torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=10)
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

#%% reguler EM test
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

#%% compare EM vs EM_l1
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

#%% plot the EM vs EM_l1 results
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

#%% test CNN nem
    import itertools, time
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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=150, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            models = torch.load('../../Hpython/data/nem_ss/models/model_4to50_em20_151epoch_1H_100Rb_v1.pt')
        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        x = x.cuda()

        torch.torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g = torch.rand(J, 1, 4, 4).cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.05)
        ll_traj = []

        for ii in range(max_iter): # EM iterations
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
            if ii > 3 and abs((ll_traj[ii] - ll_traj[ii-1])/ll_traj[ii-1]) <1e-3:
                # print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_mse, res_corr = [], []
    for id in range(1,6):
        location = f'../../Hpython/data/nem_ss/models/model_4to50_21epoch_1H_100Rb_cold_same_M3_v{id}.pt'
        for i in range(I):
            x = torch.from_numpy(x_all[i]).permute(1,2,0)
            MSE, CORR = [], []
            for ii in range(20):  # for diff initializations
                shat, Hhat, vhat, Rb = nem_func(x, seed=ii, model=location, show_plot=False)
                MSE.append(mse(vhat, v))
                CORR.append(corr(vhat.real, v.real))
            res_mse.append(MSE)
            res_corr.append(CORR)
            print(f'finished {i} samples')
        torch.save((res_mse, res_corr), f'nem_CNN_v{id}.pt')

#%% plot nem results
    for i in range(1,6):
        mse, corr = torch.load(f'../data/nem_ss/nem_res/nem_20iter_v{i}.pt')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(1, 101), torch.tensor(mse[-100:]).mean(dim=1))
        plt.boxplot(mse[-100:], showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([340, 360])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'MSE result for NEM-{i}')

        plt.subplot(2,1,2)
        plt.plot(range(1, 101), torch.tensor(corr[-100:]).mean(dim=1))
        plt.boxplot(corr[-100:], showfliers=True)        
        plt.legend(['Mean is blue'])
        # plt.ylim([0.5, 0.8])
        plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
        plt.xlabel('Sample index')
        plt.title(f'Correlation result for NEM-{i}')

        plt.subplots_adjust(hspace=0.7)
        plt.savefig(f'NEM-{i}.png')
        plt.show()

    # m, c = {'mean':[], 'std':[]}, {'mean':[], 'std':[]}
    # for i in range(1,6):
    #     mse, corr = torch.load(f'nem_v{i}.pt')
    #     m['mean'].append(torch.tensor(mse).mean())
    #     m['std'].append(torch.tensor(mse).var()**0.5)
    #     c['mean'].append(torch.tensor(corr).mean())
    #     c['std'].append(torch.tensor(corr).var()**0.5)
    # plt.figure()
    # plt.plot(range(5),np.log(m['mean']), '-x')
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of MSE')
    # plt.savefig('Mean of MSE.png')

    # plt.figure()
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.plot(np.log(m['std']), '-x')
    # plt.title('STD of MSE')
    # plt.savefig('STD of MSE.png')

    # plt.figure()
    # plt.errorbar(range(5),np.log(m['mean']), abs(np.log(m['std'])), capsize=4)
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of MSE with std')
    # plt.savefig('Mean of MSE with std.png')


    # plt.figure()
    # plt.plot(c['mean'],'-x')
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of Corr.')
    # plt.savefig('Mean of Corr.png')

    # plt.figure()
    # plt.plot(c['std'], '-x')
    # plt.xticks(ticks=range(5), \
    # labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('STD of Corr.')
    # plt.savefig('STD of Corr.png')

    # plt.figure()
    # plt.errorbar(range(len(all_lamb)),c['mean'], c['std'], capsize=4)
    # plt.xticks(ticks=range(5), \
    #     labels=('model-1','model-2','model-3','model-4','model-5'))
    # plt.xlabel('Lambda')
    # plt.title('Mean of Corr with std')
    # plt.savefig('Mean of Corr with std.png')

#%% EM with fixed random initialization M=5
    import itertools
    d = sio.loadmat('../data/nem_ss/100_test_all_M5.mat') 
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

    I = x_all.shape[0]
    res_mse, res_corr = [], []
    for i in range(I):
        x = torch.from_numpy(x_all[i]).permute(1,2,0)
        MSE, CORR = [], []
        shat, Hhat, vhat, Rb = em_func(x, seed=0, show_plot=False)
        plt.figure()
        plt.imshow(vhat[...,0].real)
        plt.show()
        MSE.append(mse(vhat, v))
        CORR.append(corr(vhat.real, v.real))
        res_mse.append(MSE)
        res_corr.append(CORR)
        print(f'finished {i} samples')

#%% MUSIC algorithm for DOA
    for i in range(20):
        x = torch.from_numpy(x_all[i]).permute(1,2,0)
        Rx = (x[..., None] @ x[:,:,None,:].conj()).sum([0,1])/2500
        l, v = torch.linalg.eigh(Rx)
        un = v[:,:2]
        res = []
        for i in torch.arange(0, np.pi, np.pi/100):
            omeg = i
            e = torch.tensor([1, np.exp(1*1j*omeg), np.exp(2*1j*omeg), np.exp(3*1j*omeg), np.exp(4*1j*omeg)])
            P = 1/(e[None, :]@un@un.conj().t()@e[:,None])
            res.append(abs(P))
        plt.figure()
        plt.plot(res)

#%% Test FCN nem on static toy
    import itertools
    class Fcn(nn.Module):
        def __init__(self, input, output):
            super().__init__()
            self.fc = nn.Linear(input , output)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.fc(x)
            x2 = self.sigmoid(x1)
            return x2

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

    d = sio.loadmat('../data/nem_ss/100_test_all.mat') 
    "x shape of [I,M,N,F], c [I,M,N,F,J], h [I,M,J]"
    x_all, c_all, h_all = d['x'], d['c_all'], d['h_all']
    d = sio.loadmat('../data/nem_ss/v.mat')
    v = torch.tensor(d['v'], dtype=torch.cdouble) # shape of [N,F,J]

    def nem_fcn(x, J=3, Hscale=1, Rbscale=100, max_iter=150, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            print('A FCN model is needed')
            return None
        models = torch.load(model)
        for param in models.parameters():
            param.requires_grad_(False)

        #%% EM part
        N, F, M = x.shape
        NF= N*F
        x = x.cuda()

        torch.torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g = torch.rand(1, J, 250).cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= 0.01)
        ll_traj = []

        for ii in range(max_iter):
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
            out = models(g)
            vhat.real = threshold(out.permute(0,2,1).reshape(1,N,F,J))
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
            if ii > 3 and abs((ll_traj[ii] - ll_traj[ii-1])/ll_traj[ii-1]) <1e-3:
                # print(f'EM early stop at iter {ii}')
                break
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
                plt.show()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_mse, res_corr = [], []
    for id in range(1,6):
        location = f'../../Hpython/data/nem_ss/models/model_FCN_21epoch_1H_100Rb_cold_same_M3_v{id}.pt'
        for i in range(I):
            x = torch.from_numpy(x_all[i]).permute(1,2,0)
            MSE, CORR = [], []
            for ii in range(20):  # for diff initializations
                shat, Hhat, vhat, Rb = nem_fcn(x, seed=ii, model=location, show_plot=False)
                MSE.append(mse(vhat, v))
                CORR.append(corr(vhat.real, v.real))
            res_mse.append(MSE)
            res_corr.append(CORR)
            print(f'finished {i} samples')
        torch.save((res_mse, res_corr), f'nem_FCN_v{id}.pt')

#%% Test EM on dynamic toy
    import itertools
    d = sio.loadmat('../data/nem_ss/test100M3_shift.mat')
    vj_all = torch.tensor(d['vj']).to(torch.cdouble)  # shape of [I, N, F, J]
    x_all = torch.tensor(d['x']).permute(0,2,3,1)  # shape of [I, M, N, F]
    cj = torch.tensor(d['cj'])  # shape of [I, M, N, F, J]

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

    import time
    t = time.time()
    I = x_all.shape[0]
    res_corr = []
    for i in range(I):
        CORR = []
        for ii in range(20):
            shat, Hhat, vhat, Rb = em_func(x_all[i], seed=ii, show_plot=False)
            CORR.append(corr(vhat.real, vj_all[i].real))
            # plt.figure()
            # plt.imshow(vhat[...,0].real)
            # plt.show()
        res_corr.append(CORR)
        print(f'finished {i} samples')
    print('Time used is ', time.time()-t)
    # torch.save(res_corr, 'res_toy100shift.pt')

#%% Test NEM on dynamic toy
    import itertools
    from skimage.transform import resize
    d = sio.loadmat('../data/nem_ss/test100M3_shift.mat')
    vj_all = torch.tensor(d['vj']).to(torch.cdouble)  # shape of [I, N, F, J]
    x_all = torch.tensor(d['x']).permute(0,2,3,1)  # shape of [I, M, N, F]
    cj = torch.tensor(d['cj'])  # shape of [I, M, N, F, J]

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=151, lamb=0, seed=0, model='', show_plot=False):
        if model == '':
            print('A model is needed')
        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F      
        gtr = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        gtr = (gtr/gtr.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([gtr[:,None] for j in range(J)], dim=1).cuda().requires_grad_()
        x = x.cuda()

        torch.manual_seed(seed)        
        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        optim_gamma = torch.optim.SGD([g], lr= 0.05)
        ll_traj = []

        for ii in range(max_iter): # EM iterations
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
                out[..., j] = torch.sigmoid(models[j](g[:,j]).squeeze())
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
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    I = x_all.shape[0]
    res_corr = []
    location = f'../data/nem_ss/models/model_rid8300.pt'
    for i in range(3):
        c = []
        for ii in range(3):
            shat, Hhat, vhat, Rb = nem_func(x_all[i], seed=ii,model=location)
            c.append(corr(vhat.real, vj_all[i].real))
        res_corr.append(c)
        print(f'finished {i} samples')
    # torch.save(res_corr, f'nem_toy_shift.pt')

#%% plot EM dynamic toy results 
    res = torch.load('../data/nem_ss/nem_res/res_toy100shift.pt')
    corr = torch.tensor(res)
    plt.figure()
    plt.plot(range(1, 101), torch.tensor(corr[-100:]).mean(dim=1))
    plt.boxplot(corr[-100:], showfliers=True)        
    plt.legend(['Mean is blue'])
    # plt.ylim([0.5, 0.8])
    plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    plt.xlabel('Sample index')
    plt.title('Correlation result for EM')
    plt.show()

#%% Prepare real data
    "raw data processing"
    FT = 128
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        # dd = np.abs(temp['x']).max(axis=1).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    np.set_printoptions(linewidth=150)
    "To generate 5000 mixture samples"
    theta = np.array([15, 60, -45])*np.pi/180  #len=M, signal AOAs  
    h = np.exp(-1j*np.pi*np.arange(0, 3)[:,None]@np.sin(theta)[None, :])  # shape of [M, J]

    np.random.seed(0)
    np.random.shuffle(data[0])
    np.random.shuffle(data[2])
    np.random.shuffle(data[5])
    d1 = h[:,0][:,None]@data[0][:,None,:] + h[:,1][:,None]@data[2][:,None,:] + h[:,2][:,None]@data[5][:,None,:]

    np.random.seed(1)
    np.random.shuffle(data[0])
    np.random.shuffle(data[2])
    np.random.shuffle(data[5])
    d2 = h[:,0][:,None]@data[0][:,None,:] + h[:,1][:,None]@data[2][:,None,:] + h[:,2][:,None]@data[5][:,None,:]

    data_pool = np.concatenate((d1, d2), axis=0)  #[I,M,time_len]
    *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary=None)
    x = torch.tensor(np.roll(Z, FT//2, axis=2))  # roll nperseg//2
    plt.figure()
    plt.imshow(x[0,0].abs().log(), aspect='auto', interpolation='None')
    plt.title('One example of 3-component mixture')
    torch.save(x[:3000], f'tr3kM3FT{FT}.pt')

    "get s and h for the val and test data"
    *_, Z = stft(data[0][1000:1500], fs=4e7, nperseg=FT, boundary=None)
    s1 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    *_, Z = stft(data[2][1000:1500], fs=4e7, nperseg=FT, boundary=None)
    s2 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    *_, Z = stft(data[5][1000:1500], fs=4e7, nperseg=FT, boundary=None)
    s3 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    s = torch.tensor(np.stack((s1, s2, s3), axis=1))  #[I, J, F, T]
    torch.save((x[3000:3500], s, h), f'val500M3FT{FT}_xsh.pt')

    *_, Z = stft(data[0][1500:], fs=4e7, nperseg=FT, boundary=None)
    s1 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    *_, Z = stft(data[2][1500:], fs=4e7, nperseg=FT, boundary=None)
    s2 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    *_, Z = stft(data[5][1500:], fs=4e7, nperseg=FT, boundary=None)
    s3 = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    s = torch.tensor(np.stack((s1, s2, s3), axis=1))  #[I, J, F, T]
    torch.save((x[3500:], s, h), f'test500M3FT{FT}_xsh.pt')

#%% Test EM on real data
    import itertools
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1)

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J
        
    def h_corr(h, hh):
        J = h.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = hh[:,torch.tensor(p)]
            s = 0
            for j in range(J):
                dino = h[:,j].norm() * temp[:, j].norm()
                nume = (temp[:, j].conj() @ h[:, j]).abs()
                s = s + nume/dino
            r.append(s/J)
        r = sorted(r, reverse=True)
        return r[0].item()

    single_data = True
    if single_data:
        ind = 0
        shat, Hhat, vhat, Rb = em_func(x[ind])
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.show()
        # sio.savemat('x0s_em.mat', {'x':x[ind].numpy(),'s_em':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(20):
                shat, Hhat, vhat, Rb = em_func(awgn(x[i], snr=20), seed=ii)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')

#%% Test NEM on real data
    from unet.unet_model import UNetHalf8to100 as UNetHalf
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for jj in permutes:
            temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
            r.append(s)
        r = sorted(r, reverse=True)
        return r[0]/J

    def h_corr(h, hh):
        J = h.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = hh[:,torch.tensor(p)]
            s = 0
            for j in range(J):
                dino = h[:,j].norm() * temp[:, j].norm()
                nume = (temp[:, j].conj() @ h[:, j]).abs()
                s = s + nume/dino
            r.append(s/J)
        r = sorted(r, reverse=True)
        return r[0].item()

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=151, lamb=0, seed=1, model='', show_plot=False):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')
        models = {}  # the following 3 lines are matching training initials
        for j in range(J): #---- see above ---
            models[j] = UNetHalf(1, 1) #---- see above ---
        del models #---- see above ---

        models = torch.load(model)
        for j in range(J):
            models[j].eval()
            for param in models[j].parameters():
                    param.requires_grad_(False)

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        gtr = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        gtr = (gtr/gtr.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([gtr[:,None] for j in range(J)], dim=1).cuda().requires_grad_()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        optim_gamma = torch.optim.SGD([g], lr= 0.001)
        ll_traj = []

        for ii in range(max_iter): # EM iterations
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
                out[..., j] = torch.sigmoid(models[j](g[:,j]).squeeze())
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
            if ii > 5 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <1e-3:
                print(f'EM early stop at iter {ii}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure(j)
                plt.subplot(1,2,1)
                plt.imshow(vhat[...,j].cpu().squeeze().real)
                plt.colorbar()
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()
    
    location = f'../data/nem_ss/models/model_rid5200.pt'
    single_data = True
    if single_data:
        ind = 0
        shat, Hhat, vhat, Rb = nem_func(awgn(x_all[ind], snr=0),seed=1,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            # plt.title(f'Estimated sources {i+1}')
            plt.show()
        print(h_corr(h, Hhat.squeeze()))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()
        # sio.savemat('sshat_nem.mat', {'s':s_all[ind].squeeze().abs().numpy(),'s_nem':(shat.squeeze().abs()*ratio[ind]).numpy()})
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(20):
                shat, Hhat, vhat, Rb = nem_func(x_all[i],seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], 'res_nem_shat_hhat.pt')

#%% show 20db, 10db, 5db, 0db result
    # res, _ = torch.load('../data/nem_ss/nem_res/res_nem_shat_hhat_snr5.pt') # s,h
    # _, res = torch.load('../data/nem_ss/nem_res/res_nem_shat_hhat_snr5.pt') # _,h
    # res, _ = torch.load('../data/nem_ss/nem_res/res_shat_hhat_snrinf.pt') # s,_ EM
    # _, res = torch.load('../data/nem_ss/nem_res/res_shat_hhat_snr20.pt') # _,h

    # plt.figure()
    # plt.plot(range(1, 101), torch.tensor(res).mean(dim=1))
    # plt.boxplot(res, showfliers=True)        
    # plt.legend(['Mean is blue'])
    # plt.ylim([0.5, 1])
    # plt.xticks([1, 20, 40, 60, 80, 100], [1, 20, 40, 60, 80, 100])
    # plt.xlabel('Sample index')
    # plt.title('NEM correlation result for h')
    # plt.show()

    ss = []
    for i in [0, 5, 10, 20, 25, 30, 40, 'inf']:
        _, res = torch.load(f'../data/nem_ss/nem_res/res_nem_shat_hhatsnr_{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 25, 30, 40, 'inf'], ss, '-x')

    ss = []
    for i in [0, 5, 10, 20, 25, 30, 40, 'inf']:
        _, res = torch.load(f'../data/nem_ss/nem_res/res_shat_hhat_snr{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 25, 30, 40, 'inf'], ss, '-o')
    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['NEM', 'EM'])
    plt.title('Correlation result for h')

    
#%%
