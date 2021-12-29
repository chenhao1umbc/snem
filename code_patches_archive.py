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

#%% toy data generation and run EM 
    v = torch.Tensor(sio.loadmat('../data/nem_ss/v.mat')['v'])
    N,F,J = v.shape
    M = 6
    max_iter = 200
    rseed = 1
    nvar = 1e-6

    torch.manual_seed(1)
    theta = torch.tensor([15, 75, -75])*np.pi/180  #len=J, signal AOAs  
    h = ((-1j*np.pi*torch.arange(0, M))[:,None]@ torch.sin(theta).to(torch.complex128)[None, :]).exp()  # shape of [M, J]
    s = torch.zeros((N,F,J), dtype=torch.complex128)
    c = torch.zeros((M,N,F,J), dtype=torch.complex128)

    for j in range(1, J):
        s[:,:,j] = (torch.randn(N,F)+1j*torch.randn(N,F))/2**0.5*v[:,:,j]**0.5
        for m in range(1,M):
            c[m,:,:,j] = h[m,j]*s[:,:,j]
    x = c.sum(3) + (torch.randn(M,N,F)+1j*torch.randn(M,N,F))/2**0.5*nvar**0.5
    x = x.permute(1,2,0)/x.abs().max()
    shat, Hhat, vhat, Rb = em_func(x, J=6, show_plot=True)

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
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
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
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')

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
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
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

#%% prepare the 6-class real data
    "raw data processing"
    FT = 100
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        # dd = np.abs(temp['x']).max(axis=1).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    M, J = 6, 6 # M is number of Channel, J is the number of sources
    np.set_printoptions(linewidth=150)
    "To generate 5000 mixture samples"
    theta = np.array([15, 45, 75, -15, -45, -75])*np.pi/180  #len=J, signal AOAs  
    h = np.exp(-1j*np.pi*np.arange(0, M)[:,None]@np.sin(theta)[None, :])  # shape of [M, J]

    np.random.seed(0)
    d1 = 0
    for i in range(J):
        np.random.shuffle(data[i])
    for i in range(J):
        d1 = d1 + h[:,i][:,None]@data[i][:,None,:] 

    np.random.seed(1)
    d2 = 0
    for i in range(J):
        np.random.shuffle(data[i])
    for i in range(J):
        d2 = d2 + h[:,i][:,None]@data[i][:,None,:] 

    data_pool = np.concatenate((d1, d2), axis=0)  #[I,M,time_len]
    *_, Z = stft(data_pool, fs=4e7, nperseg=FT, boundary=None)
    x = torch.tensor(np.roll(Z, FT//2, axis=2))  # roll nperseg//2
    plt.figure()
    plt.imshow(x[0,0].abs().log(), aspect='auto', interpolation='None')
    plt.title(f'One example of {J}-component mixture')
    torch.save(x[:3000], f'tr3kM{M}FT{FT}.pt')

    "get s and h for the val and test data"
    s = {}
    for i in range(J):
        *_, Z = stft(data[i][1000:1500], fs=4e7, nperseg=FT, boundary=None)
        s[i] = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    s = torch.tensor(np.stack((s[i] for i in range(J)), axis=1))  #[I, J, F, T]
    torch.save((x[3000:3500], s, h), f'val500M{M}FT{FT}_xsh.pt')

    s = {}
    for i in range(J):
        *_, Z = stft(data[i][1500:], fs=4e7, nperseg=FT, boundary=None)
        s[i] = np.roll(Z, FT//2, axis=1)  # roll nperseg//2
    s = torch.tensor(np.stack((s[i] for i in range(J)), axis=1))  #[I, J, F, T]
    torch.save((x[3500:], s, h), f'test500M{M}FT{FT}_xsh.pt')

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
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
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
            if torch.isnan(torch.tensor(ll_traj[-1])) : inp('nan happened')
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
    "This code shows EM is boosted by a little bit noise"
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
    for i in [0, 5, 10, 20, 'inf']:
        res, _ = torch.load(f'../data/nem_ss/nem_res/res_em_sh_init10db_snr{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-x')

    ss = []
    for i in [0, 5, 10, 20, 'inf']:
        res, _ = torch.load(f'../data/nem_ss/nem_res/EM_NEM_snr/res_nem_shat_hhatsnr_{i}.pt') # s, h NEM
        s = 0
        for i in range(100):
            for ii in range(20):
                s = s + res[i][ii]
        print(s/2000)
        ss.append(s/2000)
    plt.plot([0, 5, 10, 20, 'inf'], ss, '-o')


    plt.ylabel('Averaged correlation result')
    plt.xlabel('SNR')
    plt.legend(['EM', 'NEM'])
    plt.title('Correlation result for s')

#%% 10db result as the initial to do the EM
    import itertools, time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    h = torch.tensor(h)
    J = h.shape[-1]
    ratio = d.abs().amax(dim=(1,2,3))/3
    x = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1)

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[..., torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
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
                nume = (temp[:, j].conj() * h[:, j]).sum().abs()
                s = s + nume/dino
            r.append(s/J)
        r = sorted(r, reverse=True)
        return r[0].item()

    def em_func_mod(x, J=3, Hscale=1, Rbscale=100, max_iter=501, v_init=False, h_init=False, \
        lamb=0, seed=0, show_plot=False):
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

        N, F, M = x.shape
        NF= N*F
        torch.torch.manual_seed(seed)
        if v_init is False: 
            vhat = torch.randn(N, F, J).abs().to(torch.cdouble) 
        else :
            vhat = v_init
        if h_init is False: 
            Hhat = torch.randn(M, J, dtype=torch.cdouble)*Hscale
        else:
            Hhat = h_init
        Rb = torch.eye(M).to(torch.cdouble)*Rbscale
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
            if lamb<0:
                raise ValueError('lambda should be not negative')
            elif lamb >0:
                y = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                vhat = -0.5/lamb + 0.5*(1+4*lamb*y)**0.5/lamb
            else:
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
            if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
                print(f'EM early stop at iter {i}')
                break

        if show_plot:
            plt.figure(100)
            plt.plot(ll_traj,'o-')
            plt.show()
            "display results"
            for j in range(J):
                plt.figure()
                plt.imshow(vhat[:,:,j].real)
                plt.colorbar()

        return shat, Hhat, vhat, Rb

    # samples = 100
    # seeds = 20
    # hh_all = torch.rand(samples, seeds, J, J, dtype=torch.cdouble)
    # vh_all = torch.rand(samples, seeds, 100, 100, J, dtype=torch.cdouble)
    # for i in range(samples):
    #     for ii in range(seeds):
    #         shat, hh_all[i, ii], vh_all[i, ii], Rb = em_func(awgn(x[i], snr=10), seed=ii)
    #     print(f'finished {i} sample')
    # print(f'used {time.time()-t}s')
    # torch.save([hh_all, vh_all], 'hh_all-vh_all_resforinit.py')
    hh_all, vh_all = torch.load('hh_all-vh_all_resforinit.py')

    res, res2 = [], []
    for i in range(100):
        c, cc = [], []
        for ii in range(10):
            shat, Hhat, vhat, Rb = em_func_mod(awgn(x[i], snr=20), J=6,\
                v_init=vh_all[i,ii], h_init=hh_all[i,ii], seed=ii)
            c.append(corr(shat.squeeze().abs(), s_all[i]))
            cc.append(h_corr(h, Hhat))
        res.append(c)
        res2.append(cc)
        print(f'finished {i} samples')
    # torch.save([res, res2], 'res_em_shat_hhat_snr20.pt')

    s = 0 
    for i in range(100):
        for ii in range(20):
            s = s + res[i][ii]
    print(s/2000)

#%% Test 1 channel 1 model NEM
    "best ones rid 135100/125240/135110"
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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')

        model = torch.load(model)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        graw = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
        graw = torch.stack([graw[:,None] for j in range(J)], dim=1)  # shape of [1,J,8,8]
        noise = torch.rand(J,1,8,8)
        for j in range(J):
            noise[j,0] = awgn(graw[0,j,0], snr=10, seed=j) - graw[0,j,0]
        g = (graw + noise).cuda().requires_grad_()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        out = torch.randn(vhat.shape, device='cuda', dtype=torch.double)
        for j in range(J):
            out[..., j] = torch.sigmoid(model(g[:,j]).squeeze())
        vhat.real = threshold(out)
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
                out[..., j] = torch.sigmoid(model(g[:,j]).squeeze())
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
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
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break

        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()
    
    location = f'../data/nem_ss/models/model_rid135110.pt'
    single_data = False
    if single_data:
        ind = 70
        shat, Hhat, vhat, Rb = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
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
        torch.save([res, res2], 'res_nem_shat_hhat_rid135110.pt')

#%% Test 2 channel model 1 model NEM
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')

        model = torch.load(model)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        graw = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([graw[:,None] for j in range(J)], dim=1).cuda()  # shape of [1,J,8,8]
        lb = torch.load('../data/nem_ss/140100_lb.pt')
        lb = lb[None,...].cuda()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], lb[:,j]), dim=1))))
        out = torch.cat(outs, dim=1).permute(0,2,3,1)
        vhat.real = threshold(out)
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g.requires_grad_()
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
            outs = []
            for j in range(J):
                outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
            out = torch.cat(outs, dim=1).permute(0,2,3,1)
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
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
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break

        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu(), ll_traj

    rid = 149000
    location = f'../data/nem_ss/models/rid{rid}/model_rid{rid}_38.pt'
    single_data = False
    if single_data:
        ind = 43
        shat, Hhat, vhat, Rb, loss = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
        for i in range(3):
            plt.figure()
            plt.imshow(s_all[ind].squeeze().abs()[...,i])
            plt.colorbar()
            plt.title(f'GT sources {i+1}')
            plt.show()

        plt.figure()
        plt.plot(loss, '-x')
        plt.title('loss value')
        plt.show()
    else: # run a lot of samples
        res, res2 = [], []
        for i in range(100):
            c, cc = [], []
            for ii in range(10):
                shat, Hhat, *_ = nem_func(awgn(x_all[i], snr=20),seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], f'res_10seed_rid{rid}_snr20.pt')

#%% Test 2 channel model 1 model NEM, with mini batch
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from unet.unet_model import UNetHalf8to100_256_sig as UNetHalf
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 150, 3, 64 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)
    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    # g = torch.load('../data/nem_ss/gtest_500.pt') # preload g only works for no noise case
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    
    l = torch.load('../data/nem_ss/140100_lb.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[...,torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
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

    def nem_minibatch_test(data, ginit, model, lb, bs, seed=1):
        torch.manual_seed(seed) 
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        EM_iters = 501
        M, N, F, J = 3, 100, 100, 3
        NF, I = N*F, ginit.shape[0]

        vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
        Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
        Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

        lv, s, h, v, ll_all = ([] for i in range(5)) 
        for i, (x,) in enumerate(data): # gamma [n_batch, 4, 4]
            #%% EM part
            vhat = vtr[i*bs:(i+1)*bs].cuda()        
            Rb = Rbtr[i*bs:(i+1)*bs].cuda()
            g = ginit[i*bs:(i+1)*bs].cuda().requires_grad_()

            x = x.cuda()
            optim_gamma = torch.optim.SGD([g], lr=0.001)
            Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
            ll_traj = []

            for ii in range(EM_iters):
                "E-step"
                W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
                shat = W.permute(2,0,1,3,4) @ x[...,None]
                Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
                Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
                Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

                "M-step"
                Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
                # Hhat = (Rxshat @ Rsshat.inverse()).mean(0) # shape of [M, J]
                Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                    Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
                Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
                Rb.imag = Rb.imag - Rb.imag

                # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                # vj.imag = vj.imag - vj.imag
                outs = []
                for j in range(J):
                    outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
                out = torch.cat(outs, dim=1).permute(0,2,3,1)
                vhat.real = threshold(out)
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_([g], max_norm=1)
                optim_gamma.step()
                torch.cuda.empty_cache()
                
                "compute log-likelyhood"
                vhat = vhat.detach()
                Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
                Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
                Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
                l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
                ll = l.sum().real
                Rx = Rxperm

                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                    print(f'EM early stop at iter {ii}')
                    break
            ll_all.append(l.sum((-1, -2)).cpu().real)
            lv.append(ll.item())
            s.append(shat)
            h.append(Hhat)
            v.append(vhat)
            print(f'batch {i} is done')
        return sum(lv)/len(lv), torch.cat(ll_all), (s, h, v)

    rid = 150000
    model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_35.pt')
    meanl, l_all, shv = nem_minibatch_test(data_test, gte, model, lb, bs, seed=1)
    print('End date time ', datetime.now())

    shat, hhat, vhat = shv
    shat_all, hhat_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu()
    res_s, res_h = [], []
    for i in range(100):
        res_s.append(corr(shat_all[i].squeeze().abs(), s_all[i]))
        res_h.append(h_corr(h, hhat_all[i].squeeze()))
    print(sum(res_s)/len(res_s))
    print(sum(res_h)/len(res_h))

#%% Test 2 channel model 1 model NEM, with mini batch for 6 classes
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 130, 6, 32 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)

    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    l = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    def corr(vh, v):
        J = v.shape[-1]
        r = [] 
        permutes = list(itertools.permutations(list(range(J))))
        for p in permutes:
            temp = vh[:,:,torch.tensor(p)]
            s = 0
            for j in range(J):
                s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[...,j].flatten())[0])
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

    def nem_minibatch_test(data, ginit, model, lb, bs, seed=1):
        torch.manual_seed(seed) 
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        EM_iters = 501
        M, N, F, J = 6, 100, 100, 6
        NF, I = N*F, ginit.shape[0]

        vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
        Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
        Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

        lv, s, h, v, ll_all = ([] for i in range(5)) 
        for i, (x,) in enumerate(data): # gamma [n_batch, 4, 4]
            #%% EM part
            vhat = vtr[i*bs:(i+1)*bs].cuda()        
            Rb = Rbtr[i*bs:(i+1)*bs].cuda()
            g = ginit[i*bs:(i+1)*bs].cuda().requires_grad_()

            x = x.cuda()
            optim_gamma = torch.optim.SGD([g], lr=0.001)
            Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
            ll_traj = []

            for ii in range(EM_iters):
                "E-step"
                W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
                shat = W.permute(2,0,1,3,4) @ x[...,None]
                Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
                Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
                Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

                "M-step"
                Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
                # Hhat = (Rxshat @ Rsshat.inverse()).mean(0) # shape of [M, J]
                Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                    Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
                Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
                Rb.imag = Rb.imag - Rb.imag

                # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                # vj.imag = vj.imag - vj.imag
                outs = []
                for j in range(J):
                    outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
                out = torch.cat(outs, dim=1).permute(0,2,3,1)
                vhat.real = threshold(out)
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_([g], max_norm=1)
                optim_gamma.step()
                torch.cuda.empty_cache()
                
                "compute log-likelyhood"
                vhat = vhat.detach()
                Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
                Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
                Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
                l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
                ll = l.sum().real
                Rx = Rxperm

                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                    print(f'EM early stop at iter {ii}')
                    break
            ll_all.append(l.sum((-1, -2)).cpu().real)
            lv.append(ll.item())
            s.append(shat)
            h.append(Hhat)
            v.append(vhat)
            print(f'batch {i} is done')
        return sum(lv)/len(lv), torch.cat(ll_all), (s, h, v)

    rid = 160001
    model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_41.pt')
    meanl, l_all, shv = nem_minibatch_test(data_test, gte, model, lb, bs, seed=1)
    print('End date time ', datetime.now())

    shat, hhat, vhat = shv
    shat_all, hhat_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu()
    res_s, res_h = [], []
    for i in range(100):
        res_s.append(corr(shat_all[i].squeeze().abs(), s_all[i]))
        res_h.append(h_corr(h, hhat_all[i].squeeze()))
        print(f'{i}-th sample is done')
    print(sum(res_s)/len(res_s))
    print(sum(res_h)/len(res_h))

#%% Test 2 channel model 1 model NEM, gamma=noise as label -- abandoned
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
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

    def nem_func(x, J=3, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')

        model = torch.load(model)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        xx = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        xx = (xx/xx.max())[None,None,...].cuda()  #standardization shape of [1, 1, 8, 8]
        g = torch.load('../data/nem_ss/g_init.pt')  # shape of [J, 1, 8, 8]
        g = g[None,...].cuda() # do not put requires_grad_ here
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], xx), dim=1))))
        out = torch.cat(outs, dim=1).permute(0,2,3,1)
        vhat.real = threshold(out)
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g.requires_grad_()
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
            outs = []
            for j in range(J):
                outs.append(torch.sigmoid(model(torch.cat((g[:,j], xx), dim=1))))
            out = torch.cat(outs, dim=1).permute(0,2,3,1)
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
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
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break
        return shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze(), Rb.cpu()

    location = f'../data/nem_ss/models/model_rid141101.pt'
    # location = f'../data/nem_ss/models/rid141103/model_rid141103_58.pt'
    single_data = True
    if single_data:
        ind = 43
        shat, Hhat, vhat, Rb = nem_func(x_all[ind],seed=10,model=location)
        for i in range(3):
            plt.figure()
            plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
            plt.colorbar()
            plt.title(f'Estimated sources {i+1}')
            plt.show()
        print('h correlation:', h_corr(h, Hhat.squeeze()))
        print('s correlation:', corr(shat.squeeze().abs(), s_all[ind]))
        
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
            for ii in range(10):
                shat, Hhat, vhat, Rb = nem_func(awgn(x_all[i], snr=20),seed=ii,model=location)
                c.append(corr(shat.squeeze().abs(), s_all[i]))
                cc.append(h_corr(h, Hhat.squeeze()))
            res.append(c)
            res2.append(cc)
            print(f'finished {i} samples')
        print('Time used is ', time.time()-t)
        torch.save([res, res2], 'res_10seed_rid141103_58_snr20.pt')

#%% check loss function values
    l = torch.load('../data/nem_ss/models/rid141103/loss_rid141103.pt')
    l = torch.tensor(l)
    n = 3
    c = []
    for epoch in range(len(l)):
        if epoch > 10 :
            ll = l[:epoch]
            s1, s2 = sum(ll[epoch-2*n:epoch-n])/n, sum(ll[epoch-n:])/n
            c.append( abs((s1-s2)/s1))
            print(f'current epcoch-{epoch}: ', abs((s1-s2)/s1), s1, s2)
            if s1 - s2 < 0 :
                print('break-1')
                break
            if abs((s1-s2)/s1) < 5e-4 :
                print(epoch)
                print('break-2')
                break
    plt.plot(c, '-x')

#%% validation value check
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from unet.unet_model import UNetHalf8to100_256_sig as UNetHalf
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 130, 6, 32 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/val500M6FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_test = Data.DataLoader(data, batch_size=bs, drop_last=True)

    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    l = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    rid = 160001
    ll_all = []
    for i in range(42):
        model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_{i}.pt')
        ll = val_run(data_test, gte, model, lb, [6,6,32], seed=1)
        ll_all.append(ll)
        print(ll)
        plt.figure()
        plt.title(f'validation likelihood till epoch {i}')
        plt.plot(ll_all, '-or')
        plt.savefig(f'id{rid} validation likelihood')
        plt.close()
    print('End date time ', datetime.now())

#%% generate g from validation data for classification
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 100
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    from datetime import datetime
    print('starting date time ', datetime.now())
    torch.manual_seed(1)

    I, J, bs = 130, 6, 32 # I should be larger than bs
    d, s, h = torch.load('../data/nem_ss/val500M6FT100_xsh.pt')
    s_all, h = s.abs().permute(0,2,3,1), torch.tensor(h)
    ratio = d.abs().amax(dim=(1,2,3))/3
    xte = (d/d.abs().amax(dim=(1,2,3))[:,None,None,None]*3).permute(0,2,3,1)# [sample, N, F, channel]
    xte = awgn_batch(xte[:I], snr=1000)
    data = Data.TensorDataset(xte)
    data_val = Data.DataLoader(data, batch_size=bs, drop_last=True)

    from skimage.transform import resize
    gte = torch.tensor(resize(xte[...,0].abs(), [I,8,8], order=1, preserve_range=True ))
    gte = gte[:I]/gte[:I].amax(dim=[1,2])[...,None,None]  #standardization 
    gte = torch.cat([gte[:,None] for j in range(J)], dim=1)[:,:,None] # shape of [I_val,J,1,8,8]
    l = torch.load('../data/nem_ss/lb_c6_J188.pt')
    lb = l.repeat(bs, 1, 1, 1, 1).cuda()

    def nem_val_gtr(data, ginit, model, lb, bs, seed=1):
        torch.manual_seed(seed) 
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        EM_iters = 501
        M, N, F, J = 6, 100, 100, 6
        NF, I = N*F, ginit.shape[0]

        vtr = torch.randn(N, F, J).abs().to(torch.cdouble).repeat(I, 1, 1, 1)
        Hhat = torch.randn(M, J).to(torch.cdouble).cuda()
        Rbtr = torch.ones(I, M).diag_embed().to(torch.cdouble)*100

        lv, s, h, v, ll_all, g_all = ([] for i in range(6)) 
        for i, (x,) in enumerate(data): # gamma [n_batch, 4, 4]
            #%% EM part
            vhat = vtr[i*bs:(i+1)*bs].cuda()        
            Rb = Rbtr[i*bs:(i+1)*bs].cuda()
            g = ginit[i*bs:(i+1)*bs].cuda().requires_grad_()

            x = x.cuda()
            optim_gamma = torch.optim.SGD([g], lr=0.001)
            Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
            ll_traj = []

            for ii in range(EM_iters):
                "E-step"
                W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
                shat = W.permute(2,0,1,3,4) @ x[...,None]
                Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)
                Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
                Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

                "M-step"
                Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
                # Hhat = (Rxshat @ Rsshat.inverse()).mean(0) # shape of [M, J]
                Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                    Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
                Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
                Rb.imag = Rb.imag - Rb.imag

                # vj = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                # vj.imag = vj.imag - vj.imag
                outs = []
                for j in range(J):
                    outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
                out = torch.cat(outs, dim=1).permute(0,2,3,1)
                vhat.real = threshold(out)
                loss = loss_func(vhat, Rsshatnf.cuda())
                optim_gamma.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_([g], max_norm=1)
                optim_gamma.step()
                torch.cuda.empty_cache()
                
                "compute log-likelyhood"
                vhat = vhat.detach()
                Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
                Rxperm = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb 
                Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
                l = -(np.pi*mydet(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze()
                ll = l.sum().real
                Rx = Rxperm

                ll_traj.append(ll.item())
                if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
                if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                    print(f'EM early stop at iter {ii}')
                    break
            ll_all.append(l.sum((-1, -2)).cpu().real)
            lv.append(ll.item())
            s.append(shat)
            h.append(Hhat)
            v.append(vhat)
            g_all.append(g.detach())
            print(f'batch {i} is done')
        return g_all, torch.cat(ll_all), (s, h, v)

    rid = 160001
    model = torch.load(f'../data/nem_ss/models/rid{rid}/model_rid{rid}_41.pt')
    gval, l_all, shv = nem_val_gtr(data_val, gte, model, lb, bs, seed=1)
    print('End date time ', datetime.now())

    shat, hhat, vhat = shv
    shat_all, hhat_all, g_all = torch.cat(shat).cpu(), torch.cat(hhat).cpu(), torch.cat(gval).cpu()
    torch.save(g_all , 'g_all.pt')
    print('done with g_all')

#%% NEM test 3-mixture from 6 mixture model
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M= torch.tensor(h), s.shape[-1], s.shape[-2], d.shape[1]
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def nem_func_less(x, J=6, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')

        model = torch.load(model)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        graw = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([graw[:,None] for j in range(J)], dim=1).cuda()  # shape of [1,J,8,8]
        lb = torch.load('../data/nem_ss/lb_c6_J188.pt')
        lb = lb[None,...].cuda()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], lb[:,j]), dim=1))))
        out = torch.cat(outs, dim=1).permute(0,2,3,1)
        vhat.real = threshold(out)
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g.requires_grad_()
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
            outs = []
            for j in range(J):
                outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
            out = torch.cat(outs, dim=1).permute(0,2,3,1)
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
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
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break

        return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), g.detach().cpu(), Rb.cpu(), ll_traj

    rid = 160100
    location = f'model_rid{rid}_33.pt'
    ind = 10 # which sample to test
    J, which_class = 6, [0, 2, 5]  # J is NO. of class you guess, which_class is really there

    "prep data"
    for i, v in enumerate(which_class):
        if i == 0 : d = 0
        d = d + h[:, v, None] @ s[ind, v].reshape(1, N*F)
    d = d.reshape(M, N, F).permute(1,2,0)
    
    shv, g, Rb, loss = nem_func_less(d, J=J, seed=10, model=location)
    shat, Hhat, vhat = shv
    for i in range(6):
        plt.figure()
        plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
        plt.colorbar()
        plt.title(f'Estimated sources {i+1}')
        plt.show()

    for i, v in enumerate(which_class):
        plt.figure()
        plt.imshow(s[ind, v].abs())
        plt.colorbar()
        plt.title(f'GT sources {i+1}')
        plt.show()

    plt.figure()
    plt.plot(loss, '-x')
    plt.title('loss value')
    plt.show()

    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(g[0,i,0].abs())
        plt.colorbar()
        plt.tight_layout(pad=1.2)
        plt.title(f'gamma of source {i+1}',y=1.2)

    graw = torch.tensor(resize(d[...,0].abs(), [8,8], order=1, preserve_range=True))
    graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(g[0,i,0].abs() - graw[0].abs())
        plt.colorbar(fraction=0.046)
        plt.tight_layout(pad=1.7)    
        # plt.title(f'gamma diff of source {i+1}',y=1.2)

#%% EM for M=6 > real J, to get correct J
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    data, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6

    def em_func_(x, J=3, Hscale=1, Rbscale=100, max_iter=501, lamb=0, seed=0, show_plot=False):
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

        N, F, M = x.shape
        NF= N*F
        torch.torch.manual_seed(seed)
        vhat = torch.randn(N, F, J).abs().to(torch.cdouble)
        Hhat = torch.randn(M, J, dtype=torch.cdouble)*Hscale
        Rb = torch.eye(M).to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rj = torch.zeros(J, M, M).to(torch.cdouble)
        ll_traj = []

        for i in range(max_iter):
            "E-step"
            Rs = vhat.diag_embed()
            Rcj = Hhat @ Rs @ Hhat.t().conj()
            Rx = Rcj + Rb
            W = Rs @ Hhat.t().conj() @ Rx.inverse()
            shat = W @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - W@Hhat@Rs

            Rsshat = Rsshatnf.sum([0,1])/NF
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((0,1))/NF

            "M-step"
            if lamb<0:
                raise ValueError('lambda should be not negative')
            elif lamb >0:
                y = Rsshatnf.diagonal(dim1=-1, dim2=-2)
                vhat = -0.5/lamb + 0.5*(1+4*lamb*y)**0.5/lamb
            else:
                vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            vhat.imag = vhat.imag - vhat.imag
            Hhat = Rxshat @ Rsshat.inverse()
            Rb = Rxxhat - Hhat@Rxshat.t().conj() - \
                Rxshat@Hhat.t().conj() + Hhat@Rsshat@Hhat.t().conj()
            Rb = threshold(Rb.diag().real, floor=1e-20).diag().to(torch.cdouble)
            # Rb = Rb.diag().real.diag().to(torch.cdouble)

            "compute log-likelyhood"
            for j in range(J):
                Rj[j] = Hhat[:, j][..., None] @ Hhat[:, j][..., None].t().conj()
            ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
            if i > 30 and abs((ll_traj[i] - ll_traj[i-3])/ll_traj[i-3]) <1e-4:
                print(f'EM early stop at iter {i}')
                break

        return shat, Hhat, vhat, Rb, ll_traj, torch.linalg.matrix_rank(Rcj).double().mean()

    t = time.time()
    res = []
    for JJ in range(1, 7):
        r = []
        # ind = 0 # which sample to test
        # J, which_class = 6, [0,2,5]  # J is NO. of class you guess, which_class is really there
        comb = list(itertools.combinations(range(6), JJ))
        for which_class in comb:
            for ind in range(100): 
                for i, v in enumerate(which_class):
                    if i == 0 : d = 0
                    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
                d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

                shat, Hhat, vhat, Rb, ll_traj, rank = em_func_(d, J=6, max_iter=10)
                r.append(rank)
            print('one comb is done', which_class)
        res.append(r)
        torch.save(res, 'res.pt')
    print('done', time.time()-t)

#%% NME M=6 > real J, get hhat
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    from skimage.transform import resize
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6
    ratio = d.abs().amax(dim=(1,2,3))/3
    x_all = (d/ratio[:,None,None,None]).permute(0,2,3,1)
    s_all = s.abs().permute(0,2,3,1) 

    def nem_func_less(x, J=6, Hscale=1, Rbscale=100, max_iter=501, seed=1, model=''):
        def log_likelihood(x, vhat, Hhat, Rb, ):
            """ Hhat shape of [I, M, J] # I is NO. of samples, M is NO. of antennas, J is NO. of sources
                vhat shape of [I, N, F, J]
                Rb shape of [I, M, M]
                x shape of [I, N, F, M]
            """
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rcj = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj()
            Rxperm = Rcj + Rb 
            Rx = Rxperm.permute(2,0,1,3,4) # shape of [I, N, F, M, M]
            l = -(np.pi*torch.linalg.det(Rx)).log() - (x[...,None,:].conj()@Rx.inverse()@x[...,None]).squeeze() 
            return l.sum().real, Rs, Rxperm, Rcj

        torch.manual_seed(seed) 
        if model == '':
            print('A model is needed')

        model = torch.load(model)
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        #%% EM part
        "initial"        
        N, F, M = x.shape
        NF= N*F
        graw = torch.tensor(resize(x[...,0].abs(), [8,8], order=1, preserve_range=True))
        graw = (graw/graw.max())[None,...]  #standardization shape of [1, 8, 8]
        g = torch.stack([graw[:,None] for j in range(J)], dim=1).cuda()  # shape of [1,J,8,8]
        lb = torch.load('../data/nem_ss/lb_c6_J188.pt')
        lb = lb[None,...].cuda()
        x = x.cuda()

        vhat = torch.randn(1, N, F, J).abs().to(torch.cdouble).cuda()
        outs = []
        for j in range(J):
            outs.append(torch.sigmoid(model(torch.cat((g[:,j], lb[:,j]), dim=1))))
        out = torch.cat(outs, dim=1).permute(0,2,3,1)
        vhat.real = threshold(out)
        Hhat = torch.randn(1, M, J).to(torch.cdouble).cuda()*Hscale
        Rb = torch.ones(1, M).diag_embed().cuda().to(torch.cdouble)*Rbscale
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((0,1))/NF
        Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
        Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [N,F,I,M,M]
        g.requires_grad_()
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
            outs = []
            for j in range(J):
                outs.append(model(torch.cat((g[:,j], lb[:,j]), dim=1)))
            out = torch.cat(outs, dim=1).permute(0,2,3,1)
            vhat.real = threshold(out)
            loss = loss_func(vhat, Rsshatnf.cuda())
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=1)
            optim_gamma.step()
            torch.cuda.empty_cache()
            
            "compute log-likelyhood"
            vhat = vhat.detach()
            ll, Rs, Rx, Rcj = log_likelihood(x, vhat, Hhat, Rb)
            ll_traj.append(ll.item())
            if torch.isnan(torch.tensor(ll_traj[-1])) : input('nan happened')
            if ii > 20 and abs((ll_traj[ii] - ll_traj[ii-3])/ll_traj[ii-3]) <5e-4:
                print(f'EM early stop at iter {ii}')
                break

        return (shat.cpu(), Hhat.cpu(), vhat.cpu().squeeze()), g.detach().cpu(), Rb.cpu(), ll_traj

    rid = 160100
    model = f'model_rid{rid}_33.pt'
    t = time.time()
    res = []
    for JJ in range(2, 7):
        r = []
        comb = list(itertools.combinations(range(6), JJ))
        for which_class in comb:
            for ind in range(100): 
                for i, v in enumerate(which_class):
                    if i == 0 : d = 0
                    d = d + h[:M, v, None] @ s[ind, v].reshape(1, N*F)
                d = d.reshape(M, N, F).permute(1,2,0)/d.abs().max()

                shv, g, Rb, loss = nem_func_less(d, J=6, seed=10, model=model, max_iter=301)
                shat, Hhat, vhat = shv
                r.append(Hhat)
            print('one comb is done', which_class)
        res.append(r)
        torch.save(res, 'Hhat_2-6comb_res.pt')
    print('done', time.time()-t)

#%% Use H to do classification, if ground truth h is available
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    import time
    t = time.time()
    d, s, h = torch.load('../data/nem_ss/test500M6FT100_xsh.pt')
    h, N, F, M = torch.tensor(h), s.shape[-1], s.shape[-2], 6

    def hcorr(hi, hh):
        r = []
        for i in range(6):
            n = hi @ hh[:,i].conj()
            d = hi.norm() * hh[:,i].norm()
            r.append(n.abs()/d)
        return max(r)

    n_comb = 2
    comb = list(itertools.combinations(range(6), n_comb))
    lb = torch.tensor(comb)
    lbs = lb.unsqueeze(1).repeat(1,100,1).reshape(lb.shape[0]*100, n_comb)
    hall = torch.load(f'../data/nem_ss/nem_res/Hhat_{n_comb}comb_res.pt')
    acc = 0
    for i in range(len(hall)):
        hh = hall[i].squeeze()
        res = []
        for wc in range(6):
            res.append(hcorr(h[:,wc], hh))
        _, lb_hat = torch.topk(torch.tensor(res), n_comb)

        for ii in range(n_comb):
            if lb_hat[ii] in lbs[i]:
                acc += 1
    print(acc/len(hall)/n_comb)
    "if ground truth h is konwn, 2~0.97, 3~0.92983, 4~0.937833, 5~0.95766, 6~1.00 and 1~1.00 of coursce"

#%% Use H to do classification, using training h 
    from utils import *
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    plt.rcParams['figure.dpi'] = 150
    torch.set_printoptions(linewidth=160)
    torch.set_default_dtype(torch.double)
    import itertools
    import time
    t = time.time()
    s_all, h_all, v_all= torch.load('../data/nem_ss/shv_tr3k_M6_rid160100.pt')
    s0, h0 = s_all[0].squeeze(), h_all[0]
    "In s[0] from 0~31: 3,4,8,13,14,15,29,31 are class4 then class5, others are 5,4"
    "also, 5,6,7,8, 11, 12,23, 26, 30 really bad"
    # idx1 = torch.tensor([0,5,2,1,4,3]) # for others -- 0,3,2,5,4,1
    # idx2 = torch.tensor([0,5,2,1,3,4]) # for 3,4,8,13,14,15,29,31 -- 0,3,2,4,5,1

    def get_lb(h0, hh):
        res = torch.zeros(30,6,6)
        for i in range(3):
            if i in [3,4,8,13,14,15,29,31]:
                idx = torch.tensor([0,5,2,1,3,4])
            else:
                idx = torch.tensor([0,5,2,1,4,3])
            h = h0[0][:, idx]
            for d1 in range(6):
                hi = h[:, d1]
                for d2 in range(6):
                    n = hi@ hh[:,d2].conj()
                    d = hi.norm() * hh[:,d2].norm()
                    res[i, d1, d2] = n.abs()/d
        return res.sum(0).amax(dim=1)

    n_comb = 2
    comb = list(itertools.combinations(range(6), n_comb))
    lb = torch.tensor(comb)
    lbs = lb.unsqueeze(1).repeat(1,100,1).reshape(lb.shape[0]*100, n_comb)
    hall = torch.load(f'../data/nem_ss/nem_res/Hhat_{n_comb}comb_res.pt')
    acc = 0
    for i in range(len(hall)):
        hh = hall[i].squeeze()
        res = get_lb(h0, hh)
        _, lb_hat = torch.topk(res, n_comb)

        for ii in range(n_comb):
            if lb_hat[ii] in lbs[i]:
                acc += 1
    print(acc/len(hall)/n_comb)

######################################################Weakly supervised#############################################################
#%% Weakly supervised label and data generation
    "raw data processing"
    FT = 100
    var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    np.set_printoptions(linewidth=150)
    "To generate 5000 mixture samples"
    M, I_comb = 6, 50# M is number of Channel, I_comb is how many samples per combination
    theta = np.array([15, 45, 75, -15, -45, -75])*np.pi/180  #len=J, signal AOAs  
    h = np.exp(-1j*np.pi*np.arange(0, M)[:,None]@np.sin(theta)[None, :])  # shape of [M, J]

    data_pool, lbs = [], []
    for J in range(2, 7):
        combs = list(itertools.combinations(range(6), J))
        for i, lb in enumerate(combs):
            np.random.seed(i+10)  # val i+5, te i+10, run from scratch
            d = 0
            for ii in range(J):
                np.random.shuffle(data[lb[ii]])
            for ii in range(J):
                d = d + h[:,lb[ii]][:,None]@data[lb[ii]][:I_comb,None,:] 
            data_pool.append(d)
            lbs.append(lb)
        print(J)
    data_all = np.concatenate(data_pool, axis=0)  #[I,M,time_len]
    *_, Z = stft(data_all, fs=4e7, nperseg=FT, boundary=None)
    x = torch.tensor(np.roll(Z, FT//2, axis=2))  # roll nperseg//2
    plt.figure()
    plt.imshow(x[0,0].abs().log(), aspect='auto', interpolation='None')
    plt.title(f'One example of {J}-component mixture')
    # torch.save((x,lbs), f'weakly50percomb_tr3kM{M}FT{FT}_xlb.pt')

    #"get s and h for the val and test data"
    import itertools
    "raw data processing"
    data = {}
    for i in range(6):
        temp = sio.loadmat('/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+f'_{FT}_2k.mat')
        dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
        data[i] = temp['x'] / dd  # normalized very sample to 1

    s_pool, lbs = [], []
    for J in range(2, 7):
        combs = list(itertools.combinations(range(6), J))
        for i, lb in enumerate(combs):
            np.random.seed(i+10)  # val i+5, te i+10, run from scratch
            for ii in range(J):
                np.random.shuffle(data[lb[ii]])
            for ii in range(J):
                d = data[lb[ii]][:I_comb,:] 
                s_pool.append(d.copy())
            lbs.append(lb)
        print(J)
    data_all = np.concatenate(s_pool, axis=0)  #[I,M,time_len]
    *_, Z = stft(data_all, fs=4e7, nperseg=FT, boundary=None)
    s = torch.tensor(np.roll(Z, FT//2, axis=1))  # roll nperseg//2
    plt.figure()
    plt.imshow(s[0].abs().log(), aspect='auto', interpolation='None')
    plt.title(f'One example of {J}-component mixture')
    # torch.save((x,s,lbs), f'weakly50percomb_te3kM{M}FT{FT}_xslb.pt')
    #%% check s 
    "x.shape is [2850, 6, 100, 100], which means 2850=57*50, 57=15+20+15+6+1"
    "s.shape is [9300, 100, 100], which means 9300=186*50, 186=15*2+20*3+15*4+6*6+1*6"
    J = 2
    combs = list(itertools.combinations(range(6), J))
    print(combs)
    xr = x.reshape(57,50,6,100,100)
    sr = s.reshape(186,50,100,100)

    wc = 2
    ind = 10
    plt.figure()
    plt.imshow(xr[wc,ind,0].abs())

    plt.figure()
    if wc<15:
        for i in range(2):
            plt.figure()
            plt.imshow(sr[wc*2+i,ind].abs())
