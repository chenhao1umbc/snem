#%%
from utils import *

# %% Test data with power difference without EM
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
n_comb, n = 0, 0  # which example to test
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
db = 10  # db in [0, 20]
power_ratio = 10**(-1*db/20)
# x = sources[1, n]*power_ratio + sources[2, n] + sources[3, n]
x = mix[n_comb,n]  # mixture of 6 components without power diff.
plot_x(x, title='Input mixture')  # plot input

s_stft = torch.zeros(6, 200, 200)
fname = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
xte = st_ft(x).reshape(1,1,200,200).abs().log().float()
te_cuda = xte.cuda()
for i in range(6):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('../data/data_ss/'+fname[i]+'_unet20.pt'))
    model.eval()

    with torch.no_grad():
        s_stft[i] = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()
 
# #%% Single Channel =====================================
    # "EM to get each sources"
    # n_iter = 50
    # mse = []
    # var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']

    # which_source = torch.tensor([1,2])
    # x = sources[which_source, n].sum(0)
    # gt_stft = torch.rand(which_source.shape[0], 200, 200, dtype=torch.complex64)
    # for i in range(which_source.shape[0]):
    #     gt_stft[i] = st_ft(sources[i, n])

    # init = awgn(s_stft[which_source], snr=200) #  gt_stft.abs().log()
    # for ii in range(n_iter):
    #     # cjh, likelihood = em_simple(init_stft=init, stft_mix=st_ft(x), n_iter=ii)  # instead of import Norbert
    #     cjh, likelihood = em_10paper(init_stft=init, stft_mix=st_ft(x), n_iter=ii) 
    #     mse.append((((cjh - gt_stft).abs()**2).sum()).item())
    #     # for i in range(which_source.shape[0]):
    #     #     plot_x(cjh[i], title=var_name[which_source[i]])
    # plt.figure()
    # plt.plot(mse, '-x')

#%% Multi-Channel
"EM to get each sources"
n_iter = 100
n_c = 2  # 2 channels
mse = []
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']

which_source = torch.tensor([3,4])
x = sources[which_source, n].sum(0)
gt_stft = torch.rand(which_source.shape[0],200, 200, n_c, dtype=torch.complex64)
for i in range(which_source.shape[0]):
    s = sources[which_source[i], n]
    gt_stft[i, ... , 0] = st_ft(s)
    gt_stft[i, ... , 1] = st_ft(s*e**(1j*pi/12*(i+1))) # awgn(st_ft(s), snr=20)

init = awgn(s_stft[which_source], snr=10) #  gt_stft.abs().log()
# init = torch.rand(2, 200, 200) -9.8
cjh_list, likelihood = em10(init_stft=init, stft_mix=gt_stft.sum(0), n_iter=n_iter) 

for i in [0,1,30,50,90]:
    for ii in range(which_source.shape[0]):
        plot_x(cjh_list[i][ii,...,0], title=f'{var_name[which_source[ii]]} iter {i}')

for i in range(n_iter+1):
    mse.append((((cjh_list[i] - gt_stft).abs()**2).sum()).item())   
plt.figure()
plt.plot(mse, '-x')
plt.figure()
plt.plot(likelihood, '-x')

# # %% Norbert Multi-Channel
    # import norbert
    # mse = []
    # x = gt_stft.sum(0).permute(1,0,2).numpy()
    # y = torch.stack((init.permute(2,1,0), init.permute(2,1,0)), -1 ).numpy()
    # init = awgn(s_stft[which_source], snr=20)
    # for ii in range(20):
    #     yh, vh, rh = norbert.expectation_maximization(
    #         y=y, x=x, iterations=ii) 
    #     mse.append((((torch.tensor(yh) - gt_stft.permute(2,1, 3, 0)).abs()**2).sum()).item())
    # plt.figure()
    # plt.plot(mse, '-x')

# %%  how to do inverse STFT
a = np.random.rand(20100) +1j*np.random.rand(20100)
_, _, A = stft(a, fs=4e7, nperseg=200, boundary=None)
_, aa = istft(A, fs=4e7, nperseg=200, input_onesided=False, boundary=None )
plt.plot(aa.imag[:500]+1)
plt.plot(a.imag[:500])
# %%
