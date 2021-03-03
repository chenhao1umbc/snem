# %%  visualize raw data
from utils import *
plt.rcParams['figure.dpi'] = 100

a = sio.loadmat('/home/chenhao1/Matlab/LMdata/wifi1_256_2k.mat')
d = a['x']
print(d.shape)

f, t, Z = stft(d[0], fs=4e7, nperseg=256, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 128, axis=0)), aspect='auto', interpolation='None')

f, t, Z = stft(d[1], fs=4e7, nperseg=256, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 128, axis=0)), aspect='auto', interpolation='None')

# a = h5py.File('/home/chenhao1/Matlab/Hsctnet/test.mat', 'r')
# data = a['x'][0][:10]['real'] + 1j*a['x'][0][:10]['imag']
# d = a['x'][0]['real'] + 1j*a['x'][0]['imag']
# dd = d[:int(4e5)]
# f, t, Z = stft(dd, fs=4e7, nperseg=128, boundary=None)
# plt.imshow(abs(np.roll(Z, 64, axis=0)), aspect='auto', interpolation='None')
# print(f'Z is shape of {Z.shape}')
#%%
from utils import *
plt.rcParams['figure.dpi'] = 100

a = sio.loadmat('/home/chenhao1/Matlab/LMdata/wifi1_128_2k.mat')
d = a['x']
print(d.shape)

f, t, Z = stft(d[0], fs=4e7, nperseg=128, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 64, axis=0)), aspect='auto', interpolation='None')

#%%
from utils import *
plt.rcParams['figure.dpi'] = 100

a = sio.loadmat('/home/chenhao1/Matlab/LMdata/wifi1_200_2k.mat')
d = a['x']
print(d.shape)

f, t, Z = stft(d[0], fs=4e7, nperseg=200, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 100, axis=0)), aspect='auto', interpolation='None')

# %% view mixture
from utils import *
plt.rcParams['figure.dpi'] = 100

dict = torch.load('../data_ss/test_dict_mix_1.pt')
f, t, Z = stft(dict['data'][4,0], fs=4e7, nperseg=256, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 128, axis=0)), aspect='auto', interpolation='None')

# %%
from utils import *
plt.rcParams['figure.dpi'] = 100

dict = torch.load('../data_ss/test_128_dict_mix_1.pt')
f, t, Z = stft(dict['data'][4,0], fs=4e7, nperseg=128, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 64, axis=0)), aspect='auto', interpolation='None')

# %%
