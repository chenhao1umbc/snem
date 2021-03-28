"""
This file is made to generate the mixture data from LM data set.
The original data was at 40 MHz sampling rate, 1.8e9=25 seconds for each class
we use the compressed data at
/home/chenhao1/Matlab/LMdata/compressed/

the compressed data was generated using the file
/home/chenhao1/Matlab/LMdata/compressed/data_resize.m
basically, what it does is take 4e6 length (0.1s) data make STFT,
then using the image resize to n by n, e.g. we use n=200 here
next, applying iSTFT to get a complex 1-d sequence.

in this file, we have the loaded
temp = sio.loadmat( '/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+'_200_2k.mat')
temp['x'] has the shape of 2000 by 20100, meaning 2000 samples, with each sample lenght 20100
(this 1-d series will be 200 by 200 after STFT, but STFT not processed in this file)

______________ after loading the [2000, 20100] for each class_______________
each sample is normalized to 1
after shuffled, 1600 samples as training, 400 samples for test
labels are one hot encoding, 1 or 0, as dtype float
also this 1-d series are exteded to 6 channels
each classs has angle of arrival (AOA) as 60, 40, 20, -20, -40, -60
positive aoa generates the channel delays as [0, delta_omge, ... delta_omge*5]
negative aoa generates the channel dalays as [delta_omge*5, ...delta_omge, 0]

The 6 channel mixture data are savd as
'pre' + 'dict_mix_' + 'n'
e.g. train_c6_dict_mix_2.pt = 'train_c6_' + 'dict_mix_' + '2' + '.pt'
pre is for train or test, c6 means 6channels with heigh and width =200, 
'n' is how many sources in the mixture
in total, the training has 1600*6 + 1600*15 + 1600*20 + 1600*15+ 1600*6 +1600 samples
saved as dictionary, with keys as 'data' and 'label'

e.g. train_c6_4800_mix_101000.pt = 'train_c6_' + '4800' +'_mix_' + '101000' + '.pt'
pre is for train or test, c6 means 6channels with heigh and width =200, 
'4800' is how many samples of the mixture data. Here 4800 in total
'101000', is the class labels, meaning class 1 and 3 are the mixture sources
saved as dictionary, with keys as 'data' and 'label'
"""
#%%
from utils import *

# %% generate toy real data 
d = sio.loadmat('./data/vj.mat')
vj = torch.tensor(d['vj']).float()
J = vj.shape[-1] # how many sources, J =3
max_db = 20
n_channel = 3

aoa = (torch.rand(J)-0.5)*90 # in degrees
power_db = torch.rand(3)*max_db # power diff for each source
steer_vec = get_steer_vec(aoa, n_channel, J)  # shape [n_sources, n_channel]
cjnf = torch.zeros( 50*50, n_channel, J ) # [n_sources, F,T, n_channel]

for j in range(J):
    temp = vj[:,:,j]
    st_sq = steer_vec[j,:]**2
    cj_nf = (temp.reshape(2500, 1)@st_sq[None, :])**0.5 # x >=0
    cjnf[:, :, j] = cj_nf * (torch.rand(50*50, n_channel)-0.5).sign()*steer_vec[j,:]

for j in range(J):
    cjnf[:,:,j] = 10^[power_db[j]/20] * cjnf[:, :, j]


xnf = sum(cjnf, 3) # sum over all the sources, shape of [N*F, n_channel]
# %%
