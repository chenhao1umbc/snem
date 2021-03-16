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

"""
#%%
from utils import *
# stft will be done on the last dimension

#%% Generate multi-channel data -- as stage 2
"""
The following code will try to make the single channel data into multipy channel
delta_omega = 2*pi*d*sin(theta)/lambda
d is the antenna space length( distance from one to the next one), typically d=lambda/2
theta is the angle of arrival, -pi/2 to pi/2
lambda is the wave length lambda = c/f, c is light speed, f is frequency of the wave

# All the files are saved in /home/chenhao1/Hpython/data/data_ss/
"""

#%% data processing 
"raw data processing"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
data = {}
for i in range(6):
    temp = sio.loadmat( '/home/chenhao1/Matlab/LMdata/compressed/'+var_name[i]+'_200_2k.mat')
    dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
    data[i] = temp['x'] / dd  # normalized very data to 1
    # d = torch.tensor(a['x']).to(torch.cfloat)  # torch complex64

"shuffle and split data to train_val and test"
np.random.seed(0)
for i in range(6): # data[i].shape is (2000, 32896)
    np.random.shuffle(data[i])

d = torch.zeros(6, data[0].shape[0], data[0].shape[1]).to(torch.cfloat)
for i in range(6): # combine all the classes
    d[i] = torch.tensor( data[i] )

# %%  extend one channel to multiple channel
n_c = 6  #number of channels
aoa = torch.tensor([60, 40, 20, -20, -40, -60])  # in degrees for each class
del_omega = (1j* np.pi * (aoa/180*np.pi).sin()).exp()  # for each class

"[n_class=6, n_c, n_samples=2000, time_length]"
d_mul_c = torch.zeros(6, n_c, d.shape[1], d.shape[2]).to(torch.cfloat)
for i in range(6):  # loop through class i
    d_mul_c[i] = d[i] # class i data, copy n_c times
    if i < 3:
        "delay for each channel, first channel has delay 0, for positive angles"
        "Last channel has delay 0, for positive angles"
        for ii in range(n_c):  # get delays for each channel
            delay = del_omega[i]**ii
            "apply delay for each channel"
            d_mul_c[i, ii] = d_mul_c[i, ii] * delay
    else:
        for ii in range(n_c):  # get delays for each channel
            delay = del_omega[i]**(n_c - ii - 1)
            "apply delay for each channel"
            d_mul_c[i, ii] = d_mul_c[i, ii] * delay

d_mul_c = d_mul_c.permute(0, 2, 3, 1) #[n_class, n_samples, time_length, n_c] channel last

#%% generate labels
idx = torch.arange(6)
label1 = torch.zeros(6,6)
label1[idx,idx] = 1  # one hot encoding

label2, label3 = label_gen(2), label_gen(3)
label4, label5 = label_gen(4), label_gen(5)
label6 = torch.ones((1,6))

#%% save mixture data
train_val = d_mul_c[:,:1600]
test = d_mul_c[:,1600:]
save_mix(train_val, label1, label2, label3, label4, label5, label6, pre='train_c6_')
save_mix(test, label1, label2, label3, label4, label5, label6, pre='test_c6_')

# %%


