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
(this 1-d series will be 200 by 200 after STFT, not processed in this file)


______________ after loading the [2000, 20100] for each class_______________
each sample is normalized to 1
after shuffled, 1600 samples as training, 400 samples for test
labels are one hot encoding, 1 or 0, as dtype float64
"""

# All the files are saved in /home/chenhao1/Hpython/data/data_ss/stage_1

"""
The 1 channel mixture data are savd as
'pre' + 'dict_mix_' + 'n'
e.g. train_200_dict_mix_2.pt = 'train_200_' + 'dict_mix_' + '2' + '.pt'
pre is for train or test, 200 means, heigh and width =200, 'n' is how many sources in the mixture
in total, the training has 1600*6 + 1600*15 + 1600*20 + 1600*15+ 1600*6 +1600 samples
saved as dictionary, with keys as 'data' and 'label'

it also saved data for training u-net as
'class_names'+ 'tr or va or te' +'.pt'
e.g. ble_tr_200.pt = 'ble_' + 'tr_200' + '.pt'
class_names is the class name, tr is for training, va for validation, te for testing
For each class, there are 800*(6+15+20+15+6+1) samples from the train_dict_mix_n as 
training samples for u-net, which means it used the mixtures samples for supervised learning
the mixture labels are not used for u-net.
the va and te data with 800*(6+15+20+15+6+1) samples for each,vare also from the train_dict_mix_n.
the save data format is torch.Dataloader, with batch_size = 30
"""

#%%
from utils import *
# stft will be done on the last dimension

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

train_val = np.zeros((6, 1600, data[0].shape[1])).astype('complex64')
test = np.zeros((6, 400, data[0].shape[1])).astype('complex64')
for i in range(6): # split 1600 for tain_val, 400 for test
    train_val[i] = data[i][:1600]
    test[i] = data[i][-400:]

#%% generate labels
idx = np.arange(6)
label1 = np.zeros((idx.size, idx.max()+1))
label1[np.arange(idx.size),idx] = 1  # one hot encoding

label2, label3 = label_gen(2), label_gen(3)
label4, label5 = label_gen(4), label_gen(5)
label6 = np.ones((1,6))

#%% save mixture data
save_mix(train_val, label1, label2, label3, label4, label5, label6, pre='train_200_')
save_mix(test, label1, label2, label3, label4, label5, label6, pre='test_200_')

print('done')

#%% ___________________assuming mixture data is done___________________
# dict = torch.load('../data_ss/train_dict_mix_6.pt')  # see 256 data
# f, t, Z = stft(dict['data'][0,0], fs=4e7, nperseg=256, boundary=None)
# plt.figure()
# plt.imshow(abs(np.roll(Z, 128, axis=0)), aspect='auto', interpolation='None')

dict = torch.load('../data/data_ss/train_200_dict_mix_6.pt')
f, t, Z = stft(dict['data'][0,0], fs=4e7, nperseg=200, boundary=None)
plt.figure()
plt.imshow(np.log(abs(np.roll(Z, 100, axis=0))), aspect='auto', interpolation='None')
plt.title('One example of 6-component mixture')

#%% 
"""dict have keys ['data'] shape of [n_comb, n_sample, time_len]
    ['label'] shape of [n_comb, n_class=6]
"""
d, l = get_mixdata_label(mix=1, pre='train_200_')
d1 = d.clone()
for i in range(2,7):
    dt, lt = get_mixdata_label(mix=i, pre='train_200_')  # temp
    d, l = torch.cat( (d, dt)), torch.cat( (l , lt))
xtr, ltr, ytr = d[:, :700], l[:, :700], d1[:, :700]
xva, lva, yva = d[:, 700:800], l[:, 700:800], d1[:, 700:800]
xte, lte, yte = d[:, 800:1000], l[:, 800:1000], d1[:, 800:1000]

"train data for ble" # "training data is the log(abs(stft(x)))"
"0-5 is ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']"
for i in np.arange(0, 6):
    get_Unet_input(xtr, ltr, ytr, which_class=i, tr_va_te='_tr_200')
    get_Unet_input(xva, lva, yva, which_class=i, tr_va_te='_va_200')
    get_Unet_input(xte, lte, yte, which_class=i, tr_va_te='_te_200', shuffle=False)


#%% Generate multi-channel data -- as stage 2
"""
The following code will try to make the single channel data into multipy channel
delta_omega = 2*pi*d*sin(theta)/lambda
d is the antenna space length( distance from one to the next one), typically d=lambda/2
theta is the angle of arrival, -pi/2 to pi/2
lambda is the wave length lambda = c/f, c is light speed, f is frequency of the wave
"""
# All the files are saved in /home/chenhao1/Hpython/data/data_ss/

"""asdf"""
n_c = 6  #number of channels
aoa = torch.tensor([60, 40, 20, -20, -40, -60])  # in degrees for each class
del_omega = (1j* np.pi * (aoa/180*np.pi).sin()).exp()  # for each class

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

train_val = torch.zeros((6, 1600, data[0].shape[1])).to(torch.cfloat)
test = torch.zeros((6, 400, data[0].shape[1])).to(torch.cfloat)
for i in range(6): # split 1600 for tain_val, 400 for test
    train_val[i] = torch.tensor(data[i][:1600])
    test[i] = torch.tensor(data[i][-400:])

# %%  extend one channel to multiple channel
data = torch.cat((train_val, test), 1)  # [n_class=6, n_samples=2000, time_length]
d_mul_c = torch.zeros(6, n_c, data.shape[1], data.shape[2]).to(torch.cfloat)
for i in range(6):
    d_mul_c[i] = data[i] # class i data, copy n_c times
    delays = torch.zeros(n_c, 1, dtype=torch.cfloat)
    for ii in range(n_c):  # get delays for each channel
        delays[ii] = del_omega[i]**ii




def mix_data_torch(x, labels):
    n = labels.shape[0]
    output = np.zeros( (n, x.shape[1], x.shape[2]) ).astype('complex64')
    for i1 in range(n):
        s = 0
        for i2 in range(6):
            if labels[i1, i2] == 1:
                s = s + x[i2]
            else:
                pass
        output[i1] = s
    return torch.tensor(output), torch.tensor(labels).to(torch.float)
# %%
