"""
%This file will generate real(not complex) toy data
% Experiment 1
% Suppose there are 3 classes, the vj is given, 6 Chennels with real "steer vector"
% vj shape of 50*50, the component sources have variation of AOA and power 
% all the x(n,f) the stft is real number
% 1. try Duang's method
% 2. try neural EM with 3 networks

% Experiment 2
% Suppose there are 3 classes, AOA and power are given
% there are variation of vj in each class
% try neural EM with 3 networks to see how \gamma_j works
"""
#%%
from utils import *

# %% generate toy real data for experiment 1
"""
% real world data is complex time series cj(t)
% it has n_channel channels as cj(t).*steer_vec
% vj(n,f) = sum(|STFT(cj(t).*steer_vec)|.^2)/n_channel

% Here to make sure cj(n,f) is real number, is real number, we have
% cj(n,f) = cj_(n,f).*steer_vec
% vj(n,f) = sum(|cj(n,f)|.^2)/n_channel, where cj(n,f) is the real number

% the following code is a demo of showing the correctness
% the code for exp_1 data generation starts at line 104

% check if vj can be calculated from cjnf
cjnf = cjnf.permute(3,1,2,0) 
cjnf = cjnf /(10**(power_db[None, None, :]/20))
cjnf = cjnf.permute(3,1,2,0) 
v = ((cjnf*steer_vec[:, None, None, :])**2).sum(-1)/n_channel
"""

d = sio.loadmat('./data/vj.mat')
vj = torch.tensor(d['vj']).float()
J = vj.shape[-1] # how many sources, J =3
max_db = 20
n_channel = 3

N = 20000
x = torch.zeros(N, 50, 50, 3)
cj = torch.zeros(N, n_channel, 50, 50, 3)
for i in range(N):
    aoa = (torch.rand(J)-0.5)*90 # in degrees
    power_db = torch.rand(J)*max_db # power diff for each source
    steer_vec = get_steer_vec(aoa, n_channel, J)  # shape of [n_sources, n_channel]
    cjnf = vj**0.5 * 1/steer_vec.t()[:, None, None, :]# shape of [n_channel, F, T, n_sources]
    cjnf = 10**(power_db[None, None, :]/20) * cjnf
    cjnf = cjnf.permute(3,1,2,0)  # shape as [n_sources, F, T, n_channel]
    xnf = cjnf.sum(0) # sum over all the sources, shape of [F, T, n_channel]

    x[i] = xnf
    cj[i] = cjnf
# torch.save(x, 'x_toy1.pt')
# torch.save(cj, 'cj_toy1.pt')

# %% toy experiment 2
d = sio.loadmat('./data/vj.mat')
vj = torch.tensor(d['vj']).float()
J = vj.shape[-1] # how many sources, J =3
max_db = 20
n_channel = 3

N = 10000
x = torch.zeros(N, 50, 50, 3)
cj = torch.zeros(N, n_channel, 50, 50, 3)
for i in range(N):
    move_v = {0:[torch.randint(-5, 15,(1,)), torch.randint(-10, 15,(1,))],
            1:[torch.randint(-2, 18,(1,)), torch.randint(-15, 5,(1,))],
            2:[torch.randint(-12, 5,(1,)), torch.randint(-10, 10,(1,))]  
            }
    for j in range(J):
        vj [..., j] = torch.roll(vj [..., j], move_v[j], (0,1))
    # plt.imshow(vj[..., 2])

    aoa = torch.tensor([20, 45, 70]) # in degrees
    steer_vec = get_steer_vec(aoa, n_channel, J)  # shape of [n_sources, n_channel]
    cjnf = vj**0.5 * 1/steer_vec.t()[:, None, None, :] # shape of [n_channel, F, T, n_sources]
    cjnf = cjnf.permute(3,1,2,0)  # shape as [n_sources, F, T, n_channel]
    xnf = cjnf.sum(0) # sum over all the sources, shape of [F, T, n_channel]
    x[i] = xnf
    cj[i] = cjnf
torch.save(x, 'x_toy2.pt')
torch.save(cj, 'cj_toy2.pt')
# %%
