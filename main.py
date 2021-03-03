#%%
from utils import *

#%% the body of the EM structure
opts = {}
opts['n_epochs'] = 25 
opts['lr'] = 0.001
opts['n_batch'] = 64
opts['n_iter'] = 10 # EM iterations

V, X = load_data(data='train')
neural_net = init_neural_network(opts)
vj, cj, Rj, neural_net = train_NEM(V, X, neural_net, opts)

 
# %% test data
vj, cj, Rj, neural_net = test_NEM(V, X, neural_net, opts)