#%%
from utils import *

#%% the body of the EM structure
opts = load_options()
V, X = load_data(data='train')
neural_net = init_neural_network(opts)
vj, cj, Rj, neural_net = train_NEM(V, X, neural_net, opts)

# %% test data
vj, cj, Rj, neural_net = test_NEM(V, X, neural_net, opts)