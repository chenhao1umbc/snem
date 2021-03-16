#%%
from utils import *

#%% the body of the EM structure
opts = load_options()
model = init_neural_network(opts)
x, v = load_data(data='train')
# vj, cj, Rj, neural_net = train_NEM(x, v, model, opts)

# %% test data
vj, cj, Rj, neural_net = test_NEM(V, X, neural_net, opts)

# %%
