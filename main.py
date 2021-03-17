#%%
from utils import *

#%% the body of the EM structure
opts = load_options()
model = init_neural_network(opts)
x, v = load_data(data='val')
vj, cj, Rj, neural_net = train_NEM(x, v, model, opts)

# %% test data
vj, cj, Rj, neural_net = test_NEM(v, x, neural_net, opts)

# %%
