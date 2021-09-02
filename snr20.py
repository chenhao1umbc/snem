from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
plt.rcParams['figure.dpi'] = 150
torch.set_printoptions(linewidth=160)
torch.set_default_dtype(torch.double)
import time
import itertools
d, s, h = torch.load('../data/nem_ss/test500M3FT100_xsh.pt')
h = torch.tensor(h)
ratio = d.abs().amax(dim=(1,2,3))/3
x = (d/ratio[:,None,None,None]).permute(0,2,3,1)
s_all = s.abs().permute(0,2,3,1)
t = time.time()

def corr(vh, v):
	J = v.shape[-1]
	r = [] 
	permutes = list(itertools.permutations(list(range(J))))
	for jj in permutes:
		temp = vh[...,jj[0]], vh[...,jj[1]], vh[...,jj[2]]
		s = 0
		for j in range(J):
			s = s + abs(stats.pearsonr(v[...,j].flatten(), temp[j].flatten())[0])
		r.append(s)
	r = sorted(r, reverse=True)
	return r[0]/J
	
def h_corr(h, hh):
	J = h.shape[-1]
	r = [] 
	permutes = list(itertools.permutations(list(range(J))))
	for p in permutes:
		temp = hh[:,torch.tensor(p)]
		s = 0
		for j in range(J):
			dino = h[:,j].norm() * temp[:, j].norm()
			nume = (temp[:, j].conj() @ h[:, j]).abs()
			s = s + nume/dino
		r.append(s/J)
	r = sorted(r, reverse=True)
	return r[0].item()
		
single_data = False
if single_data:
	ind = 0
	shat, Hhat, vhat, Rb = em_func(x[ind])
	for i in range(3):
		plt.figure()
		plt.imshow(shat.squeeze().abs()[...,i]*ratio[ind])
		plt.colorbar()
		plt.show()
	# sio.savemat('x0s_em.mat', {'x':x[ind].numpy(),'s_em':(shat.squeeze().abs()*ratio[ind]).numpy()})
else: # run a lot of samples
	res, res2 = [], []
	for i in range(100):
		c, cc = [], []
		for ii in range(20):
			shat, Hhat, vhat, Rb = em_func(awgn(x[i], snr=20), seed=ii)
			c.append(corr(shat.squeeze().abs(), s_all[i]))
			cc.append(h_corr(h, Hhat))
		res.append(c)
		res2.append(cc)
		print(f'finished {i} samples')

print('Time used is ', time.time()-t)
torch.save([res, res2], 'res_shat_hhat_snr20.pt')