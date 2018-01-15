import numpy as np
import scipy.io as scio
import pickle

nld = 4
nsd = 5
nn = 300
nts = 100

x = np.load('../outputs/xinf.npy')
A = np.load('../outputs/Ainf.npy')
B = np.load('../outputs/Binf.npy')
C = np.load('../outputs/Cinf.npy')
d = np.load('../outputs/dinf.npy')
Q = np.load('../outputs/Qinf.npy')
Q0 = np.load('../outputs/Q0inf.npy')
m0 = np.load('../outputs/m0inf.npy')

y = np.array(np.exp(C @ x[:nld] + d))
for t in range(nts-1):
    y = np.concatenate([y, np.exp(C @ x[(t+1)*nld:(t+2)*nld] + d)])

data = scio.loadmat('../data/compiled_dF033016.mat')
ygen = data['behavdF'].flatten()[:nts*nn]

print(np.mean((y-ygen)**2))