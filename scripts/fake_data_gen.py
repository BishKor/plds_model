import numpy as np
import pickle

nts = 1000
nn = 40
nld = 2
nsd = 4
A = np.identity(nld)*.4 + .1
B = np.random.randn(nld, nsd)/2
C = np.random.randn(nn, nld)/5
d = np.ones(nn)*.7
Q = np.identity(nld)
Q0 = np.identity(nld)
m0 = np.zeros(nld)
u = np.zeros((nts, nsd))

numstim = 150
stimlength = 5
for stim in range(numstim):
    tmpu = np.zeros(nsd)
    tmpu[np.random.randint(0, nsd)] = 1.
    u[stim*(nts//numstim + stimlength):stim*(nts//numstim + stimlength)+stimlength] = tmpu
u = u.flatten()

x = Q0 @ np.random.randn(nld) + m0
y = np.exp(C @ x + d)
# y = [np.random.poisson(lam=np.exp(C @ x[0] + d))]
for t in range(nts-1):
    x = np.concatenate([x, A @ x[t*nld:(t+1)*nld] + Q @ np.random.randn(nld) + B @ u[t*nsd:(t+1)*nsd]])
    y = np.concatenate([y, np.exp(C @ x[(t+1)*nld:(t+2)*nld] + d)])

testdatacontents = {'A': A, 'B': B, 'C': C, 'd': d, 'm0': m0, 'Q': Q, 'Q0': Q0, 'u': u, 'x': x, 'y': y, 'nts': nts,
                    'nn': nn, 'nld': nld, 'nsd': nsd}

opfile = open('../testmats/testparamsanddata.pldsip', 'wb')
pickle.dump(testdatacontents, opfile)
opfile.close()
