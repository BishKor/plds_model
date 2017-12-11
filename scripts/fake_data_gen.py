import numpy as np

nts = 100
nn = 3
nld = 2

C = np.random.randn(nn, nld)
d = np.ones(nn)
A = np.identity(nld)*.5
Q = np.identity(nld)
Q0 = np.identity(nld)
m0 = np.zeros(nld)

x = Q0 @ np.random.randn(nld) + m0
y = C @ x + d
r = np.exp(C @ x + d)
# y = [np.random.poisson(lam=np.exp(C @ x[0] + d))]

for t in range(nts-1):
    x = np.concatenate([x, A @ x[t*nld:(t+1)*nld] + Q @ np.random.randn(nld)])
    y = np.concatenate([y, C @ x[(t+1)*nld:(t+2)*nld] + d])
    r = np.concatenate([r, np.exp(C @ x[(t+1)*nld:(t+2)*nld] + d)])

np.save("../testmats/m0gen.npy", m0)
np.save("../testmats/Q0gen.npy", Q0)
np.save("../testmats/Qgen.npy", Q)
np.save("../testmats/Cgen.npy", C)
np.save("../testmats/dgen.npy", d)
np.save("../testmats/xgen.npy", x)
np.save("../testmats/Agen.npy", A)
np.save("../testmats/ygen.npy", y)
np.save("../testmats/rgen.npy", r)
