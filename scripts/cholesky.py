import numpy as np
import scipy.sparse as scsp
np.set_printoptions(precision=2)
import sksparse.cholmod as chol


def computecov(h, nld, nts):
    # compute U
    Udiag = list(range(nts))
    Uoff = list(range(nts-1))
    Udiag[0] = np.linalg.cholesky(h[:nld,:nld]).T
    for t in range(1, nts): #t goes from 1 to nts-1
        Uoff[t-1] = np.linalg.inv(Udiag[t-1].T) @ h[(t-1)*nld:t*nld, t*nld:(t+1)*nld]
        Udiag[t] = np.linalg.cholesky(h[t*nld:(t+1)*nld, t*nld:(t+1)*nld] - Uoff[t-1].T @ Uoff[t-1]).T

    covdiag = list(range(nts))
    covoff = list(range(nts-1))
    covdiag[-1]  = np.linalg.inv(Udiag[-1].T @ Udiag[-1])   #[-1] = [nts-1]
    for t in range(nts-2, -1, -1): #from nts-2 to 0
        A = np.linalg.inv(Udiag[t]) @ Uoff[t]
        covoff[t] = - A @ covdiag[t+1]
        covdiag[t] = np.linalg.inv(Udiag[t].T @ Udiag[t]) - covoff[t] @ A.T

    return covdiag, covoff