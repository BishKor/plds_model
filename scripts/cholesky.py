import numpy as np

def computecov(h, nld, nts):
    # compute U
    U=np.zeros((nld*nts,nld*nts))
    U[:nld, :nld] = np.linalg.cholesky(h[:nld,:nld]).T
    for t in range(1, nts):
        U[(t-1)*nld:t*nld,t*nld:(t+1)*nld] = -np.linalg.inv(U[(t-1)*nld:t*nld,(t-1)*nld:t*nld]) @ h[(t-1)*nld:t*nld,t*nld:(t+1)*nld]
        U[t*nld:(t+1)*nld,t*nld:(t+1)*nld] = np.linalg.cholesky(h[t*nld:(t+1)*nld,t*nld:(t+1)*nld] - U[(t-1)*nld:t*nld,t*nld:(t+1)*nld].T@U[(t-1)*nld:t*nld,t*nld:(t+1)*nld])

    cov = np.zeros((nld*nts, nld*nts))
    cov[-nts:,-nts:]  = np.linalg.inv(U[-nts:,-nts:].T @ U[-nts:,-nts:])
    for t in range(nts, -1, -1):
        cov[t*nld:(t+1)*nld,(t+1)*nld:(t+2)*nld] = -np.linalg.inv(U[t*nld:(t+1)*nld,t*nld:(t+1)*nld]) @ U[t*nld:(t+1)*nld,(t+1)*nld:(t+2)*nld] @ cov[(t+1)*nld:(t+2)*nld,(t+1)*nld:(t+2)*nld]
        cov[t*nld:(t+1)*nld,t*nld:(t+1)*nld] = np.linalg.inv(U[t*nld:(t+1)*nld,t*nld:(t+1)*nld].T @ U[t*nld:(t+1)*nld,t*nld:(t+1)*nld]) - \
                            cov[t*nld:(t+1)*nld,(t+1)*nld:(t+2)*nld] @ (U[t*nld:(t+1)*nld,t*nld:(t+1)*nld] @ \
                                                  U[t*nld:(t+1)*nld,(t+1)*nld:(t+2)*nld]).T
    return cov
