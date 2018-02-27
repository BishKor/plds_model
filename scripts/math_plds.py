import scipy.sparse as scsp
from newton_method import *
import numpy as np
from memory_profiler import profile


def logposterior(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld):

    """
    :param y: numpy array for flattened neuron G2+ timeseries spiking data
    :param C: latent -> neuron space transformation matrix
    :param d: mean spike rates
    :param A: deterministic component of the transition s[t]->s[t+1]
    :param Q: covariance of noise perturbing latent state
    :param Q0: covariance of the initial state latent state x1
    :param B: mapping of stimuli to latent space driver
    :param u: stimuli (4-D one-or-none-hot)
    :param nts: number of time steps
    :param nn: number of neurons
    :return: the log-posterior of eQ.4 in Macke et al. 2011
    """

    Q0inv = np.linalg.inv(Q0)
    Qinv = np.linalg.inv(Q)

    @profile
    def f(x):
        lptotal = .5 * np.log(np.linalg.det(Q0)) + .5 * (nts-1) * np.log(np.linalg.det(Q))

        lptotal += - sum(y[t*nn:(t+1)*nn] @ (C @ x[t*nld:(t+1)*nld] + d) for t in range(nts))

        lptotal += sum(np.sum(np.exp(C @ x[t*nld:(t+1)*nld] + d)) for t in range(nts))

        lptotal += .5 * (x[:nld] - m0).T @ Q0inv @ (x[:nld] - m0)

        lptotal += .5 * sum((x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd]).T
                         @ Qinv @
                         (x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd])
                         for t in range(nts-1))

        return - lptotal
    return f


def logposteriorderivative(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld):
    Qinv = np.linalg.inv(Q)
    Q0inv = np.linalg.inv(Q0)
    ATQinv = A.T @ Qinv
    ATQinvA = A.T @ Qinv @ A
    ATQinvB = A.T @ Qinv @ B
    QinvA = Qinv @ A
    QinvB = Qinv @ B

    @profile
    def f(x):
        df = np.zeros_like(x)
        df[:nld] += - C.T @ y[:nn] + C.T @ np.exp(C @ x[:nld] + d) + Q0inv @ (x[:nld] - m0) - \
                   ATQinv @ (x[nld:2*nld] - A @ x[:nld] - B @ u[:nsd])

        for t in range(1, nts-1):
            df[t*nld:(t+1)*nld] += - C.T @ y[t*nn:(t+1)*nn] \
                                     + C.T @ np.exp(C @ x[t*nld:(t+1)*nld] + d) \
                                  - ATQinv @ x[(t+1)*nld:(t+2)*nld] \
                                  + ATQinvA @ x[t*nld:(t+1)*nld] \
                                  + ATQinvB @ u[t*nsd:(t+1)*nsd] \
                                  + Qinv @ x[t*nld:(t+1)*nld] \
                                  - QinvA @ x[(t-1)*nld:t*nld] \
                                  - QinvB @ u[(t-1)*nsd:t*nsd]

        df[-nld:] += - C.T @ y[-nn:] + C.T @ np.exp(C @ x[-nld:] + d) \
                    + Qinv @ (x[-nld:] - A @ x[-2*nld:-nld] - B @ u[-2*nsd:-nsd])

        return df
    return f


def logposteriorhessian(C, d, A, Q, Q0, nts, nn, nld):

    Qinv = np.linalg.inv(Q)
    Q0inv = np.linalg.inv(Q0)
    ATQinvA = A.T @ Qinv @ A
    ATQinvAplusQinv = ATQinvA + Qinv
    ATQinv = A.T @ Qinv

    @profile
    def f(x):

        diag = []
        off_diag = []
        diag.append(scsp.lil_matrix(Q0inv + ATQinvA + sum(np.exp(C[i] @ x[:nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn))))
        for t in range(1, nts-1):
            diag.append(scsp.lil_matrix(sum(np.exp(C[i] @ x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn)) + ATQinvAplusQinv))
        diag.append(scsp.lil_matrix(sum(np.exp(C[i] @ x[-nld:] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn)) + Qinv))

        for t in range(0, nts-1):
            off_diag.append(scsp.lil_matrix(-ATQinv))

        h = scsp.block_diag(diag).tolil()
        od = scsp.block_diag(off_diag).tolil()

        h[:-nld, nld:] += od
        h[nld:, :-nld] += od.T

        return h.tocsc()
    return f


def jointloglikelihood(nld, nn, nts, mu, covd, y):
    @profile
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        jll = sum(-y[t*nn:(t+1)*nn] @ C @ mu[t*nld:(t+1)*nld] - y[t*nn:(t+1)*nn] @ d \
                  + sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + .5 * (C[i] @ covd[t] @ C[i]) +
                    d[i]) for i in range(nn)) for t in range(nts-1)) + .5 * .1 * np.sum(C**2) + .1 * np.sum(np.abs(C))
        return jll
    return f


def jllDerivative(nn, nld, mu, covd, nts, y):
    @profile
    def f(dC):
        print(nld)
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        djlld = np.zeros(nn)
        for i in range(nn):
            djlld[i] = sum(-y[t*nn+i] +
                           np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i]))
                           for t in range(nts-1))

        djllC = [sum(-y[t*nn+i] * mu[t*nld:(t+1)*nld] + \
                     np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i])) * \
                     (mu[t*nld:(t+1)*nld] + covd[t] @ C[i])
                     for t in range(nts-1)) for i in range(nn)]

        djlldC = np.empty(nn*(nld+1))
        for i in range(nn):
            djlldC[i*(nld+1)] = djlld[i]
            djlldC[i*(nld+1) + 1:(i+1)*(nld+1)] = djllC[i] + .1 * np.sign(C[i]) + .1 * C[i]

        return djlldC
    return f


def jllHessian(nn, nld, mu, covd, nts):
    @profile
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        blocks = []

        for i in range(nn):
            block = .1 * np.identity(1 + nld)
            block[0, 0] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i])) for t in range(nts-1))

            block[0, 1:] += sum((mu[t*nld:(t+1)*nld] + covd[t] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + \
                    .5 * (C[i] @ covd[t] @ C[i])) for t in range(nts-1))

            block[1:, 0] += block[0, 1:]

            block[1:, 1:] += sum((covd[t] +
                np.outer(mu[t*nld:(t+1)*nld] + covd[t] @ C[i], (mu[t*nld:(t+1)*nld] + covd[t] @ C[i]).T))*\
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i]))
                for t in range(nts-1))

            blocks.append(scsp.lil_matrix(block))

        HJLL = scsp.block_diag(blocks)
        return HJLL.tocsc()
    return f

@profile
def computecov(h, nld, nts):
    # compute U
    Udiag = list(range(nts))
    Uoff = list(range(nts-1))
    Udiag[0] = np.linalg.cholesky(h[:nld, :nld]).T
    for t in range(1, nts):  # t goes from 1 to nts-1
        Uoff[t-1] = np.linalg.inv(Udiag[t-1].T) @ h[(t-1)*nld:t*nld, t*nld:(t+1)*nld]
        Udiag[t] = np.linalg.cholesky(h[t*nld:(t+1)*nld, t*nld:(t+1)*nld] - Uoff[t-1].T @ Uoff[t-1]).T

    # Udiag[0] = np.linalg.cholesky(hdiag[0]).T
    # for t in range(1, nts):  # t goes from 1 to nts-1
    #     Uoff[t-1] = np.linalg.inv(Udiag[t-1].T) @ hoffdiag[t]
    #     Udiag[t] = np.linalg.cholesky(hdiag[t] - Uoff[t-1].T @ Uoff[t-1]).T

    covdiag = list(range(nts))
    covoff = list(range(nts-1))
    covdiag[-1] = np.linalg.inv(Udiag[-1].T @ Udiag[-1])  # [-1] = [nts-1]
    for t in range(nts-2, -1, -1):  # from nts-2 to 0
        A = np.linalg.inv(Udiag[t]) @ Uoff[t]
        covoff[t] = - A @ covdiag[t+1]
        covdiag[t] = np.linalg.inv(Udiag[t].T @ Udiag[t]) - covoff[t] @ A.T

    return covdiag, covoff

@profile
def laplace_approximation(f, df, hf, x, nts, nld):
    # use NR algorithm to compute minimum of log-likelihood
    x = nr_algo(f, df, hf, x)
    covdiag, covoffdiag = computecov(hf(x).toarray(), nld, nts)
    return x, covdiag, covoffdiag

