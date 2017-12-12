# coding: utf-8

import scipy.io as scio
import scipy.sparse as scsp
import scipy.sparse.linalg as splin
import numpy as np
from newton_method import nr_algo
import scipy.optimize as sciop
import numdifftools as nd

import matplotlib.pyplot as plt


def kd(i, j):
    if i == j:
        return 1.
    else:
        return 0.


def logposterior(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd):
    """
    :param y: neuron capture data
    :param C: latent space to neuron space transformation matrix
    :param d: mean firing rates
    :param A: deterministic component of the evolution state[t] to state[t+1]
    :param q: covariance of the innovations that perturb the latent state at each time step
    :param q0: covariance of the initial state x1 of each trial
    :param B: mapping of stimuli to "latent space stimuli"
    :param u: stimuli (4-D one-hot)
    :param nts: number of time steps
    :param nn: number of neurons
    :return: the log-posterior of eq.4 in Macke et al. 2011
    """

    def f(x):

        q0inv = np.linalg.inv(q0)
        qinv = np.linalg.inv(q)

        constants = .5 * np.log(np.linalg.det(q0)) + .5 * (nts-1) * np.log(np.linalg.det(q))

        term1a = - sum(y[t*nn:(t+1)*nn] @ (C @ x[t*nld:(t+1)*nld] + d) for t in range(nts))

        term1b = sum(np.sum(np.exp(C @ x[t*nld:(t+1)*nld] + d)) for t in range(nts))

        term2 = .5 * (x[:nld] - m0).T @ q0inv @ (x[:nld] - m0)

        term3 = .5 * sum((x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd]).T
                         @ qinv @
                         (x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd])
                         for t in range(nts-1))

        return constants + term1a + term1b + term2 + term3
    return f


def logposteriorderivative(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld):
    def f(x):
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        ATQinv = A.T @ Qinv
        ATQinvA = A.T @ Qinv @ A
        ATQinvB = A.T @ Qinv @ B
        QinvA = Qinv @ A
        QinvB = Qinv @ B

        df = np.zeros_like(x)
        df[:nld] = - C.T @ y[:nn] + C.T @ np.exp(C @ x[:nld] + d) + \
                   Q0inv @ (x[:nld] - m0) - ATQinv @ (x[nld:2*nld] - A @ x[:nld] - B @ u[:nsd])

        for t in range(1, nts-1):
            df[t*nld:(t+1)*nld] = - C.T @ y[t*nn:(t+1)*nn] \
                                  + C.T @ np.exp(C @ x[t*nld:(t+1)*nld] + d) \
                                  - ATQinv @ x[(t+1)*nld:(t+2)*nld] \
                                  + ATQinvA @ x[t*nld:(t+1)*nld] \
                                  + ATQinvB @ u[t*nsd:(t+1)*nsd] \
                                  + Qinv @ x[t*nld:(t+1)*nld] \
                                  - QinvA @ x[(t-1)*nld:t*nld] \
                                  - QinvB @ u[(t-1)*nsd:t*nsd]

        df[-nld:] = - C.T @ y[-nn:] \
                    + C.T @ np.exp(C @ x[-nld:] + d) \
                    + Qinv @ (x[-nld:] - A @ x[-2*nld:-nld] - B @ u[-2*nsd:-nsd])
        return df
    return f


def logposteriorhessian(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld):
    def f(x):

        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        ATQinvA = A.T @ Qinv @ A
        ATQinvAplusQinv = ATQinvA + Qinv
        ATQinv = A.T @ Qinv

        diag = []
        off_diag = []
        diag.append(scsp.lil_matrix(Q0inv + ATQinvA + sum(np.exp(C[i] @ x[:nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn))))
        for t in range(1, nts-1):
            diag.append(scsp.lil_matrix(sum(np.exp(C[i] @ x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn)) + ATQinvAplusQinv))
        diag.append(scsp.lil_matrix(Qinv + sum(np.exp(C[i] @ x[-nld:] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(nn))))

        for t in range(0, nts-1):
            off_diag.append(scsp.lil_matrix(-ATQinv))

        h = scsp.block_diag(diag).tolil()
        od = scsp.block_diag(off_diag).tolil()

        h[:-nld, nld:] += od
        h[nld:, :-nld] += od.T

        return h.tocsc()
    return f


def jointloglikelihood(y, nsd, nn, nts, mu, cov, Q, Q0, m0, A, u, B):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        jll = sum(-y[t*nn:(t+1)*nn] @ C @ mu[t*nld:(t+1)*nld] - y[t*nn:(t+1)*nn] @ d \
                  + sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) +
                        d[i]) for i in range(nn))
                  for t in range(nts-1))
        return jll
    return f


def jllDerivative(nn, nld, mu, cov, nts, y):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        djlld = np.empty(nn)
        for i in range(nn):
            djlld[i] = sum(-y[t*nn+i] \
                    + np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]))
                    for t in range(nts-1))

        djllC = [sum(-y[t*nn+i] * mu[t*nld:(t+1)*nld] + \
                    np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])) * \
                    (mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                    for t in range(nts-1)) for i in range(nn)]

        djlldC = np.empty(nn*(nld+1))
        for i in range(nn):
            djlldC[i*(nld+1)] = djlld[i]
            djlldC[i*(nld+1) + 1:(i+1)*(nld+1)] = djllC[i]

        return djlldC
    return f


def jllHessian(nn, nld, mu, cov, nts, y):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        blocks = []

        for i in range(nn):
            block = np.zeros((1 + nld, 1 + nld))
            block[0, 0] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])) for t in range(nts-1))

            block[0, 1:] = sum((mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + \
                    .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])) for t in range(nts-1))

            block[1:, 0] = block[0, 1:]

            block[1:, 1:] = sum((cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
                np.outer(mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i], (mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]).T))*\
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]))
                for t in range(nts-1))

            blocks.append(scsp.lil_matrix(block))

        HJLL = scsp.block_diag(blocks)
        return HJLL.tocsc()
    return f


def blockdiaginv(m, nb, bw):
    minv = np.zeros_like(m)
    for b in range(nb):
        minv[b*bw:(b+1)*bw, b*bw:(b+1)*bw] = np.linalg.inv(m[b*bw:(b+1)*bw, b*bw:(b+1)*bw])
    return minv


def laplace_approximation(f, df, hf, x, nts, nld):
    # use NR algorithm to compute minimum of log-likelihood
    x = nr_algo(f, df, hf, x)
    # x = sciop.minimize(f, x0, jac=df, hess=hf, method='Newton-CG').x
    # negative inverse of Hessian is covariance matrix
    covariance = blockdiaginv(hf(x), nts, nld)
    return x, covariance


if __name__ == "__main__":
    # load data
    print('loading data')

    nts = 10
    nn = 6
    nld = 3
    nsd = 4
    y = np.load('ytest.npy')
    u = np.zeros(nts * nsd)

    print('variable initialization')
    # Initialize parameters to random values
    C = np.load('Ctest.npy')
    # C = np.random.randn(nn, nld)
    d = np.load('dtest.npy')
    # d = np.random.randn(nn)
    m0 = np.zeros(nld)
    A = np.zeros((nld, nld))
    # A = np.load('Atest.npy')
    q0 = np.identity(nld)
    q = np.identity(nld)
    B = np.zeros((nld, nsd))
    mu = np.load('xtest.npy') # + np.random.randn(nld*nts)
    xgen = np.load('xtest.npy')
    Cgen = np.load('Ctest.npy')
    dgen = np.load('dtest.npy')

    rsquaredC = [1 - np.mean((C-Cgen)**2 / Cgen**2)]
    rsquaredd = [1 - np.mean((d-dgen)**2 / dgen**2)]
    rsquaredx = [1 - np.mean((mu-xgen)**2 / xgen**2)]

    print('begin training')
    max_epochs = 25
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch))
        print('performing laplace approximation')
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance

        lp = logposterior(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd)

        numdif = np.empty_like(mu)
        epsilon = .0001
        for dim in range(len(mu)):
            delt = np.zeros_like(mu)
            delt[dim] = epsilon
            numdif[dim] = (lp(mu + delt) - lp(mu - delt))/(2*epsilon)

        mu, cov = laplace_approximation(logposterior(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd),
                                        logposteriorderivative(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld),
                                        logposteriorhessian(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld),
                                        mu, nts, nld)

        print("rsquared of mu and xgen = ", 1 - np.mean((mu-xgen)**2 / xgen**2))

        print('assigning analytic expressions')
        # Use analytic expressions to compute parameters m0, Q, Q0, A, B

        m0 = mu[:nld]
        q0 = cov[:nld, :nld]

        A = sum(cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) -
                np.outer(B @ u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)) @ \
                np.linalg.inv(sum(cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
                np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)))

        B = sum(np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd]) - A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd])
                for t in range(nts-1)) @ np.linalg.inv(sum(np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd].T) for t in range(nts-1)))

        q = (1/(nts-1))*sum(
            cov[(t+1)*nld:(t+2)*nld, (t+1)*nld:(t+2)*nld] \
            + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[(t+1)*nld:(t+2)*nld].T)
            - (cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T)) @ A.T
            - np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd].T) @ B.T
            - A @ (cov[t*nld:(t+1)*nld, (t+1)*nld:(t+2)*nld] + np.outer(mu[t*nld:(t+1)*nld], mu[(t + 1)*nld:(t+2)*nld].T))
            + A @ (cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] + np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T)) @ A.T
            + A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd].T) @ B.T
            - B @ np.outer(u[t*nsd:(t+1)*nsd], mu[(t+1)*nld:(t+2)*nld].T)
            + B @ np.outer(u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) @ A.T
            + B @ np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd].T) @ B.T
            for t in range(nts-1))

        # Second NR minimization to compute C, d (and in principle, D)
        print('performing NR algorithm for parameters C, d')
        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = []
        for i in range(nn):
            dC.append(d[i])
            dC += list(C[i])
        dC = np.array(dC)

        jl = jointloglikelihood(y, nsd, nn, nts, mu, cov, q, q0, m0, A, u, B)

        numdif = np.empty_like(dC)
        epsilon = .0001
        for dim in range(len(dC)):
            delt = np.zeros_like(dC)
            delt[dim] = epsilon
            numdif[dim] = (jl(dC + delt) - jl(dC - delt))/(2*epsilon)

        dC = nr_algo(jointloglikelihood(y, nsd, nn, nts, mu, cov, q, q0, m0, A, u, B),
                     jllDerivative(nn, nld, mu, cov, nts, y),
                     jllHessian(nn, nld, mu, cov, nts, y),
                     dC)

        for i in range(nn):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1)+1:(i+1)*(nld+1)]

        rsquaredC.append(1 - np.mean((C-Cgen)**2 / Cgen**2))
        rsquaredd.append(1 - np.mean((d-dgen)**2 / dgen**2))
        rsquaredx.append(1 - np.mean((mu-xgen)**2 / xgen**2))

    plt.plot(rsquaredC, 'k-')
    plt.plot(rsquaredd, 'b-')
    plt.plot(rsquaredx, 'g-')
    plt.show()