# coding: utf-8

import scipy.io as scio
import scipy.sparse as scsp
import scipy.sparse.linalg as splin
import numpy as np
from newton_method import nr_algo


def kd(i, j):
    if i == j:
        return 1.
    else:
        return 0.


def logposterior(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd):
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
    :param n_neurons: number of neurons
    :return: the log-posterior of eq.4 in Macke et al. 2011
    """

    def logpost(x):

        # first compute useful values
        q0inv = np.linalg.inv(q0)
        qinv = np.linalg.inv(q)

        constants = .5 * np.log(np.linalg.det(q0)) + .5 * (nts-1) * np.log(np.linalg.det(q))

        term1a = - sum(y[t*n_neurons:(t+1)*n_neurons] @ (C @ x[t*nld:(t+1)*nld] + d) for t in range(nts))

        term1b = sum(np.sum(np.exp(C @ x[t*nld:(t+1)*nld] + d)) for t in range(nts))

        term2 = .5 * (x[:nld] - x0).T @ q0inv @ (x[:nld] - x0)

        term3 = .5 * sum((x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd])
                         @ qinv @
                         (x[(t+1)*nld:(t+2)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*nsd:(t+1)*nsd])
                         for t in range(nts-1))

        return constants + term1a + term1b + term2 + term3
    return logpost


def logposteriorderivative(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd, nld):
    def f(x):
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        ATQinv = A.T @ Qinv
        ATQinvA = A.T @ Qinv @ A
        ATQinvB = A.T @ Qinv @ B
        QinvA = Qinv @ A
        QinvB = Qinv @ B


        df = np.empty_like(x)
        df[:nld] = Q0inv @ (x[:nld] - x0) + ATQinv @ (x[nld:2*nld] - A @ x[:nld] - B @ u[:nsd])

        for t in range(1, nts-1):
            df[t*nld:(t+1)*nld] = - C.T @ y[t*n_neurons:(t+1)*n_neurons] \
                                  + C.T @ np.exp(C @ x[t*nld:(t+1)*nld] + d) \
                                  - ATQinv @ x[(t+1)*nld:(t+2)*nld] \
                                  + ATQinvA @ x[t*nld:(t+1)*nld] \
                                  + ATQinvB @ u[t*nsd:(t+1)*nsd] \
                                  + Qinv @ x[t*nld:(t+1)*nld] \
                                  - QinvA @ x[(t-1)*nld:t*nld] \
                                  - QinvB @ u[(t-1)*nsd:t*nsd]

        df[-nld:] = - C.T @ y[-n_neurons:] + C.T @ np.exp(C @ x[-nld:] + d) \
                           + Q0inv @ (x[-nld:] - A @ x[-2*nld:-nld] - B @ u[-2*nsd:-nsd])
        return df
    return f


def logposteriorhessian(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd, nld):
    def f(x):
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        ATQinvA = A.T @ Qinv @ A
        ATQinv = A.T @ Qinv
        ATQinvAplusQinv = ATQinvA + Qinv
        QinvA = Qinv @ A

        diag = []
        off_diag = []
        diag.append(scsp.lil_matrix(Q0inv - ATQinvA))
        for t in range(1, nts-1):
            diag.append(scsp.lil_matrix(sum(np.exp(C[i]*x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(n_neurons)) + ATQinvAplusQinv))
        diag.append(scsp.lil_matrix(Qinv + sum(np.exp(C[i]*x[-nld:] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(n_neurons))))

        for t in range(0, nts):
            off_diag.append(scsp.lil_matrix(-QinvA))

        h = scsp.block_diag(diag).tolil()
        od = scsp.block_diag(off_diag).tolil()
        h[nld:, :] += od[:-nld, :]
        h[:, nld:] += od.T[:, :-nld]
        return h.tocsc()
    return f


def jointloglikelihood(y, nsd, n_neurons, nts, mu, cov, Q, Q0, x0, A, u, B):
    def jointll(dC):

        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        # precompute for efficiency
        # q0inv = np.linalg.inv(Q0)
        # qinv = np.linalg.inv(Q)
        #
        # jll = sum(y[t*n_neurons:(t+1)*n_neurons].T @ C @ mu[t*nld:(t+1)*nld] + y[t*n_neurons:(t+1)*n_neurons].T @ d \
        #           - .5 * np.sum(np.exp(C @ mu[t*nld:(t+1)*nld] + .5 * np.diag(C @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C.T) + d)) \
        #           - .5 * mu[(t+1)*nld:(t+2)*nld].T @ qinv @ mu[(t+1)*nld:(t+2)*nld] + np.trace(qinv @ cov[(t+1)*nld:(t+2)*nld, (t+1)*nld:(t+2)*nld]) \
        #           + .5 * mu[(t+1)*nld:(t+2)*nld].T @ qinv @ A @ mu[t*nld:(t+1)*nld] + np.trace(qinv @ A @ cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld]) \
        #           + .5 * mu[(t+1)*nld:(t+2)*nld].T @ qinv @ B @ u[t*nsd:(t+1)*nsd] \
        #           + .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ mu[(t+1)*nld:(t+2)*nld] + np.trace(A @ qinv @ cov[t*nld:(t+1)*nld, (t+1)*nld:(t+2)*nld]) \
        #           - .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ A @ mu[t*nld:(t+1)*nld] + np.trace(A.T @ qinv @ A @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld]) \
        #           - .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ B @ u[t*nsd:(t+1)*nsd] \
        #           + .5 * u[t*nsd:(t+1)*nsd].T @ B.T @ qinv @ mu[t*nld:(t+1)*nld] \
        #           - .5 * u[t*nsd:(t+1)*nsd].T @ B.T @ qinv @ A @ mu[t*nld:(t+1)*nld] \
        #           - .5 * u[t*nsd:(t+1)*nsd].T @ B.T @ qinv @ B @ u[t*nsd:(t+1)*nsd]
        #           for t in range(nts-1)) \
        #     - .5 * np.log(np.linalg.det(Q0)) \
        #     - .5 * (nts-1) * np.log(np.linalg.det(Q)) \
        #     + .5 * np.trace(q0inv @ cov[0*nld:1*nld, 0*nld:1*nld]) \
        #     + .5 * (2 * mu[0*nld:1*nld].T - 2 * x0.T) @ q0inv @ (2 * mu[0*nld:1*nld] - 2 * x0)

        jll = sum(-y[t*n_neurons:(t+1)*n_neurons].T @ C @ mu[t*nld:(t+1)*nld] - y[t*n_neurons:(t+1)*n_neurons].T @ d \
                  + np.sum(np.exp(C @ mu[t*nld:(t+1)*nld] + .5 * np.diag(C @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C.T) + d))
                  for t in range(nts-1))
        return jll
    return jointll


def jllDerivative(n_neurons, nld, mu, cov, nts, y):
    def f(dC):
        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        djlld = sum(
                -y[t*n_neurons:(t+1)*n_neurons] \
                + np.exp(C @ mu[t*nld:(t+1)*nld] + d + .5 * np.diag(C @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C.T))
                for t in range(nts-1))

        djllC = np.array([sum(-y[t*n_neurons:(t+1)*n_neurons][i] * mu[t*nld:(t+1)*nld] +
                    np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) *
                    (mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                    for t in range(nts)) for i in range(n_neurons)])

        djlldC = np.empty(n_neurons + n_neurons*nld)
        for i in range(n_neurons):
            djlldC[i*(nld+1)] = djlld[i]
            djlldC[i*(nld+1) + 1:(i+1)*(nld+1)] = djllC[i]

        return djlldC
    return f


def jllHessian(n_neurons, nld, mu, cov, nts, y):
    def f(dC):
        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        blocks = []
        block = np.zeros((1 + nld)*(1 + nld)).reshape(-1, 1+nld)
        
        for i in range(n_neurons):
            block[0, 0] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) for t in range(nts))

            block[1, 1:] = sum((mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + \
                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) for t in range(nts))

            block[1:, 1] = block[1, 1:]

            block[1:, 1:] = sum((cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
                np.outer(mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i], (mu[t*nld:(t+1)*nld] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]).T)) *
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                for t in range(nts))

            blocks.append(scsp.lil_matrix(block))

        HJLL = scsp.block_diag(blocks)
        return HJLL.tocsc()
    return f


def laplace_approximation(f, df, hf, x):
    # use NR algorithm to compute minimum of log-likelihood
    x = nr_algo(f, df, hf, x)
    # negative inverse of Hessian is covariance matrix
    covariance = -splin.inv(hf(x)).toarray()
    return mu, covariance


if __name__ == "__main__":
    # load data
    print('loading data')
    nts = 100
    n_neurons = 300  # number of neurons
    nld = 5  # number of latent dimensions
    nsd = 4
    frameHz = 10  # frames per seconds
    data = scio.loadmat('data/compiled_dF033016.mat')
    y = data['behavdF'].flatten()[:nts*n_neurons]
    onset = np.array(data['onsetFrame'].T[0], np.int8)
    resptime = data['resptime'].T[0]
    correct = data['correct'][0]
    orient = np.array(data['orient'][0], np.int8)
    location = np.array((data['location'][0]+1)//2, np.int8)
    u = np.zeros((nts, nsd))
    for ot, rt, cor, ori, loc in zip(onset, resptime, correct, orient, location):
        # compute what u should be here
        u[int(ot):ot+int((rt+2.75+(4.85-2.75)*(1-cor))*frameHz)] = np.array([ori*loc, (1-ori)*loc, ori*(1-loc), (1-ori)*(1-loc)], np.int)
    u = u.flatten()

    print('variable initialization')
    # Initialize parameters to random values
    C = np.random.randn(n_neurons, nld)/10
    d = np.random.randn(n_neurons)/10
    x0 = np.random.randn(nld)/10
    A = np.random.rand(nld, nld)/10
    q0 = np.random.rand(nld, nld)
    q0 = np.dot(q0, q0.T)/10
    q = np.random.rand(nld, nld)
    q = np.dot(q, q.T)/10
    B = np.random.rand(nld, nsd)/10
    mu = np.random.rand(nld*nts)/10
    cov = np.random.rand(nld, nld)/10

    print('begin training')
    max_epochs = 5
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch))
        print('performing laplace approximation')
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        mu, cov = laplace_approximation(logposterior(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd),
                                        logposteriorderivative(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd, nld),
                                        logposteriorhessian(y, C, d, A, B, q, q0, x0, u, nts, n_neurons, nsd, nld),
                                        mu)

        print('assigning analytic expressions')
        # Use analytic expressions to compute parameters x0, Q, Q0, A, B
        x0 = mu[:nld]
        q0 = cov[:nld, :nld]

        A = sum(cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) -
                np.outer(B @ u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)) @ \
            np.linalg.inv(sum(cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
            np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)))


        B = sum(np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd]) - A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd])
                for t in range(nts-1)) @ np.linalg.inv(sum(np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd]) for t in range(nts-1)))

        q = (1/(nts-1))*sum(cov[(t+1)*nld:(t+2)*nld, (t+1)*nld:(t+2)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[(t+1)*nld:(t+2)*nld].T)
            - (cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T)) @ A.T
            - np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd].T) @ B.T
            - A @ (cov[t*nld:(t+1)*nld, (t+1)*nld:(t+2)*nld] + mu[t*nld:(t+1)*nld] @ mu[(t + 1)*nld:(t+2)*nld].T)
            + A @ (cov[t*nld:(t + 1)*nld, t*nld:(t + 1)*nld] + np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T)) @ A.T
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
        for i in range(n_neurons):
            dC.append(d[i])
            dC += list(C[i])
        dC = np.array(dC)

        dC = nr_algo(jointloglikelihood(y, nsd, n_neurons, nts, mu, cov, q, q0, x0, A, u, B),
                     jllDerivative(n_neurons, nld, mu, cov, nts, y),
                     jllHessian(n_neurons, nld, mu, cov, nts, y),
                     dC)

        for i in range(n_neurons):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]
