# coding: utf-8

import scipy.io as scio
import scipy.sparse as scsp
import numpy as np
from newton_method import nr_algo


def kd(i, j):
    if i == j:
        return 1.
    else:
        return 0.


def logposterior(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims):
    """
    :param y: neuron capture data
    :param C: latent space to neuron space transformation matrix
    :param d: mean firing rates
    :param A: deterministic component of the evolution state[t] to state[t+1]
    :param q: covariance of the innovations that perturb the latent state at each time step
    :param q0: covariance of the initial state x1 of each trial
    :param B: mapping of stimuli to "latent space stimuli"
    :param u: stimuli (4-D one-hot)
    :param n_time_steps: number of time steps
    :param n_neurons: number of neurons
    :return: the log-posterior of eq.4 in Macke et al. 2011
    """

    def logpost(x):

        # first compute useful values
        q0inv = np.linalg.inv(q0)
        qinv = np.linalg.inv(q)

        constants = .5 * (n_time_steps-1) * np.log(np.linalg.det(q)) + .5 * np.log(np.linalg.det(q0))
        term1 = sum(y[t*n_neurons:(t+1)*n_neurons].T @ (C @ x[t*nld:(t+1)*nld] + d) -
                    np.sum(np.exp(C @ x[t*nld:(t+1)*nld] + d)) for t in range(n_time_steps-1))

        term2 = - .5 * (x[1*nld:(1+1)*nld] - x[0*nld:(0+1)*nld]).T @ q0inv @ (x[1*nld:(1+1)*nld] - x[0*nld:(0+1)*nld])

        term3 = - .5 * sum((x[(t+1)*nld:(t+2)*nld] -
                    A @ x[t*nld:(t+1)*nld] -
                    B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims]).T @ qinv @ (x[(t+1)*nld:(t+2)*nld] -
                    A @ x[t*nld:(t+1)*nld] -
                    B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims])
                    for t in range(n_time_steps - 1))

        return constants + term1 + term2 + term3

    return logpost


def logposteriorDerivative(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld):
    def f(x):
        df = np.empty_like(x)
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        for t in range(1, n_time_steps):
            df[t*nld:(t+1)*nld] = sum((y[t*n_neurons:(t+1)*n_neurons][i] - np.exp(C[i]@x[t*nld:(t+1)*nld] + d[i]))*C[i] 
             for i in range(n_neurons)) + A.T @ Qinv @ (x[t*nld:(t+1)*nld] - A @ x[t*nld:(t+1)*nld] -
                                                        B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims]) \
            - Qinv @ (x[t*nld:(t+1)*nld] - A @ x[(t-1)*nld:t*nld] - B @ u[(t-1)*n_stimuli_dims:t*n_stimuli_dims]) \
            - kd(t, 1) * (Q0inv @ (x[1*nld:(1+1)*nld] - x[0*nld:(0+1)*nld]))
        return df
    return f


def logposteriorhessianoptomized(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld):
    def f(x):
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        ATQinvA = A.T @ Qinv @ A
        ATQinv = - A.T @ Qinv
        ATQinvAminusQinv = - ATQinvA - Qinv

        diag = []
        off_diag = []
        diag.append(scsp.lil_matrix(- ATQinvA - Q0inv))
        for t in range(n_time_steps-2):
            diag.append(scsp.lil_matrix(-sum(np.exp(C[i]*x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T)
                                          for i in range(n_neurons)) + ATQinvAminusQinv))

        diag.append(scsp.lil_matrix(-Qinv))

        for t in range(0, n_time_steps):
            off_diag.append(scsp.lil_matrix(.5 * ATQinv))

        h = scsp.block_diag(diag).tolil()
        od = scsp.block_diag(off_diag).tolil()
        h[nld:, :] += od[:-nld, :]
        h[:, nld:] += od.T[:, :-nld]
        return h.tocsc()
    return f


def jointloglikelihood(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, B):
    def jointll(dC):

        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        # precompute for efficiency
        q0inv = np.linalg.inv(Q0)
        qinv = np.linalg.inv(Q)

        jll = sum(y[t*n_neurons:(t+1)*n_neurons].T @ C @ mu[t*nld:(t+1)*nld] + y[t*n_neurons:(t+1)*n_neurons].T @ d \
                  - .5 * np.exp(C @ mu[t*nld:(t+1)*nld] +
                    .5 * C.T @ cov[t*n_neurons:(t+1)*n_neurons, t*n_neurons:(t+1)*n_neurons] @ C + d) \
                   - .5 * mu[nld:2*nld].T @ q0inv @ mu[nld:2*nld] \
                   + .5 * np.trace(q0inv @ cov[nld:2*nld, 1*nld:2*nld]) \
                   + .5 * mu[nld:2*nld].T @ q0inv @ x0 \
                   + .5 * x0.T @ q0inv @ mu[nld:2*nld] \
                   - .5 * x0.T @ q0inv @ mu[nld:2*nld] \
                   - .5 * x0.T @ q0inv @ x0 \
                   - .5 * mu[(t+1)*nld:(t+2)*nld] @ q0inv @ mu[(t+1)*nld:(t+2)*nld] \
                   + np.trace(qinv @ cov[(t+1)*nld, (t+1)*nld]) \
                   + .5 * mu[(t+1)*nld:(t+2)*nld].T * qinv @ A @ mu[t*nld:(t+1)*nld] \
                   + np.trace(qinv @ A @ cov[t*nld:(t+1)*nld, (t+1)*nld:(t+2)*nld]) \
                   + .5 * mu[(t+1)*nld].T @ qinv @ B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims] \
                   + .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ mu[(t+1)*nld:(t+2)*nld] \
                   + np.trace(A @ qinv @ cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld]) \
                   - .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ A @ B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims] \
                   + np.trace(A.T @ qinv @ A @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld]) \
                   - .5 * mu[t*nld:(t+1)*nld].T @ A.T @ qinv @ B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims] \
                   + .5 * u[t*n_stimuli_dims:(t+1)*n_stimuli_dims].T @ B.T @ qinv @ mu[(t+1)*nld:(t+2)*nld] \
                   - .5 * u[t*n_stimuli_dims:(t+1)*n_stimuli_dims].T @ B.T @ qinv @ A @ mu[(t+1)*nld:(t+2)*nld] \
                   - .5 * u[t*n_stimuli_dims:(t+1)*n_stimuli_dims].T @ B.T @ qinv @ B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims] for t in range(n_time_steps-1)) \
              - .5 * np.log(np.abs(np.det(Q0))) - .5 * (n_time_steps-1) * np.log(np.abs(np.det(Q)))
        return jll
    return jointll


def laplace_approximation(f, df, hf, x):
    # use NR algorithm to compute minimum of log-likelihood
    x = nr_algo(f, df, hf, x)
    # negative inverse of Hessian is covariance matrix
    covariance = -scsp.linalg.inv(hf(x))
    return mu, covariance


def jllDerivative(n_neurons, nld, mu, cov, n_time_steps, y):
    def f(dC):
        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]

        djlld = sum(
                y[t] + np.exp(C @ mu[t*nld:(t+1)*nld] + d + .5 * np.diag(C.T @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C))
                for t in range(n_time_steps))

        djllC = np.array([
                sum(
                y[t][i] * cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i] + mu[t] +
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) *
                (u[t] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                for t in range(n_time_steps))
                for i in range(n_neurons)])

        djlldC = np.array([np.concatenate(djlld[i], djllC[i]) for i in range(n_neurons)])
        return djlldC
    return f


def jllHessianOptimized(n_neurons, nld, mu, cov, n_time_steps, y):
    def f(dC):
        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]

        blocks = []
        block = np.zeros((1 + nld)*(1 + nld)).reshape(-1, 1+nld)
        
        for i in range(n_neurons):
            block[0, 0] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                                                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                                                    for t in range(n_time_steps))

            block[1, 1:] = sum((mu[t] + cov[t, t] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) for t in range(n_time_steps))

            block[1:, 1] = block[1, 1:].T

            block[1:, 1:] = np.array([sum(y[t][i] * cov[t, t] @ C[i] + mu[t] +
                np.outer(mu[t] + cov[t, t] @ C[i], mu[t] + cov[t, t] @ C[i])[i] +
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) *
                (u[t] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                for t in range(n_time_steps))
                ])
        HJLL = scsp.block_diag(blocks)
        return HJLL
    return f


if __name__ == "__main__":
    # load data
    print('loading data')
    n_time_steps = 100
    n_neurons = 300  # number of neurons
    nld = 5  # number of latent dimensions
    n_stimuli_dims = 4
    frameHz = 10  # frames per seconds
    data = scio.loadmat('data/compiled_dF033016.mat')
    y = data['behavdF'].flatten()
    onset = data['onsetFrame'].T[0]
    resptime = data['resptime'].T[0]
    correct = data['correct'][0]
    orient = np.array(data['orient'][0], np.int)
    location = (data['location'][0]+1)//2
    u = np.zeros((n_time_steps, n_stimuli_dims))
    for ot, rt, cor, ori, loc in zip(onset, resptime, correct, orient, location):
        # compute what u should be here
        u[ot:ot+int((rt+2.75+(4.85-2.75)*(1-cor))*frameHz)] = \
            np.array([ori*loc, (1-ori)*loc, ori*(1-loc), (1-ori)*(1-loc)], np.int)
    u = u.flatten()

    print('variable initialization')
    # Initialize parameters to random values
    C = np.random.randn(n_neurons * nld).reshape(-1, nld)
    d = np.random.randn(n_neurons)
    x0 = np.random.randn(nld)
    A = np.random.randn(nld * nld).reshape(-1, nld)
    q0 = np.random.randn(nld)
    q0 = np.outer(q0, q0.T)
    q = np.random.randn(nld)
    q = np.outer(q, q.T)
    B = np.random.randn(nld * n_stimuli_dims).reshape(-1, n_stimuli_dims)
    mu = np.random.randn(nld*n_time_steps)
    cov = np.random.randn(nld * nld).reshape(-1, nld)

    print('begin training')
    max_epochs = 1000
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch))
        print('performing laplace approximation')
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        mu, cov = laplace_approximation(logposterior(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims),
                                        logposteriorDerivative(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld),
                                        logposteriorhessianoptomized(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld),
                                        mu)
        print('laplace approximation complete')

        print('assigning analytic expression')
        # Use analytic expressions to compute parameters x0, Q, Q0, A, B
        x0 = mu[0:nld]
        q0 = cov[0:nld, 0:nld]

        A = sum(cov[(t+1)*nld:(t+2)*nld, t*nld:(t+1)*nld] + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) for t in range(n_time_steps - 1)) @ \
            np.linalg.inv(sum(cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
            np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(n_time_steps - 1)))

        q = sum(cov[(t + 1)*nld:(t + 2)*nld, (t + 1)*nld:(t + 2)*nld] +
                np.outer(mu[(t + 1)*nld:(t + 2)*nld], mu[(t + 1)*nld:(t + 2)*nld].T) -
                (cov[(t + 1)*nld:(t + 2)*nld, t*nld:(t + 1)*nld] +
                np.outer(mu[(t + 1)*nld:(t + 2)*nld], mu[t*nld:(t + 1)*nld])) @ A.T -
                np.outer(mu[(t + 1)*nld:(t + 2)*nld], u[t*n_stimuli_dims:(t+1)*n_stimuli_dims].T) @ B.T +
                A @ (cov[t*nld:(t + 1)*nld, (t + 1)*nld:(t + 2)*nld] +
                mu[t*nld:(t+1)*nld] @ mu[(t + 1)*nld:(t+2)*nld].T) +
                A @ (cov[t*nld:(t + 1)*nld, t*nld:(t + 1)*nld] +
                np.outer(mu[t*nld:(t+1)*nld], mu[t*nld].T)) @ A.T +
                A @ np.outer(mu[t*nld:(t+1)*nld], u[t*n_stimuli_dims:(t+1)*n_stimuli_dims].T) @ B.T -
                B @ np.outer(u[t*n_stimuli_dims:(t+1)*n_stimuli_dims], mu[t*nld:(t+1)*nld].T) +
                B @ np.outer(u[t*n_stimuli_dims:(t+1)*n_stimuli_dims], mu[t*nld:(t+1)*nld].T) @ A.T +
                B @ np.outer(u[t*n_stimuli_dims:(t+1)*n_stimuli_dims], u[t*n_stimuli_dims:(t+1)*n_stimuli_dims]) @ B.T
                for t in range(n_time_steps - 1))

        B = sum(np.outer(mu[(t+1)*nld], u[t*nld].T) - A @ np.outer(mu[t*nld], u[t*nld])
                for t in n_time_steps-1) @ np.linalg.inv(sum(np.outer(u[t*nld], u[t*nld])for t in n_time_steps-1))

        print('creating instance of joint log likelihood')
        # Create instance of joint log posterior with determined parameters
        jll = jointloglikelihood(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, q, q0, x0, A, u, B)

        # Second NR minimization to compute C, d (and in principle, D)

        print('performing NR algorithm for parameters C, d')
        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = np.array([np.concatenate(d[i], C[i]) for i in range(n_neurons)])

        dC = nr_algo(jll, jllDerivative(n_neurons, nld, mu, cov, n_time_steps, y),
                     jllHessianOptimized(n_neurons, nld, mu, cov, n_time_steps, y), dC)

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]
