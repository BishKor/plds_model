# coding: utf-8

import scipy.io as scio
import scipy.sparse as scsp
import numpy as np
from newton_method import nr_algo


def kd(i,j):
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
        
        constants = .5 * (n_time_steps-1) * np.log(np.abs(np.linalg.det(q))) + .5 * np.log(np.abs(np.linalg.det(q0)))
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
             for i in range(n_neurons)) + A.T @ Qinv @ (x[t*nld:(t+1)*nld] - A @ x[t*nld:(t+1)*nld] - B @ u[t*n_stimuli_dims:(t+1)*n_stimuli_dims]) \
            - Qinv @ (x[t*nld:(t+1)*nld] - A @ x[(t-1)*nld:t*nld] - B @ u[(t-1)*n_stimuli_dims:t*n_stimuli_dims]) \
            - kd(t, 1) * (Q0inv @ (x[1*nld:(1+1)*nld] - x[0*nld:(0+1)*nld]))
        return df
    return f


def logposteriorHessian(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld):
    def f(x):
        h = np.zeros((n_time_steps*nld, n_time_steps*nld))
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        h[1*nld:(1+1)*nld, 1*nld:(1+1)*nld] += - Q0inv
        
        for t in range(1, n_time_steps-1):
            h[t*nld:(t+1)*nld, t*nld:(t+1)*nld] += -sum(np.exp(C[i]*x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T) 
                                          for i in range(n_neurons)) + A.T @ Qinv @ A - Qinv
            h[t*nld:(t+1)*nld, (t+1)*nld:(t+2)*nld] += - A.T @ Qinv
            h[t*nld:(t+1)*nld, (t-1)*nld:t*nld] += - Qinv @ A
            
        h[n_time_steps*nld:(n_time_steps+1)*nld, n_time_steps*nld:(n_time_steps+1)*nld] +=         -sum(np.exp(C[i]*x[n_time_steps*nld:(n_time_steps+1)*nld] + d[i]) * np.outer(C[i], C[i].T) for i in range(n_neurons))         + A.T @ Qinv @ A - Qinv
        h[n_time_steps*nld:(n_time_steps+1)*nld, (n_time_steps-1)*nld:n_time_steps*nld] += - Qinv @ A
        return h
    return f


def logposteriorhessianoptomized(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons, n_stimuli_dims, nld):
    def f(x):
        Qinv = np.linalg.inv(q)
        Q0inv = np.linalg.inv(q0)
        
        diag = []
        off_diag = []
        diag.append(scsp.coo_matrix(-Q0inv))
        ATQinvA = A.T @ Qinv @ A
        ATQinv = A.T @ Qinv
        QinvA = Qinv @ A
        ATQinvAminusQinv = ATQinvA - Qinv
        diag.append(- QinvA)
        for t in range(0, n_time_steps-1):
            
            diag.append(scsp.coo_matrix(-sum(np.exp(C[i]*x[t*nld:(t+1)*nld] + d[i]) * np.outer(C[i], C[i].T) 
                                          for i in range(n_neurons)) + ATQinvAminusQinv))
            off_diag.append(scsp.coo_matrix(-ATQinv))
        
        diag.append(scsp.coo_matrix(np.zeros_like(diag[0])))
        h = scsp.block_diag(diag)
        od = scsp.block_diag(off_diag)
        h[n_time_steps:, :] = diag[:ntimesteps, :]
        h[:, n_time_steps:] = diag[:, :ntimesteps]
        return h
    return f


def jointloglikelihood(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, B):
    def jointll(dC):

        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]

        # precompute for efficiency
        q0inv = np.linalg.inv(Q0)
        qinv = np.linalg.inv(Q)

        jll = sum(y[t*n_neurons:(t+1)*n_neurons].T @ C @ mu[t*nld:(t+1)*nld]                    + y[t*n_neurons:(t+1)*n_neurons].T @ d                    - .5 * np.exp(C @ mu[t*nld:(t+1)*nld] +
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


def laplace_approximation(f, df, Hf, mu):
    # use NR algorithm to compute minimum of log-likelihood
    mu = nr_algo(f, df, Hf, mu)

    # negative inverse of Hessian is covariance matrix
    covariance = -scsp.linalg.inv(H(mu))
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


def jllHessian(n_neurons, nld, mu, cov, n_time_steps, y):
    def f(dC):

        d = np.empty(n_neurons)
        C = np.empty((n_neurons, nld))

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]

        Hjll = np.zeros((n_neurons + n_neurons * nld, n_neurons + n_neurons * nld))
        
        
        
        for i in range(n_neurons):
            Hjll[i * (nld + 1), i * (nld + 1)] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                                                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                                                    for t in range(n_time_steps))

            Hjll[i * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)] =                 sum((mu[t] + cov[t, t] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) for t in range(n_time_steps))

            Hjll[i * (nld + 1) + 1:(i + 2) * (nld + 1), i * (nld + 1)] =                 Hjll[i * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)].T

            Hjll[i * (nld + 1) + 1:(i + 2) * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)] = np.array([
                sum(y[t][i] * cov[t, t] @ C[i] + mu[t] +
                np.outer(mu[t] + cov[t, t] @ C[i], mu[t] + cov[t, t] @ C[i])[i] +
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) *
                (u[t] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                for t in range(n_time_steps))
                ])
        return Hjll
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
    n_time_steps = 26187
    n_neurons = 300  # number of neurons
    nld = 8  # number of latent dimensions
    n_stimuli_dims = 4
    frameHz = 10  # frames per seconds
    data = scio.loadmat('data/compiled_dF033016.mat')
    y = data['behavdF'][0].flatten()
    onset = data['onsetFrame'].T[0]
    resptime = data['resptime'].T[0]
    correct = data['correct'][0]
    orient = np.array(data['orient'][0], np.int)
    location = (data['location'][0]+1)//2

    # create empty u
    u = np.zeros((n_time_steps, n_stimuli_dims))
    # set stimuli
    for ot, rt, cor, ori, loc in zip(onset, resptime, correct, orient, location):
        # compute what u should be here
        u[ot:ot+int((rt+2.75+(4.85-2.75)*(1-cor))*frameHz)] = \
            np.array([ori*loc, (1-ori)*loc, ori*(1-loc), (1-ori)*(1-loc)], np.int)

    # Initialize parameters to random values
    C = np.empty((n_neurons, nld))
    d = np.empty(n_neurons)
    x0 = np.empty(nld)
    A = np.empty((nld, nld))
    Q0 = np.empty((nld, nld))
    Q = np.empty((nld, nld))
    B = np.empty((nld, n_stimuli_dims))
    mu = np.empty(nld*n_time_steps)
    cov = np.empty((nld, nld))

    max_epochs = 1000
    for epoch in range(max_epochs):
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        mu, cov = laplace_approximation(logposterior(y, C, d, A, B, Q, Q0, u, n_time_steps, n_neurons), mu)

        # Use analytic expressions to compute parameters x0, Q, Q0, A, B
        x0 = mu[0:nld]
        q0 = cov[0:nld, 0:nld]

        A = sum(cov[(t+1)*nld, t] + np.outer(mu[(t+1)*nld], mu[t*nld].T) for t in range(n_time_steps - 1)) @ \
            np.linalg.inv(sum(cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] +
            mu[t*nld] @ mu[t*nld].T for t in range(n_time_steps - 1)))

        q = sum(cov[(t + 1)*nld:(t + 2)*nld, (t + 1)*nld:(t + 2)*nld] +
                np.outer(mu[(t + 1)*nld], mu[(t + 1)*nld].T) -
                (cov[(t + 1)*nld:(t + 2)*nld, t*nld:(t + 1)*nld] +
                np.outer(mu[(t + 1)*nld], mu[t*nld])) @ A.T -
                np.outer(mu[(t + 1)*nld], u[t*nld].T) @ B.T +
                A @ (cov[t*nld:(t + 1)*nld, (t + 1)*nld:(t + 2)*nld] +
                mu[t*nld] @ mu[(t + 1)*nld].T) +
                A @ (cov[t*nld:(t + 1)*nld, t*nld:(t + 1)*nld] +
                np.outer(mu[t*nld], mu[t*nld].T)) @ A.T +
                A @ np.outer(mu[t*nld], u[t].T) @ B.T -
                B @ np.outer(u[t], mu[t*nld].T) +
                B @ np.outer(u[t], mu[t*nld].T) @ A.T +
                B @ np.outer(u[t*nld], u[t]) @ B.T
                for t in range(n_time_steps - 1))

        B = sum(np.outer(mu[(t+1)*nld], u[t*nld].T) - A @ np.outer(mu[t*nld], u[t*nld])
                for t in n_time_steps-1) @ np.linalg.inv(sum(np.outer(u[t*nld], u[t*nld])for t in n_time_steps-1))

        # Create instance of joint log posterior with determined parameters
        jll = jointloglikelihood(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, B)

        # Second NR minimization to compute C, d (and in principle, D)

        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = np.array([np.concatenate(d[i], C[i]) for i in range(n_neurons)])

        dC = nr_algo(jll, jllDerivative(n_neurons, nld, mu, cov, n_time_steps, y),
                     jllHessian(n_neurons, nld, mu, cov, n_time_steps, y), dC)

        for i in range(n_neurons+1):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:i*(nld+1)]
