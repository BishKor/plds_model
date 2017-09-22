import scipy.io as scio
import numpy as np
import numdifftools as nd
from newton_method import nr_algo


def logposterior(y, C, d, A, q, q0, G, u, num_time_steps, num_neurons):
    """
    :param x: latent space
    :param y: neuron capture data
    :param C: latent space to neuron space transformation matrix
    :param d: mean firing rates
    :param A: deterministic component of the evolution state[t] to state[t+1]
    :param Q: covariance of the innovations that perturb the latent state at each time step
    :param Q0: covariance of the initial state x1 of each trial
    :param G: mapping of stimuli to "latent space stimuli"
    :param u: stimuli (4-D one-hot)
    :param T: number of time steps
    :param K: number of trials
    :param q: number of neurons
    :return: the log-posterior of eq.4 in Macke et al. 2011
    """

    def logpost(x):

        # first compute useful values
        q0inv = np.inv(q0)
        qinv = np.inv(q)

        term1 = 0
        term2 = 0
        term3 = 0
        constants = .5 * (num_time_steps-1) * np.log(q) + .5 * np.log(q0)

        for k in range(num_neurons):
            for t in range(num_time_steps):
                term1 += y[k, t].T @ (C @ x[k, t] + d) + np.sum(-np.exp(C @ x[k, t] + d))

            term2 += -np.dot(np.dot(x[k, 1] - x[0], q0inv), x[k, 1] - x[0])

            for t in range(T-1):
                alp = x[k, t+1] - np.dot(A, x[k, t]) - np.dot(G, u[k, t])
                term3 += np.dot(np.dot(alp.T, qinv), alp)

        return constants + term1 + term2 + term3

    return logpost


def jointloglikelihood(y, num_neurons, num_time_steps, C, mu, d, cov, Q, Q0, x, A, u, G):

    # precompute for efficiency
    q0inv = np.inv(Q0)
    qinv = np.inv(Q)

    jll = 0
    for k in range(num_neurons):
        for t in range(num_time_steps-1):
            jll += y[k, t].T @ C @ mu[t] + y[k, t].T @ d \
                   - .5 * np.exp(C @ mu[t] + .5 * C.T @ cov[t, t] @ C + d) \
                   - .5 * mu[1].T @ q0inv @ mu[1] + .5 * np.trace(q0inv @ cov[1, 1]) \
                   + .5 * mu[1].T @ q0inv @ x[0] \
                   + .5 * x[0].T @ q0inv @ mu[1] \
                   - .5 * x[0].T @ q0inv @ mu[1] \
                   - .5 * x[0].T @ q0inv @ x[1] \
                   - .5 * mu[t + 1] @ q0inv @ mu[t + 1] + np.trace(qinv @ cov[t + 1, t + 1]) \
                   + .5 * mu[t + 1].T * qinv @ A @ mu[t] + np.trace(qinv @ A @ cov[t.t + 1]) \
                   + .5 * mu[t + 1].T @ qinv @ G @ u[t] \
                   + .5 * mu[t].T @ A.T @ qinv @ mu[t + 1] + np.trace(A @ qinv @ cov[t + 1, t]) \
                   - .5 * mu[t].T @ A.T @ qinv @ A @ G @ u[t] + np.trace(A.T @ qinv @ A @ cov[t, t]) \
                   - .5 * mu[t].T @ A.T @ qinv @ G @ u[t] \
                   + .5 * u[t].T @ G.T @ qinv @ mu[t + 1] \
                   - .5 * u[t].T @ G.T @ qinv @ A @ mu[t + 1] \
                   - .5 * u[t].T @ G.T @ qinv @ G @ u[t]
        jll += -.5 * np.log(q0inv) - .5 * (T-1) * np.log(q0inv)
    return jll


def jllmake(y, num_neurons, num_time_steps, mu, cov, Q, Q0, x0, A, u, G):
    def jointloglikelihood(C, d):

        # precompute for efficiency
        q0inv = np.inv(Q0)
        qinv = np.inv(Q)

        jll = 0
        for k in range(num_neurons):
            for t in range(num_time_steps-1):
                jll += y[k, t].T @ C @ mu[t] + y[k, t].T @ d \
                       - .5 * np.exp(C @ mu[t] + .5 * C.T @ cov[t, t] @ C + d) \
                       - .5 * mu[1].T @ q0inv @ mu[1] + .5 * np.trace(q0inv @ cov[1, 1]) \
                       + .5 * mu[1].T @ q0inv @ x0 \
                       + .5 * x0.T @ q0inv @ mu[1] \
                       - .5 * x0.T @ q0inv @ mu[1] \
                       - .5 * x0.T @ q0inv @ x0 \
                       - .5 * mu[t + 1] @ q0inv @ mu[t + 1] + np.trace(qinv @ cov[t + 1, t + 1]) \
                       + .5 * mu[t + 1].T * qinv @ A @ mu[t] + np.trace(qinv @ A @ cov[t,t + 1]) \
                       + .5 * mu[t + 1].T @ qinv @ G @ u[t] \
                       + .5 * mu[t].T @ A.T @ qinv @ mu[t + 1] + np.trace(A @ qinv @ cov[t + 1, t]) \
                       - .5 * mu[t].T @ A.T @ qinv @ A @ G @ u[t] + np.trace(A.T @ qinv @ A @ cov[t, t]) \
                       - .5 * mu[t].T @ A.T @ qinv @ G @ u[t] \
                       + .5 * u[t].T @ G.T @ qinv @ mu[t + 1] \
                       - .5 * u[t].T @ G.T @ qinv @ A @ mu[t + 1] \
                       - .5 * u[t].T @ G.T @ qinv @ G @ u[t]
            jll += -.5 * np.log(q0inv) - .5 * (T-1) * np.log(q0inv)
        return jll
    return jointloglikelihood


def laplace_approximation(f, mu0):
    # use NR algorithm to compute minimum of jointloglikelihood
    mu = nr_algo(f, mu0)
    # negative inverse of Hessian is covariance matrix
    covariance = -np.inv(nd.Hessian(f)(mu0))
    return mu, covariance


if __name__ == "__main__":
    # load data
    data = scio.loadmat('data/compiled_dF033016.mat')
    neurons = data['behavdF']
    orient = np.array(data['orient'], np.int)
    location = np.array(data['location'] > 0, np.int)
    stimuli = np.array([o*l, (1-o)*l, o*(1-l), (1-o)*(1-l)] for o, l in zip(orient, location))
    num_time_steps = 30  # number of time steps
    num_trials = 372  # number of trials
    num_neurons = 300  # number of neurons

    # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
    mu, cov = laplace_approximation()

    # Use analytic expressions to compute parameters x0, Q, Q0, A, G

    # Create joint log posterior with determined parameters
    jll = jllmake(y, num_neurons, num_time_steps, mu, cov, Q, Q0, x0, A, u, G)

    # Second NR minimization to compute C, d (and in principle, D)
    C0
    d0
    C, d = nr_algo(jll, [C0, d0])
