import scipy.io as scio
import numpy as np
import numdifftools as nd
from newton_method import nr_algo


def logposterior(y, C, d, A, q, q0, G, u, n_time_steps, n_trials, n_neurons):
    """
    :param y: neuron capture data
    :param C: latent space to neuron space transformation matrix
    :param d: mean firing rates
    :param A: deterministic component of the evolution state[t] to state[t+1]
    :param q: covariance of the innovations that perturb the latent state at each time step
    :param q0: covariance of the initial state x1 of each trial
    :param G: mapping of stimuli to "latent space stimuli"
    :param u: stimuli (4-D one-hot)
    :param n_time_steps: number of time steps
    :param n_trials: number of trials
    :param n_neurons: number of neurons
    :return: the log-posterior of eq.4 in Macke et al. 2011
    """

    def logpost(x):

        # first compute useful values
        q0inv = np.inv(q0)
        qinv = np.inv(q)

        term1 = 0
        term2 = 0
        term3 = 0
        constants = .5 * (n_time_steps-1) * np.log(q) + .5 * np.log(q0)

        for k in range(n_neurons):
            for t in range(n_time_steps):
                term1 += y[k, t].T @ (C @ x[k, t] + d) + np.sum(-np.exp(C @ x[k, t] + d))

            term2 += -np.dot(np.dot(x[k, 1] - x[0], q0inv), x[k, 1] - x[0])

            for t in range(n_time_steps-1):
                alp = x[k, t+1] - np.dot(A, x[k, t]) - np.dot(G, u[k, t])
                term3 += np.dot(np.dot(alp.T, qinv), alp)

        return constants + term1 + term2 + term3

    return logpost


def jointloglikelihood(y, n_neurons, n_time_steps, C, mu, d, cov, Q, Q0, x, A, u, G):

    # precompute for efficiency
    q0inv = np.inv(Q0)
    qinv = np.inv(Q)

    jll = 0
    for k in range(n_neurons):
        for t in range(n_time_steps-1):
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
        jll += -.5 * np.log(q0inv) - .5 * (n_time_steps-1) * np.log(q0inv)
    return jll


def jllmake(y, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, G):
    def jointloglikelihood(C, d):

        # precompute for efficiency
        q0inv = np.inv(Q0)
        qinv = np.inv(Q)

        jll = 0
        for k in range(n_neurons):
            for t in range(n_time_steps-1):
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
            jll += -.5 * np.log(q0inv) - .5 * (n_time_steps-1) * np.log(q0inv)
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
    frameHz = .1 # seconds
    data = scio.loadmat('data/compiled_dF033016.mat')
    y = data['behavdF']
    onset = data['onsetFrame']
    resptime = data['resptime']
    correct = data['correct']
    orient = np.array(data['orient'], np.int)
    location = np.array(data['location'] > 0, np.int)
    n_time_steps = 26187
    n_trials = 1  # number of trials
    n_neurons = 300  # number of neurons
    n_latent_dims = 8
    stimuli_dims = 4

    # create empty u
    u = np.zeros((n_latent_dims, n_time_steps))

    for ot, rt, cor, ori, loc in zip(onset, resptime, correct, orient, location):
        stimuli = np.array([ori*loc, (1-ori)*loc, ori*(1-loc), (1-ori)*(1-loc)])
        # compute what u should be here
        u[ot:ot+int((rt+2.75+(4.85-2.75)*(1-cor))*frameHz)] = np.array([1, 1, 1, 1])

    # Initialize parameters to random values
    C = np.zeros((n_neurons, n_latent_dims))
    d = np.zeros(n_neurons)
    x0 = np.zeros(n_latent_dims)
    A = np.zeros((n_latent_dims, n_latent_dims))
    Q0 = np.zeros((n_latent_dims, n_latent_dims))
    Q = np.zeros((n_latent_dims, n_latent_dims))
    G = np.zeros((n_latent_dims, stimuli_dims))
    B = np.zeros(n_latent_dims)

    max_epochs = 1000
    for epoch in max_epochs:
        # Create instance of log posterior
        f = logposterior(y, C, d, A, Q, Q0, G, u, n_time_steps, n_neurons)
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        mu, cov = laplace_approximation(f, (mu, cov))

        # Use analytic expressions to compute parameters x0, Q, Q0, A, G
        x0 = mu[0:n_latent_dims]
        q0 = cov[0:n_latent_dims, 0:n_latent_dims]

        A = sum(cov[(t+1)*n_latent_dims, t] + np.outer(mu[(t+1)*n_latent_dims], mu[t*n_latent_dims].T) for t in range(n_time_steps - 1)) @ \
            np.inv(sum(cov[t*n_latent_dims:(t+1)*n_latent_dims, t*n_latent_dims:(t+1)*n_latent_dims] +
                    np.dot(mu[t*n_latent_dims], mu[t*n_latent_dims].T) for t in range(n_time_steps - 1)))

        q = sum(cov[(t + 1)*n_latent_dims:(t + 2)*n_latent_dims, (t + 1)*n_latent_dims:(t + 2)*n_latent_dims] +
                    np.outer(mu[(t + 1)*n_latent_dims], mu[(t + 1)*n_latent_dims].T) -
                    (cov[(t + 1)*n_latent_dims:(t + 2)*n_latent_dims, t*n_latent_dims:(t + 1)*n_latent_dims] +
                    np.outer(mu[(t + 1)*n_latent_dims], mu[t*n_latent_dims])) @ A.T -
                    np.outer(mu[(t + 1)*n_latent_dims], u[t*n_latent_dims].T) @ B.T +
                    A @ (cov[t*n_latent_dims:(t + 1)*n_latent_dims, (t + 1)*n_latent_dims:(t + 2)*n_latent_dims] +
                         mu[t*n_latent_dims] @ mu[(t + 1)*n_latent_dims].T) +
                    A @ (cov[t*n_latent_dims:(t + 1)*n_latent_dims, t*n_latent_dims:(t + 1)*n_latent_dims] +
                         np.outer(mu[t*n_latent_dims], mu[t*n_latent_dims].T)) @ A.T +
                    A @ np.outer(mu[t*n_latent_dims], u[t*n_latent_dims].T) @ B.T -
                    B @ np.outer(u[t*n_latent_dims], mu[t*n_latent_dims].T) +
                    B @ np.outer(u[t*n_latent_dims], mu[t*n_latent_dims].T) @ A.T +
                    B @ np.outer(u[t*n_latent_dims], u[t*n_latent_dims]) @ B.T
                    for t in n_time_steps - 1)
        B = sum(np.outer(mu[(t+1)*n_latent_dims], u[t*n_latent_dims].T) - A @ np.outer(mu[t*n_latent_dims], u[t*n_latent_dims])
                for t in n_time_steps-1) @ np.inv(sum(np.outer(u[t*n_latent_dims], u[t*n_latent_dims])for t in n_time_steps-1))


        # Create instance of joint log posterior with determined parameters
        jll = jllmake(y, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, G)

        # Second NR minimization to compute C, d (and in principle, D)

        C, d = nr_algo(jll, (C, d))

