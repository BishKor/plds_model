import scipy.io as scio
import numpy as np
import numdifftools as nd
from newton_method import nr_algo


def logposterior(y, C, d, A, B, q, q0, u, n_time_steps, n_neurons):
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
        constants = .5 * (n_time_steps-1) * np.log(q) + .5 * np.log(q0)
        term1 = sum(y[t*n_neurons:(t+1)*n_neurons].T @ (C @ x[t*nld:(t+1)*nld] + d) +
                    np.sum(-np.exp(C @ x[t*nld:(t+1)*nld] + d)) for t in range(n_time_steps-1))
        term2 = - .5 * (x[1] - x[0]).T @ q0inv @ (x[1] - x[0])

        term3 = - .5 * sum((x[(t + 1) * nld:(t + 2) * nld] -
                    A @ x[t * nld:(t + 1) * nld] -
                    B @ u[t*nld:(t+1)*nld]).T @ qinv @
                    (x[(t + 1) * nld:(t + 2) * nld] -
                    A @ x[t * nld:(t + 1) * nld] -
                    B @ u[t*nld:(t+1)*nld])
                    for t in range(n_time_steps - 1))

        return constants + term1 + term2 + term3

    return logpost

#
# def jointloglikelihood(y, n_stimuli_dims, n_neurons, n_time_steps, C, mu, d, cov, Q, Q0, x, A, u, B):
#
#     # precompute for efficiency
#     q0inv = np.linalg.inv(Q0)
#     qinv = np.linalg.inv(Q)
#
#     jll = 0
#     for k in range(n_neurons):
#         for t in range(n_time_steps-1):
#             jll += y[k, t].T @ C @ mu[t] + y[k, t].T @ d \
#                    - .5 * np.exp(C @ mu[t] + .5 * C.T @ cov[t, t] @ C + d) \
#                    - .5 * mu[1].T @ q0inv @ mu[1] + .5 * np.trace(q0inv @ cov[1, 1]) \
#                    + .5 * mu[1].T @ q0inv @ x[0] \
#                    + .5 * x[0].T @ q0inv @ mu[1] \
#                    - .5 * x[0].T @ q0inv @ mu[1] \
#                    - .5 * x[0].T @ q0inv @ x[1] \
#                    - .5 * mu[t + 1] @ q0inv @ mu[t + 1] + np.trace(qinv @ cov[t + 1, t + 1]) \
#                    + .5 * mu[t + 1].T * qinv @ A @ mu[t] + np.trace(qinv @ A @ cov[t.t + 1]) \
#                    + .5 * mu[t + 1].T @ qinv @ B @ u[t] \
#                    + .5 * mu[t].T @ A.T @ qinv @ mu[t + 1] + np.trace(A @ qinv @ cov[t + 1, t]) \
#                    - .5 * mu[t].T @ A.T @ qinv @ A @ B @ u[t] + np.trace(A.T @ qinv @ A @ cov[t, t]) \
#                    - .5 * mu[t].T @ A.T @ qinv @ B @ u[t] \
#                    + .5 * u[t].T @ B.T @ qinv @ mu[t + 1] \
#                    - .5 * u[t].T @ B.T @ qinv @ A @ mu[t + 1] \
#                    - .5 * u[t].T @ B.T @ qinv @ B @ u[t]
#         jll += -.5 * np.log(q0inv) - .5 * (n_time_steps-1) * np.log(q0inv)
#     return jll


def jllmake(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, B):
    def jointloglikelihood(dC):

        d = dC[:n_neurons]
        C = dC[n_neurons:].reshape(-1, n_neurons)
        # precompute for efficiency
        q0inv = np.linalg.inv(Q0)
        qinv = np.linalg.inv(Q)

        jll = sum(y[t*n_neurons:(t+1)*n_neurons].T @ C @ mu[t*nld:(t+1)*nld] \
                   + y[t*n_neurons:(t+1)*n_neurons].T @ d \
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
              - .5 * np.log(q0inv) - .5 * (n_time_steps-1) * np.log(q0inv)

        return jll
    return jointloglikelihood


def laplace_approximation(f, mu):
    # use NR algorithm to compute minimum of log-likelihood
    df = nd.Gradient(f)
    H = nd.Hessian(f)
    # compute hessian of log-likelihood
    mu = nr_algo(f, mu, df, H)

    # negative inverse of Hessian is covariance matrix
    covariance = -np.linalg.inv(H(mu))
    return mu, covariance


def jllHessian():
    return None


def jllDerivative():
    return None

if __name__ == "__main__":
    # load data
    n_time_steps = 26187
    n_neurons = 300  # number of neurons
    nld = 8 # number of latent dimensions
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
                A @ np.outer(mu[t*nld], u[t*nld].T) @ B.T -
                B @ np.outer(u[t*nld], mu[t*nld].T) +
                B @ np.outer(u[t*nld], mu[t*nld].T) @ A.T +
                B @ np.outer(u[t*nld], u[t*nld]) @ B.T
                for t in range(n_time_steps - 1))

        B = sum(np.outer(mu[(t+1)*nld], u[t*nld].T) - A @ np.outer(mu[t*nld], u[t*nld])
                for t in n_time_steps-1) @ np.linalg.inv(sum(np.outer(u[t*nld], u[t*nld])for t in n_time_steps-1))

        # Create instance of joint log posterior with determined parameters
        jll = jllmake(y, n_stimuli_dims, n_neurons, n_time_steps, mu, cov, Q, Q0, x0, A, u, B)

        # Second NR minimization to compute C, d (and in principle, D)

        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = np.array([np.concatenate(d[i], C[i]) for i in range(n_neurons)])

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

        Hjll = np.zeros((n_neurons + n_neurons * nld, n_neurons + n_neurons * nld))

        for i in range(n_neurons):
            Hjll[i * (nld + 1), i * (nld + 1)] = sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                                                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                                                    for t in range(n_time_steps))

            Hjll[i * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)] = \
                sum((mu[t] + cov[t, t] @ C[i]) * np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] +
                    .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) for t in range(n_time_steps))

            Hjll[i * (nld + 1) + 1:(i + 2) * (nld + 1), i * (nld + 1)] = \
                Hjll[i * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)].T

            Hjll[i * (nld + 1) + 1:(i + 2) * (nld + 1), i * (nld + 1) + 1:(i + 2) * (nld + 1)] = np.array([
                sum(y[t][i] * cov[t, t] @ C[i] + mu[t] +
                np.outer(mu[t] + cov[t, t] @ C[i], mu[t] + cov[t, t] @ C[i])[i] +
                np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * C[i] @ cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i]) *
                (u[t] + cov[t*nld:(t+1)*nld, t*nld:(t+1)*nld] @ C[i])
                for t in range(n_time_steps))
                ])

        Hjll[n_neurons:, :n_neurons] = Hjll[:n_neurons, n_neurons:].T

        dC = nr_algo(jll, djlldC, Hjll, dC)

        # NEEDS TO BE FIXED !!!!!!!
        d = dC[:n_neurons]
        C = dC[n_neurons:].reshape(-1, n_neurons)
        # !!!!!!!!
