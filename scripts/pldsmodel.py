# coding: utf-8
import scipy.io as scio
import scipy.sparse as scsp
import numpy as np
from newton_method import nr_algo
from cholesky import computecov
import pickle
import time




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


def jointloglikelihood(y, nsd, nn, nts, mu, covd, Q, Q0, m0, A, u, B):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        jll = sum(-y[t*nn:(t+1)*nn] @ C @ mu[t*nld:(t+1)*nld] - y[t*nn:(t+1)*nn] @ d \
                  + sum(np.exp(C[i] @ mu[t*nld:(t+1)*nld] + .5 * (C[i] @ covd[t] @ C[i]) +
                    d[i]) for i in range(nn))
                    for t in range(nts-1))
        return jll
    return f


def jllDerivative(nn, nld, mu, covd, covod, nts, y):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        djlld = np.zeros(nn)
        for i in range(nn):
            djlld[i] = sum(-y[t*nn+i] \
                           + np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i]))
                           for t in range(nts-1))

        djllC = [sum(-y[t*nn+i] * mu[t*nld:(t+1)*nld] + \
                     np.exp(C[i] @ mu[t*nld:(t+1)*nld] + d[i] + .5 * (C[i] @ covd[t] @ C[i])) * \
                     (mu[t*nld:(t+1)*nld] + covd[t] @ C[i])
                     for t in range(nts-1)) for i in range(nn)]

        djlldC = np.empty(nn*(nld+1))
        for i in range(nn):
            djlldC[i*(nld+1)] = djlld[i]
            djlldC[i*(nld+1) + 1:(i+1)*(nld+1)] = djllC[i] + .5 * np.sign(C[i]) + .5 * C[i]

        return djlldC
    return f


def jllHessian(nn, nld, mu, covd, covod, nts, y):
    def f(dC):
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])

        blocks = []

        for i in range(nn):
            block = .5 * np.identity(1 + nld)
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


def laplace_approximation(f, df, hf, x, nts, nld):
    # use NR algorithm to compute minimum of log-likelihood
    x = nr_algo(f, df, hf, x)
    # inverse of Hessian is covariance matrix
    # covariance = blocktridiaginv(hf(x), nld, nts)
    # res = minimize(f, x, jac=df, hess=hf, method="Newton-CG")
    # covariance = blocktridiaginv(hf(res.x), nld, nts)
    covdiag, covoffdiag = computecov(hf(x).toarray(), nld, nts)
    return x, covdiag, covoffdiag


def runmodel(y, u, nts, nn, nld, nsd):

    output = []
    print('variable initialization')
    # Initialize parameters to random values
    C = np.random.rand(nn, nld) * 0.
    d = np.random.rand(nn) * .0
    m0 = np.random.rand(nld)
    A = np.random.rand(nld, nld) * 0.
    q0 = np.random.rand(nld, nld)
    q0 = q0 @ q0.T
    q = np.random.rand(nld, nld)
    q = q @ q.T
    B = np.random.rand(nld, nsd) * 0.
    mu = np.random.rand(nld*nts)
    previouslogpost = 1000000000
    previousjll = 1000000000

    # print('begin training')
    max_epochs = 50
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch))
        # print('performing laplace approximation')
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        # covd is covariance diagonal blocks. (nld, nld*nts)
        # covod is covariance off diabonal blocks. (nld, nld*(nts-1))
        mu, covd, covod = laplace_approximation(logposterior(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd),
                                        logposteriorderivative(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld),
                                        logposteriorhessian(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd, nld),
                                        mu, nts, nld)

        # print('assigning analytic expressions')
        # Use analytic expressions to compute parameters m0, Q, Q0, A, B

        m0 = mu[:nld]
        q0 = covd[0]

        A = sum(covod[t].T + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) -
            B @ np.outer(u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)) @ \
            np.linalg.inv(sum(covd[t] +
            np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)))

        B = sum(np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd]) - A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd])
                for t in range(nts-1)) @ np.linalg.inv(sum(np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd].T) for t in range(nts-1)))

        q = (1/(nts-1))*sum(np.outer((mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]),
                            (mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]).T) +
                            covd[t+1] - A @ covod[t] - covod[t].T @ A.T + A @ covd[t] @ A.T
                            for t in range(nts-1))

        # Second NR minimization to compute C, d (and in principle, D)
        # print('performing NR algorithm for parameters C, d')
        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = []
        for i in range(nn):
            dC.append(d[i])
            dC += list(C[i])
        dC = np.array(dC)

        dC = nr_algo(jointloglikelihood(y, nsd, nn, nts, mu, covd, q, q0, m0, A, u, B),
                     jllDerivative(nn, nld, mu, covd, covod, nts, y),
                     jllHessian(nn, nld, mu, covd, covod, nts, y),
                     dC)

        for i in range(nn):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        newjll = jointloglikelihood(y, nsd, nn, nts, mu, covd, q, q0, m0, A, u, B)(dC)
        newlogpost = logposterior(y, C, d, A, B, q, q0, m0, u, nts, nn, nsd)(mu)
        if abs(previousjll - newjll) < .5 and abs(previouslogpost - newlogpost) < .5:
            break
        else:
            previousjll = newjll
            previouslogpost = newlogpost

        output.append(outputdict = {'x':mu, 'A':A, 'B':B, 'C':C, 'd':d, 'Q':q, 'Q0':q0, 'm0':m0})
    return output

if __name__ == "__main__":
    # load data
    nts = 20000
    nn = 300  # number of neurons
    nld = 5  # number of latent dimensions
    nsd = 4
    frameHz = 10  # frames per seconds
    data = scio.loadmat('../data/compiled_dF033016.mat')
    y = data['behavdF'].flatten()[:nts*nn]
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

    output = runmodel(y, u, nts, nn, nld, nsd)

    outputfile = open("plds_output_timed.pickle","wb")
    pickle.dump(output, outputfile)
    outputfile.close()
