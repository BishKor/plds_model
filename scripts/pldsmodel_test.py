# coding: utf-8
from math_plds import *
import numpy as np
from newton_method import nr_algo
import pickle
import datetime


def runmodel(y, u, nts, nn, nld, nsd, actualparams):
    output = {'x': [], 'A': [], 'B': [], 'C': [], 'd': [], 'Q': [], 'Q0': [], 'm0': []}
    print('variable initialization')
    # Initialize parameters to random values

    # C = actualparams['C']
    # d = actualparams['d']
    # m0 = actualparams['m0']
    # A = actualparams['A']
    # Q0 = actualparams['Q0']
    # Q = actualparams['Q']
    # B = actualparams['B']
    # mu = actualparams['x']

    C = np.random.rand(nn, nld)
    d = np.random.rand(nn)
    m0 = np.random.rand(nld)
    A = np.random.rand(nld, nld) * .1
    Q0 = np.random.rand(nld, nld)
    Q0 = Q0 @ Q0.T
    Q = np.random.rand(nld, nld)
    Q = Q @ Q.T
    B = np.random.rand(nld, nsd) * .1
    mu = np.random.rand(nld*nts)

    tmpop = {'x': mu, 'A': A, 'B': B, 'C': C, 'd': d, 'Q': Q, 'Q0': Q0, 'm0': m0}
    for k in tmpop.keys():
        output[k].append(tmpop[k])

    previoustheta = np.concatenate([A.flatten(), B.flatten(), C.flatten(), d, Q.flatten(), Q0.flatten(), m0])

    # print('begin training')
    max_epochs = 20
    for epoch in range(max_epochs):
        print('epoch {}'.format(epoch))
        # print('performing laplace approximation')
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        # covd is covariance diagonal blocks. (nld, nld*nts)
        # covod is covariance off diabonal blocks. (nld, nld*(nts-1))
        mu, covd, covod = laplace_approximation(logposterior(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld),
                                                logposteriorderivative(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld),
                                                logposteriorhessian(C, d, A, Q, Q0, nts, nn, nld),
                                                mu, nts, nld)

        # Use analytic expressions to compute parameters m0, Q, Q0, A, B
        #
        m0 = mu[:nld]
        Q0 = covd[0]

        A = sum(covod[t].T + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) -
            B @ np.outer(u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)) @ \
            np.linalg.inv(sum(covd[t] + np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)))

        B = sum(np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd]) -
                A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd])
                for t in range(nts-1)) @ np.linalg.inv(sum(np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd].T)
                                                           for t in range(nts-1)))

        Q = (1/(nts-1))*sum(np.outer((mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]),
                            (mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]).T) +
                            covd[t+1] - A @ covod[t] - covod[t].T @ A.T + A @ covd[t] @ A.T
                            for t in range(nts-1))

        # Second NR minimization to compute C, d (and in principle, D)
        # print('performing NR algorithm for parameters C, d')
        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron
        dC = np.insert(C, 0, d, axis=1).flatten()

        dC = nr_algo(jointloglikelihood(nld, nn, nts, mu, covd, y),
                     jllDerivative(nn, nld, mu, covd, nts, y),
                     jllHessian(nn, nld, mu, covd, nts),
                     dC)

        for i in range(nn):
            d[i] = dC[i*(nld+1)]
            C[i] = dC[i*(nld+1) + 1:(i+1)*(nld+1)]

        tmpop = {'x': mu, 'A': A, 'B': B, 'C': C, 'd': d, 'Q': Q, 'Q0': Q0, 'm0': m0}
        for k in tmpop.keys():
            output[k].append(tmpop[k])
        theta = np.concatenate([A.flatten(), B.flatten(), C.flatten(), d, Q.flatten(), Q0.flatten(), m0])
        criterion = np.max(np.abs(theta - previoustheta)/np.maximum(1e-4, np.abs(previoustheta)))
        if criterion < 1e-5:
            break
        else:
            previoustheta = theta

    return output


def load_data():
    d = pickle.load(open('../testmats/testparamsanddata.pldsip', 'rb'))
    return d


if __name__ == "__main__":
    # load data
    vals = load_data()  # actual values of the parameters created in fake data gen
    
    output = runmodel(vals['y'], vals['u'], vals['nts'], vals['nn'], vals['nld'], vals['nsd'], vals)
    now = datetime.datetime.now()
    outputfilepath = "../testmats/all_free.pldsop"
    # outputfilepath = "../testmats/pldstestrun_{}-{}-{}-{}-{}.pldsop".format(now.year, now.month, now.day, now.hour,
    #                                                                          now.minute)
    outputfile = open(outputfilepath, "wb")
    pickle.dump(output, outputfile)
    outputfile.close()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def rsq(a, b):
        return np.mean(np.abs((a-b)**2)/np.maximum(1e-4, np.abs(b)**2))

    inf = pickle.load(open(outputfilepath, 'rb'))
    actual = pickle.load(open('../testmats/testparamsanddata.pldsip', 'rb'))
    pars = ['Q0', 'Q', 'C', 'd', 'A', 'B', 'x', 'm0']

    plt.subplot(len(pars), 1, len(pars))
    for i, p in enumerate(pars):
        plt.subplot(len(pars), 1, i+1)
        paramplot = []
        for a in inf[p]:
            paramplot.append(rsq(a, actual[p]))
        plt.plot(np.arange(0, len(paramplot)), paramplot, 'k-')
        plt.ylabel(p)
        print("Inferred Value for " + p)
        print(inf[p][-1])
        print("Actual Value for " + p)
        print(actual[p])
    plt.xlabel('training epoch')
    plt.savefig('../testmats/all_free.png')
