# coding: utf-8
import scipy.io as scio
from math_plds import *
import numpy as np
from newton_method import nr_algo
import pickle
import datetime
# from memory_profiler import profile
import argparse
import os
from shutil import copyfile
import time

def runmodel(rundirpath, nld, y, u, nn, nsd, nts, onsettimes, offsettimes,
             Ainit=None, Binit=None, Cinit=None, dinit=None, Qinit=None, Q0init=None, m0init=None, muinit=None):
    
    copyfile('pldsmodel.py', rundirpath + '/pldsmodel.py')
    copyfile('newton_method.py', rundirpath + '/newton_method.py')
    copyfile('math_plds.py', rundirpath + '/math_plds.py')

    outputpath = rundirpath + '/output.pldsop'
    print(outputpath)
    print('variable initialization')
    # Initialize parameters to random values
    
    A = Ainit if Ainit is not None else np.zeros((nld, nld))
    
    if Binit is not None:
        B = Binit
    else:
        B = np.zeros((nld, nsd))
        if nld >= nsd:
            B[:nsd] += np.identity(nsd)
        else:
            B[:,:nld] += np.identity(nld)
    
    C = Cinit if Cinit is not None else ((np.std(np.log(y + .0001), axis=0) / np.sqrt(nld)) * np.random.randn(nld, nn)).T
    
    if dinit is not None:
        d = dinit
    else:
        ysilent = []
        yvec = y.reshape(-1, 300)
        for t in onsettimes:
            for i in range(1, 7):
                ysilent.append(yvec[t-i])
        d = np.log(np.mean(ysilent, axis=0))
        
    Q = Qinit if Qinit is not None else np.identity(nld) * .1
    Q0 = Q0init if Q0init is not None else np.identity(nld) * .1
    m0 = m0init if m0init is not None else np.zeros(nld)
    mu = muinit if muinit is not None else np.array([B @ u[t*nsd:(t+1)*nsd] + Q @ np.random.rand(nld) for t in range(nts)]).flatten()
    
    covd, covod = computecov(*logposteriorhessian(C, d, A, Q, Q0, nts, nn, nld)(mu, mode='asblocks'), nld, nts)
    output = {'nld': nld, 'nts': nts, 'nn': nn, 'nsd': nsd,
              'x': [mu], 'A': [A], 'B': [B], 'C': [C.copy()], 'd': [d.copy()], 'Q': [Q], 'Q0': [Q0], 'm0': [m0],
              'epochdurations':[],
              'terminationcriterion':[],
              'logposterior':[logposterior(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld)(mu)],
              'jointloglikelihood': [jointloglikelihood(nld, nn, nts, mu, covd, y)(np.insert(C, 0, d, axis=1).flatten())]}
    
    previoustheta = np.concatenate([A.flatten(), B.flatten(), C.flatten(), d.copy(), Q.flatten(), Q0.flatten(), m0])
    
    epochdurations = []
    # print('begin training')
    max_epochs = 20
    for epoch in range(max_epochs):
        starttime = time.clock()
        print('epoch {}'.format(epoch))
        # perform laplace approximation on log-posterior with Newton-Raphson optimization to find mean and covariance
        # covd is covariance diagonal blocks. (nld, nld*nts)
        # covod is covariance off diabonal blocks. (nld, nld*(nts-1))
        print('Performing Laplace approximation', flush=True)
        
        mu, covd, covod = laplace_approximation(logposterior(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld),
                                                logposteriorderivative(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld),
                                                logposteriorhessian(C, d, A, Q, Q0, nts, nn, nld),
                                                mu.copy(), nts, nld, nrmode='backtracking')

        print('Assigning analytic expressions')
        # Use analytic expressions to compute parameters m0, Q, Q0, A, B
        # m0 = mu[:nld]
        Q0 = covd[0]

        A = sum(covod[t].T + np.outer(mu[(t+1)*nld:(t+2)*nld], mu[t*nld:(t+1)*nld].T) - 
            B @ np.outer(u[t*nsd:(t+1)*nsd], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)) @ \
            np.linalg.inv(sum(covd[t] + np.outer(mu[t*nld:(t+1)*nld], mu[t*nld:(t+1)*nld].T) for t in range(nts - 1)))

        B = sum(np.outer(mu[(t+1)*nld:(t+2)*nld], u[t*nsd:(t+1)*nsd]) - A @ np.outer(mu[t*nld:(t+1)*nld], u[t*nsd:(t+1)*nsd])
                for t in range(nts-1)) @ np.linalg.inv(sum(np.outer(u[t*nsd:(t+1)*nsd], u[t*nsd:(t+1)*nsd].T) for t in range(nts-1)))

        Q = (1/(nts-1))*sum(np.outer((mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]),
                            (mu[(t+1)*nld:(t+2)*nld] - A @ mu[t*nld:(t+1)*nld]-B@u[t*nsd:(t+1)*nsd]).T) +
                            covd[t+1] - A @ covod[t] - covod[t].T @ A.T + A @ covd[t] @ A.T
                            for t in range(nts-1))

        # Second NR minimization to compute C, d (and in principle, D)
        print('performing NR algorithm for parameters C, d')
        # need to vectorize C for the purpose of gradient descent, thus making a vector (d[i], C[i]), i.e. hessian for
        # each neuron

        dC = nr_algo(jointloglikelihood(nld, nn, nts, mu, covd, y),
                     jllDerivative(nn, nld, mu, covd, nts, y),
                     jllHessian(nn, nld, mu, covd, nts),
                     np.insert(C, 0, d, axis=1).flatten(),
                     mode='backtracking')
        
        d = dC[::nld+1]
        C = np.array([dC[i*(nld+1)+1:(i+1)*(nld+1)] for i in range(nn)])
        
        # dC = nr_algo(Cjointloglikelihood(nld, nn, nts, mu, covd, y, d),
        #      CjllDerivative(nn, nld, mu, covd, nts, y, d),
        #      CjllHessian(nn, nld, mu, covd, nts, d),
        #      C.flatten())
        # C = np.array([dC[i*nld:(i+1)*nld] for i in range(nn)])

        
        print('Evaluating termination criterion')
        theta = np.concatenate([A.flatten(), B.flatten(), C.flatten(), d.copy(), Q.flatten(), Q0.flatten(), m0])
        criterion = np.max(np.abs(theta - previoustheta) / np.maximum(1e-4, np.abs(previoustheta)))
        
        output['x'].append(mu)
        output['A'].append(A)
        output['B'].append(B)
        output['C'].append(C)
        output['d'].append(d.copy())
        output['Q'].append(Q)
        output['Q0'].append(Q0)
        output['m0'].append(m0)
        output['epochdurations'].append(time.clock() - starttime)
        output['terminationcriterion'].append(criterion)
        output['logposterior'].append(logposterior(y, C, d, A, B, Q, Q0, m0, u, nts, nn, nsd, nld)(mu))
        output['jointloglikelihood'].append(jointloglikelihood(nld, nn, nts, mu, covd, y)(np.insert(C, 0, d, axis=1).flatten()))
        
        if criterion < 1e-3:
            break
        else:
            previoustheta = theta.copy()

        outputfile = open(outputpath + '.tmp', 'wb')
        pickle.dump(output, outputfile)
        outputfile.close()
        try:
            os.remove(outputpath)
        except OSError:
            pass

        os.rename(outputpath + '.tmp', outputpath)
    print('run complete')


def load_data(datapath, nts, initpath=None):
    data = pickle.load(open(datapath, 'rb'))
    if nts == None:
        nts = data['nts']

    onsetimes = data['onsetframes'][data['onsetframes'] <= nts]
    offsettimes = data['offsetframes'][data['offsetframes'] <= nts]

    if initpath == None:
        return data['y'][:nts].flatten(), data['u'][:nts].flatten(), data['nn'], data['nsd'], nts, onsetimes, offsettimes
    else:
        if datapath == initpath:
            initdata = data
        else:
            initdata = pickle.load(open(initpath, 'rb'))

        return initdata['nld'], data['y'][:nts].flatten(), data['u'][:nts].flatten(), data['nn'], data['nsd'], nts, onsetimes, offsettimes, initdata['A'][-1], initdata['B'][-1], initdata['C'][-1], initdata['d'][-1], initdata['Q'][-1], initdata['Q0'][-1], initdata['m0'][-1], initdata['x'][-1]


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('all', nargs='+')
    callargs = parser.parse_args()
    args = {'nts': None}
    for arg in callargs.all:
        args[arg.split('=')[0]] = arg.split('=')[1]
    now = datetime.datetime.now()
    # d_ln(<ysilent>)
    rundirpath = "../runs/{}_nld_{}_plds_{}-{}-{}-{}-{}-{}".format(args['datapath'].split('/')[-1], args['nld'], now.year, now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(rundirpath)
    
    if 'initpath' in args.keys():
        print('running model')
        runmodel(rundirpath, *load_data(args['datapath'], int(args['nts']), initpath=args['initpath']))
    else:
        print('running model')
        runmodel(rundirpath, int(args['nld']), *load_data(args['datapath'], int(args['nts'])))
    

