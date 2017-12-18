import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    pars = ['Q0', 'Q', 'C', 'd', 'A', 'x', 'm0']
    plt.subplot(len(pars), 1, len(pars))
    plt.title('')
    for i, p in enumerate(pars):
        print(p)
        plt.subplot(len(pars), 1, i+1)
        path = '../testmats/' + p + 'rsq.npy'
        f = np.load(path)
        plt.plot(np.arange(0, len(f)), f, 'k-')
        plt.ylabel(p)
    plt.xlabel('training epoch')
    plt.savefig('Losses scipy optimize.pdf')