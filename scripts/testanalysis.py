import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


def rsq(a, b):
    return 1-np.mean(np.abs(a-b)/np.maximum(1.0e-4, np.abs(b)))

inf = pickle.load(open('../testmats/allparamsfree.pldsop', 'rb'))
actual = pickle.load(open('../testmats/testparamsanddata.pldsip', 'rb'))
pars = ['Q0', 'Q', 'C', 'd', 'A', 'B', 'x', 'm0']

print("All Params free")
plt.subplot(len(pars), 1, len(pars))
for i, p in enumerate(pars):
    # plt.subplot(len(pars), 1, i+1)
    # paramplot = []
    # for a in inf[p]:
    #     paramplot.append(rsq(a, actual[p]))
    # plt.plot(np.arange(0, len(paramplot)), paramplot, 'k-')
    # plt.ylabel(p)
    print("Inferred Value for " + p)
    print(inf[p][-1])
    print("Actual Value for " + p)
    print(actual[p])
plt.xlabel('training epoch')
# plt.savefig('../testmats/Losses scipy optimize.png')
