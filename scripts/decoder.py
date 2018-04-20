from sklearn import svm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_access import dataset
import argparse


def decoder(precenter, postcenter, span, N, mode='standard', latentvarpath=None):
    data = dataset(mode=mode, latentvarpath=latentvarpath)
    performance = np.zeros((N, precenter+postcenter, precenter+postcenter))
    weights=np.zeros((precenter+postcenter, data.y.shape[1]))
    for n in range(N):
        print('Iter {}'.format(n+1))
        for traintime in np.arange(-precenter, postcenter):
            tweights = []
            for testtime in np.arange(-precenter, postcenter):
                trainX, trainY, testX, testY = data.gettrials(traintime, testtime, span)                
                s = svm.LinearSVC()
                s.fit(trainX, trainY)
                tweights.append(s.coef_)
                performance[n, traintime+precenter, testtime+precenter] += np.mean(testY == s.predict(testX))
            # print(np.mean(tweights, axis=0))
            weights[traintime+precenter] += np.mean(tweights, axis=0)[0]
    weights /= N
    avgperformance = np.mean(performance, axis=0)
    stddev = np.std(performance, axis=0)
    nvalid = np.array([data.getnumvalidtrials(t, span) for t in range(-precenter, postcenter)])
    
    return performance, avgperformance, stddev, nvalid, weights

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('delay')
    # args = parser.parse_args()
    # main(int(args.delay))
    precenter = 10
    postcenter = 40
    span = 1
    N = 5    
    performance, avgperformance, stddev, nvalids = decoder(precenter, postcenter, span, N, mode='standard')
    plt.figure(figsize=(6, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(avgperformance, cmap='hot', interpolation='nearest', extent=[-precenter, postcenter, postcenter, -precenter])
    plt.colorbar()
    plt.xlabel("test time")
    plt.ylabel("train time")
    plt.show()
    # plt.savefig('../outputs/plot_for_pres_{}_decoder_perf_delay={}_span={}.png'.format(mode, postcenter, span))
    # plt.clf() 
    plt.figure(figsize=(6, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.imshow(stddev, cmap='hot', interpolation='nearest', extent=[-precenter, postcenter, postcenter, -precenter])
    plt.colorbar()
    plt.xlabel("test time")
    plt.ylabel("train time")
    plt.show()
    # plt.savefig('../outputs/plot_for_pres_{}_decoder_error_delay={}_span={}.png'.format(mode, postcenter, span))
    # plt.clf()
    plt.figure(figsize=(6, 6), dpi= 80, facecolor='w', edgecolor='k')
    plt.xlabel("test time")
    plt.ylabel("train time")
    plt.plot(np.diag(avgperformance))
    # plt.savefig('../outputs/plot_for_pres_{}_decoder_diag_perf_delay={}_span={}.png'.format(mode, postcenter, span))
    plt.show()