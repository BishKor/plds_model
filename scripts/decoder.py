from sklearn import svm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_access import caimgdataset  
import argparse


def main(delay, span, N):
    data = caimgdataset(mode='standard')
    performance = np.zeros((N, 10+delay, 10+delay))
    for n in range(N):
        print(n)
        for traintime in np.arange(0, 10+delay):            
            for testtime in np.arange(0, 10+delay):
                trainX, trainY, testX, testY = data.gettrials(traintime, testtime, span)
                s = svm.LinearSVC()
                s.fit(trainX, trainY)
                performance[n, traintime, testtime] += np.mean(testY == s.predict(testX))
                
    avgperformance = np.mean(performance, axis=0)
    stddev = np.std(performance, axis=0)
    plt.imshow(avgperformance, cmap='hot', interpolation='nearest', extent=[-10, delay, delay, -10])
    plt.colorbar()
    plt.xlabel("test time")
    plt.ylabel("train time")
    plt.savefig('../outputs/standard_decoder_perf_delay={}_span={}.png'.format(delay, span))
    plt.clf() 
    plt.imshow(stddev, cmap='hot', interpolation='nearest', extent=[-10, delay, delay, -10])
    plt.colorbar()
    plt.xlabel("test time")
    plt.ylabel("train time")
    plt.savefig('../outputs/standard_decoder_error.png')
    plt.clf()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('delay')
    # args = parser.parse_args()
    # main(int(args.delay))
    main(30, 3, 10)