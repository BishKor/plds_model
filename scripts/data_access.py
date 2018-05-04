import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA
import pickle
from math import floor, ceil

class dataset():
    def __init__(self, mode='standard', target='stimuli', latentvarpath=None):
        
        self.data = scio.loadmat('../data/compiled_dF033016.mat')
        
        if mode == 'standard':    
            self.y = self.data['behavdF'].T
        elif mode == 'pca':
            pca = PCA(n_components=10)
            pca.fit(self.data['behavdF'].T)
            self.y = pca.transform(self.data['behavdF'].T)
        elif mode == 'latent':
            latentdata = pickle.load(open(latentvarpath, 'rb'))
            self.y = latentdata['x'][-1].reshape(-1, latentdata['nld'])

        self.frameHz = 10  # frames per seconds
        self.onsetframe = self.data['onsetFrame'].T[0]
        self.onsettime = np.array(self.data['onsetFrame'].T[0]) / self.frameHz

        self.resptime = self.data['resptime'].T[0]
        self.correct = self.data['correct'][0]

        self.offsettime = self.onsettime + self.resptime + 2.75 + (4.85 - 2.75) * (1 - self.correct)
        self.offsetframe = np.rint(self.offsettime * self.frameHz).astype(np.int32)

        self.orientation = np.array(self.data['orient'][0], np.int8)
        self.location = np.array((self.data['location'][0] + 1) // 2, np.int8)

        # make a list of targets (stimuli)
        if target == 'stimuli':
            self.stimuli = 1 * (1 - self.orientation) * self.location + \
                           2 * self.orientation * (1 - self.location) + \
                           3 * (1 - self.orientation) * (1 - self.location)
        elif target == 'orientation':
            self.stimuli = self.orientation.copy()
        elif target == 'location':
            self.stimuli = self.location.copy()

        self.delaytimes = np.insert(self.onsetframe[1:], len(self.onsetframe)-1, self.y.shape[0]) - self.offsetframe

    def gettrials(self, traintime, testtime, span=1):
        # generate list of indices of trials whose following trial begins in time larger than delay time plus span time
        trainindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= traintime + span - 1:
                trainindices.append(i)
        trainindices = np.array(trainindices)
        
        testindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= testtime + span - 1:
                testindices.append(i)
        testindices = np.array(testindices)
        
        sharedindices = np.intersect1d(testindices, trainindices)
        if testtime > traintime:
            testindices = testindices[np.random.permutation(len(testindices))[:int(.2 * len(testindices))]]
            potentialtrainindices = np.setdiff1d(trainindices, testindices)
            trainindices = potentialtrainindices[np.random.permutation(len(potentialtrainindices))]
        elif testtime < traintime:
            trainindices = trainindices[np.random.permutation(len(trainindices))[:int(.8 * len(trainindices))]]
            potentialtestindices = np.setdiff1d(testindices, trainindices)
            testindices = potentialtestindices[np.random.permutation(len(potentialtestindices))]
        else:
            split = np.random.permutation(len(sharedindices))
            trainindices = sharedindices[split[:int(.8 * len(sharedindices))]]
            testindices = sharedindices[split[-int(.2 * len(sharedindices)):]]

        traintrials = []
        trainstims = []
        for i in trainindices:
            traintrials.append(np.concatenate([self.y[self.offsetframe[i] + traintime + s] for s in range(span)]))
            trainstims.append(self.stimuli[i])
        
        testtrials = []
        teststims = []
        for i in testindices:
            testtrials.append(np.concatenate([self.y[self.offsetframe[i] + testtime + s] for s in range(span)]))
            teststims.append(self.stimuli[i])
        
        return np.array(traintrials), np.array(trainstims), np.array(testtrials), np.array(teststims)

    
    def getnumvalidtrials(self, time, span):
        nvalids = 0
        for t in self.delaytimes:
            if t >= time + span - 1:
                nvalids += 1
        return nvalids
    
    def as_dict():
        return {}