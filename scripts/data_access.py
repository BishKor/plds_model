import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA

class caimgdataset():
    def __init__(self, mode='standard'):
        
        self.data = scio.loadmat('../data/compiled_dF033016.mat')
        
        if mode == 'standard':    
            self.y = self.data['behavdF'].T
        elif mode == 'pca':
            pca = PCA(n_components=10)
            pca.fit(self.data['behavdF'].T)
            self.y = pca.transform(self.data['behavdF'].T)

        self.frameHz = 10  # frames per seconds
        self.onsetframe = self.data['onsetFrame'].T[0]
        self.onsettime = np.array(self.data['onsetFrame'].T[0]) / self.frameHz

        self.resptime = self.data['resptime'].T[0]
        self.correct = self.data['correct'][0]

        self.offsettime = self.onsettime + self.resptime + 2.75 + (4.85 - 2.75) * (1 - self.correct)
        self.offsetframe = (self.offsettime * self.frameHz).astype(np.int32)

        self.orientation = np.array(self.data['orient'][0], np.int8)
        self.location = np.array((self.data['location'][0] + 1) // 2, np.int8)

        # make a list of targets (stimuli)
        self.stimuli = 1 * (1 - self.orientation) * self.location + 2 * self.orientation * (1 - self.location) + 3 * (1 - self.orientation) * (1 - self.location)
        self.delaytimes = np.insert(self.onsetframe[1:], len(self.onsetframe)-1, self.y.shape[0]) - self.offsetframe

#         shuffledindices = np.random.permutation(self.stimuli.shape[0])
#         self.batchsize = int(.8 * self.stimuli.shape[0])
#         self.testtrialindices = shuffledindices[:self.batchsize]
#         self.traintrialindices = shuffledindices[self.batchsize:]

    
    def gettrials(self, traintime, testtime, span):
        # generate list of indices of trials whose following trial begins in time larger than delay time plus span time
        trainindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= traintime - 10 + span:
                trainindices.append(i)
        trainindices = np.array(trainindices)
        
        testindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= testtime - 10 + span:
                testindices.append(i)
        testindices = np.array(testindices)
        
        sharedindices = np.intersect1d(testindices, trainindices)
        if testtime > traintime:
            testindices = testindices[np.random.permutation(int(.2 * len(testindices)))]
            potentialtrainindices = np.setdiff1d(trainindices, testindices)
            trainindices = potentialtrainindices[np.random.permutation(int(.8 * len(potentialtrainindices)))]
        elif testtime < traintime:
            trainindices = trainindices[np.random.permutation(int(.8 * len(trainindices)))]
            potentialtestindices = np.setdiff1d(testindices, trainindices)
            testindices = potentialtestindices[np.random.permutation(int(.2* len(potentialtestindices)))]
        else:
            split = np.random.permutation(len(sharedindices))
            trainindices = sharedindices[split[:int(.8 * len(sharedindices))]]
            testindices = sharedindices[split[int(.8 * len(sharedindices)):]]
            
    
        traintrials = []
        trainstims = []
        for i in trainindices:
            for s in range(span):
                traintrials.append(list(self.y[self.offsetframe[i] - 10 + traintime + s]))
                trainstims.append(self.stimuli[i])
        
        testtrials = []
        teststims = []
        for i in testindices:
            for s in range(span):
                testtrials.append(list(self.y[self.offsetframe[i] - 10 + testtime + s]))
                teststims.append(self.stimuli[i])
        
        return np.array(traintrials), np.array(trainstims), np.array(testtrials), np.array(teststims)


    def gettraintrialsattime(self, time, span):
        # generate list of indices of trials whose following trial begins in time larger that delay time plus span time
        filteredindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= time - 10 + span:
                filteredindices.append(i)

        # make a list of inputs (ca2data) with only data that is N time steps before and after stimuli offset
        traintrials = []
        trainstims = []
        for i in filteredindices:
            if np.isin(i, self.traintrialindices):
                for s in range(span):
                    traintrials.append(list(self.y[self.offsetframe[i] - 10 + time + s]))
                    trainstims.append(self.stimuli[i])
        
        return np.array(traintrials), np.array(trainstims)
    
        
    def gettesttrialsattime(self, time, span):
        # generate list of indices of trials whose following trial begins in time larger that delay time plus span time
        filteredindices = []
        for i, t in enumerate(self.delaytimes):
            if t >= time - 10 + span:
                filteredindices.append(i)

        # make a list of inputs (ca2data) with only data that is N time steps before and after stimuli offset
        testtrials = []
        teststims = []
        for i in filteredindices:
            if np.isin(i, self.testtrialindices):
                for s in range(span):
                    testtrials.append(list(self.y[self.offsetframe[i] - 10 + time + s]))
                    teststims.append(self.stimuli[i])
                
        return np.array(testtrials), np.array(teststims)

    
    def trialsoftype(stimtype):
        return np.where(self.stimuli == stimtype)
    
    def gettrialaroundoffest(i, tbefore, tafter):
        return self.y[self.offsettime[i] - tbefore, tafter]
        
        
    def shufflesets(self):
        shuffledindices = np.random.permutation(self.stimuli.shape[0])
        self.testtrialindices = shuffledindices[:self.batchsize]
        self.traintrialindices = shuffledindices[self.batchsize:]