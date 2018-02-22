from sklearn import svm
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

nsd = 4
frameHz = 10  # frames per seconds
data = scio.loadmat('../data/compiled_dF033016.mat')
y = data['behavdF'].T
onsetframe = data['onsetFrame'].T[0]
onsettime = np.array(data['onsetFrame'].T[0])/frameHz

resptime = data['resptime'].T[0]
correct = data['correct'][0]

offsettime = onsettime + resptime + 2.75 + (4.85-2.75)*(1-correct)
offsetframe = (offsettime * frameHz).astype(np.int32)

orient = np.array(data['orient'][0], np.int8)
location = np.array((data['location'][0]+1)//2, np.int8)

# make a list of targets (stimuli)
stimuli = []
for onf, off, ori, loc in zip(onsetframe, offsetframe, orient, location):
    # compute what u should be here
    stimuli.append(1 * (1-ori)*loc + 2*ori*(1-loc) + 3*(1-ori)*(1-loc))

# make a list of inputs (ca2data) with only data that is N time steps before and after stimuli offset
inputs = []
for off in offsetframe:
    inputs.append(list(y[off-10:off+10]))
inputs = np.array(inputs)

# train an SVM for each time step

batchsize = 300
svms = []
correctness = np.zeros((20, 20))
for traintime in np.arange(0, 20):
    svms.append(svm.LinearSVC())
    svms[-1].fit(inputs[:batchsize, traintime], stimuli[:batchsize])
    for testtime in np.arange(0, 20):
        correctness[traintime, testtime] = np.mean(stimuli[batchsize:] == svms[-1].predict(inputs[batchsize:, testtime]))

plt.imshow(correctness, cmap='hot', interpolation='nearest', extent=[-10, 10, 10, -10])
plt.xlabel("test time")
plt.ylabel("train time")
plt.show()

np.save("../outputs/test_time_vs_train_time.npy", correctness)
np.save("../outputs/supportvectormachines.npy", svms)
