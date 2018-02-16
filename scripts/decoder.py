from sklearn import svm
import scipy.io as scio
import numpy as np

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



# make a list of targets (stimuli)
# make a list of inputs (ca2data) with only data that is N time steps before and after stimuli offset
# train an SVM for each time step
# test each of those SVMs on each of the time steps (this will produce a 2D heatmap plot: train time vs test time)
# create visualization
