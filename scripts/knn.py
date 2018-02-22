import scipy.io as scio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

nsd = 4
frameHz = 10  # frames per seconds
data = scio.loadmat('../data/compiled_dF033016.mat')
y = data['behavdF'].T

onsetframe = data['onsetFrame'].T[0]
onsettime = np.array(data['onsetFrame'].T[0])/frameHz

resptime = data['resptime'][:, 0]
correct = data['correct'][0]

offsettime = onsettime + resptime + 2.75 + (4.85-2.75)*(1-correct)
offsetframe = (offsettime * frameHz).astype(np.int32)

# make a list of inputs (ca2data) with only data that is N time steps before and after stimuli offset
inputs = [[] for i in range(300)]
for off in offsetframe:
    for neuron in range(y.shape[1]):
        inputs[neuron] += list(y[neuron, off-10:off+10])
inputs = np.array(inputs)

numclusters = 5
y_pred = KMeans(n_clusters=numclusters)
y_pred.fit_predict(y[:, :])
print(y_pred.labels_)
yclusters = [[] for i in range(numclusters)]
for clus in range(0, numclusters):
    for y_, label in zip(y, y_pred.labels_):
        yclusters[label].append(y_)

yclusters = np.array(yclusters)
plt.imshow(yclusters[0], cmap='hot', interpolation='nearest')
plt.show()
