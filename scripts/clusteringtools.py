import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# normalizedstimgroups = {str(i):[] for i in range(nsd)}
# for stim, on in zip(stimuli, onsetframe):
#     normalizedstimgroups[str(stim)].append(ynormalized[int(on - 10): int(on + 10)])
# normalizedstimgroups = {str(i):np.array(normalizedstimgroups[str(i)]) for i in range(nsd)}

def getpsth(y, center, stimuli, nn, nsd, pre=10, post=40, baselines=None):
    ysilent = []
    for t in center:
        for i in range(1, pre):
            ysilent.append(y[t-i])
    baselines = np.mean(ysilent, axis=0)

    normalizedstimgroups = {str(i):[] for i in range(nsd)}
    for stim, t in zip(stimuli, center):
        normalizedstimgroups[str(stim)].append(y[int(t - pre): int(t + post)])
    normalizedstimgroups = {str(i):np.array(normalizedstimgroups[str(i)]) for i in range(nsd)}

    psthmeans = np.array([np.mean(normalizedstimgroups[str(j)], axis=0) for j in range(nsd)])
    # psthmaxes = np.array([np.max(np.mean(normalizedstimgroups[str(j)], axis=0), axis=0) for j in range(4)])
    # psthmaxdevs = np.array([np.max(np.abs(np.mean(normalizedstimgroups[str(j)], axis=0) - baselines), axis=0) for j in range(4)])
    psthmaxes = np.array([np.max(np.abs(np.mean(normalizedstimgroups[str(j)], axis=0) - baselines), axis=0) for j in range(nsd)])
    psthvectors = np.zeros(shape=(nn, nsd * (pre + post)))
    psthorders = np.argsort(-psthmaxes, axis=0).T
    for i in range(nn):
        j=0
        for o in psthorders[i]:
            psthvectors[i, (pre + post)*j:(pre + post)*(j+1)] += psthmeans[o].T[i]
            j += 1

    for i in range(len(psthvectors)):
        psthvectors[i] /= np.max(psthvectors[i])
    return psthvectors


def getnormalizedpsthstimsortvectors(y, center, stimuli, nn, nsd, baselines, pre=10, post=40, normalization='max'):
    normalizedstimgroups = {str(i):[] for i in range(nsd)}
    for stim, t in zip(stimuli, center):
        normalizedstimgroups[str(stim)].append(y[int(t - pre): int(t + post)])
    normalizedstimgroups = {str(i):np.array(normalizedstimgroups[str(i)]) for i in range(nsd)}

    psthmeans = np.array([np.mean(normalizedstimgroups[str(j)], axis=0) for j in range(nsd)])
    # psthmaxes = np.array([np.max(np.mean(normalizedstimgroups[str(j)], axis=0), axis=0) for j in range(4)])
    # psthmaxdevs = np.array([np.max(np.abs(np.mean(normalizedstimgroups[str(j)], axis=0) - baselines), axis=0) for j in range(4)])
    psthmaxes = np.array([np.max(np.abs(np.mean(normalizedstimgroups[str(j)], axis=0) - baselines), axis=0) for j in range(nsd)])
    psthvectors = np.zeros(shape=(nn, nsd * (pre + post)))
    psthorders = np.argsort(-psthmaxes, axis=0).T
    for i in range(nn):
        j=0
        for o in psthorders[i]:
            psthvectors[i, (pre + post)*j:(pre + post)*(j+1)] += psthmeans[o].T[i]
            j += 1

    # for i in range(len(psthvectors)):
    #     psthvectors[i] /= np.max(psthvectors[i])

    psthlocmaxes = np.array([np.max(psthmaxes[[0, 1]], axis=0), np.max(psthmaxes[[2, 3]], axis=0)])
    psthorimaxes = np.array([np.max(psthmaxes[[0, 2]], axis=0), np.max(psthmaxes[[1, 3]], axis=0)])
    psthlocorders = np.argsort(-psthlocmaxes, axis=0).T
    psthoriorders = np.argsort(-psthorimaxes, axis=0).T
    stimulilookuptable = np.array([[0, 1], [2, 3]])
    normalizedpsthstimsortvectors = np.zeros(shape=(nn, nsd*(pre + post)))
    psthstimsortvectors = np.zeros(shape=(nn, nsd*(pre + post)))

    for i in range(300):
        psthstimsortvectors[i] += np.concatenate([
                                          psthmeans[stimulilookuptable[psthlocorders[i, 0], psthoriorders[i, 0]]].T[i],
                                          psthmeans[stimulilookuptable[psthlocorders[i, 0], psthoriorders[i, 1]]].T[i],                           
                                          psthmeans[stimulilookuptable[psthlocorders[i, 1], psthoriorders[i, 0]]].T[i],                            
                                          psthmeans[stimulilookuptable[psthlocorders[i, 1], psthoriorders[i, 1]]].T[i]                          
                                          ])
        if normalization == 'max':
            normalizedpsthstimsortvectors[i] = psthstimsortvectors[i]/np.max(psthstimsortvectors[i])
        elif normalization == 'z score':
            normalizedpsthstimsortvectors[i] = (psthstimsortvectors[i] - np.mean(psthstimsortvectors[i]))/np.std(psthstimsortvectors[i])
        
    return normalizedpsthstimsortvectors


def plotclusters(normalizedpsthstimsortvectors, nsd, nclusters=4, pre=10, post=40, title=None, inspectneurons=[]):
    sortedstimkmeans = KMeans(n_clusters=nclusters).fit(normalizedpsthstimsortvectors)
    stimsortedkmeanpredictions = sortedstimkmeans.predict(normalizedpsthstimsortvectors)

    plt.figure(figsize=(13, 14), dpi= 80, facecolor='w', edgecolor='w')
    plt.suptitle(title)
    plt.subplot(nclusters, 1, 1)
    clustermeans = [[] for i in range(nclusters)]
    for i in range(len(normalizedpsthstimsortvectors)):
        clustermeans[stimsortedkmeanpredictions[i]].append(normalizedpsthstimsortvectors[i])
        plt.subplot(nclusters, 1, stimsortedkmeanpredictions[i]+1)
        plt.plot(normalizedpsthstimsortvectors[i], 'r')

    for i in inspectneurons:
        plt.subplot(nclusters, 1, stimsortedkmeanpredictions[i]+1)
        plt.plot(normalizedpsthstimsortvectors[i], 'g', linewidth=2)
        
    for i in range(nclusters):
        plt.subplot(nclusters, 1, i+1)
        plt.plot(np.mean(clustermeans[i], axis=0), 'k', linewidth=2)
        lims = plt.ylim()
        for j in range(1, nsd):
            plt.plot([j*(pre + post), j*(pre + post)], lims, 'k--')
        for j in range(nsd):
            plt.plot([pre + j* (pre + post), pre + j*(pre + post)], lims, 'k-')
            
    plt.show()
    
# Hierarchical Clustering
def heirarchicalclustering(y, center, stimuli, nn, nsd, baselines, nclusters=4, pre=10, post=40, inspectneurons=[]):
    normalizedpsthstimsortvectors = getnormalizedpsthstimsortvectors(y, center, stimuli, nn, nsd, baselines, pre=pre, post=post)
    Z = linkage(normalizedpsthstimsortvectors, 'ward')
    clusters = [fcluster(Z, nclusters, criterion='maxclust') for nclusters in range(1, nclusters+1)]
    changenumber = None
    for c in range(len(clusters[:-1])):
        clusterinds = []
        for cluster in clusters[c]:
            if not cluster in clusterinds:
                clusterinds.append(cluster)
        clusterinds = sorted(clusterinds)

        for changenumber in range(1, c+2):
    #         print('changenumber = {}'.format(changenumber))
            if [n for n,x in enumerate(clusters[c]) if x==changenumber] != [n for n,x in enumerate(clusters[c+1]) if x==changenumber]:
                break

        plt.figure(figsize=(13, 3 * len(clusterinds)), dpi= 80, facecolor='w', edgecolor='k')
        clustermeans = [[] for i in range(len(clusterinds))]
        changeclustermeans = [[[] for i in range(len(clusterinds))] for _ in range(2)]
        for i in range(nn):
            if clusters[c][i] <= changenumber:
                adjustedfuturenumber = clusters[c+1][i]
            else:
                adjustedfuturenumber = clusters[c+1][i] - 1

            if clusters[c][i] == adjustedfuturenumber:
                color = 'r'
                clustermeans[clusterinds.index(clusters[c][i])].append(normalizedpsthstimsortvectors[i])
                changeclustermeans[0][clusterinds.index(clusters[c][i])].append(normalizedpsthstimsortvectors[i])
            else:
                color = 'b'
                clustermeans[clusterinds.index(clusters[c][i])].append(normalizedpsthstimsortvectors[i])
                changeclustermeans[1][clusterinds.index(clusters[c][i])].append(normalizedpsthstimsortvectors[i])

            plt.subplot(np.max(clusters[c]), 1, clusterinds.index(clusters[c][i])+1)
            plt.plot(normalizedpsthstimsortvectors[i], color)

        for i in inspectneurons:
            plt.subplot(np.max(clusters[c]), 1, clusterinds.index(clusters[c][i])+1)
            plt.plot(normalizedpsthstimsortvectors[i], 'g', linewidth=2)

        for i in range(len(clusterinds)):
            plt.subplot(len(clusterinds), 1, i+1)
            plt.gca().set_title('Cluster {}'.format(clusterinds[i]))
            plt.plot(np.mean(clustermeans[i], axis=0), 'k', linewidth=2)
            plt.plot(np.mean(changeclustermeans[0][i], axis=0), 'k--', linewidth=2)
            plt.plot(np.mean(changeclustermeans[1][i], axis=0), 'k:', linewidth=2)
            lims = plt.ylim()
            for j in range(1, nsd):
                plt.plot([j*(pre + post), j*(pre + post)], lims, 'k-')
            for j in range(nsd):
                plt.plot([pre + j * (pre + post), pre + j * (pre + post)], lims, 'k--')
        
    for c in range(1):
        clusterinds = []
        for cluster in clusters[-1]:
            if not cluster in clusterinds:
                clusterinds.append(cluster)
        clusterinds = sorted(clusterinds)

        plt.figure(figsize=(13, 3 * len(clusterinds)), dpi= 80, facecolor='w', edgecolor='k')
        clustermeans = [[] for i in range(len(clusterinds))]
        for i in range(nn):
            clustermeans[clusterinds.index(clusters[-1][i])].append(normalizedpsthstimsortvectors[i])
            plt.subplot(np.max(clusters[-1]), 1, clusterinds.index(clusters[-1][i])+1)
            plt.plot(normalizedpsthstimsortvectors[i], 'r')

        for i in inspectneurons:
            plt.subplot(np.max(clusters[-1]), 1, clusterinds.index(clusters[-1][i])+1)
            plt.plot(normalizedpsthstimsortvectors[i], 'g', linewidth=2)

        for i in range(len(clusterinds)):
            plt.subplot(len(clusterinds), 1, i+1)
            plt.gca().set_title('Cluster {}'.format(clusterinds[i]))
            plt.plot(np.mean(clustermeans[i], axis=0), 'k', linewidth=2)
            lims = plt.ylim()
            for j in range(1, nsd):
                plt.plot([j*(pre + post), j*(pre + post)], lims, 'k-')
            for j in range(nsd):
                plt.plot([pre + j * (pre + post), pre + j * (pre + post)], lims, 'k--')
        plt.show()

        
def dprimearray(activity, stimuli, center, front=-10, end=40):
    dprimearray = np.zeros((activity.shape[1], post-pre))
    for neuronindex, neuron in enumerate(activity.T):
        for t in range(pre, post):
            up = []
            down = []
            for stimulus, time in zip(stimuli, center):
                if stimulus == 1:
                    up.append(neuron[time + t])
                elif stimulus == -1:
                    down.append(neuron[time + t])
            dprimearray[i, t-pre] = dprime(up, down)
    return dprimearray


def dprime(distA, distB):
    return (np.mean(distA) - np.mean(distB))/np.sqrt(.5 * (np.std(distA)**2 + np.std(distB)**2))