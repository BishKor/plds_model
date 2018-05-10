import numpy as np
         
def trialresponses(activity, eventtime, precenter, postcenter):
    responses = np.zeros((eventtime.shape[0], precenter + postcenter, activity.shape[1]))
    valideventtime = []
    
    for t in eventtime:
        if t-precenter >= 0 and t+postcenter <= activity.shape[0]:
            valideventtime.append(t)
            
    for i, t in enumerate(valideventtime):
        responses[i] += activity[t-precenter:t+postcenter]

    return responses

def stimuligroupedtrialresponses(activity, stimuli, eventtime, precenter, postcenter):
    responsedict = {}
    stimtypes = np.unique(stimuli)
    for s in stimtypes:
        stimidx = np.where(stimuli == s)
        responsedict[s] = trialresponses(activity, eventtime[stimidx], precenter, postcenter)
    return responsedict

def stimuligroupedpsths(activity, stimuli, eventtime, precenter, postcenter):
    responsedict = {}
    stimtypes = np.unique(stimuli)
    for s in stimtypes:
        stimidx = np.where(stimuli == s)
        responsedict[s] = np.mean(trialresponses(activity, eventtime[stimidx], precenter, postcenter), axis=0)
    return responsedict

def responsezscore(activity):
    return (activity - np.mean(activity, axis=0))/np.std(activity, axis=0)