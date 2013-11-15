import numpy as np
import ghmm

def init_toy_hmm():
    account = ghmm.Float()   # emission domain of this model
    transMatr = [[0.6, 0.4, 0.0],[0.0, 0.6, 0.4],[0.0, 0.0, 1.0]]   # transition matrix
    pi = [1.0,0.0,0.0]

    stateParams = [
           [ [0.0, 0.0], [2.0, 0.0, 0.0, 1.0], 
             [0.0, 0.0], [1.0, 0.0, 0.0, 1.0],
             [0.6, 0.4] ],
           [ [0.0, 0.0], [1.0, 0.0, 0.0, 1.0], 
             [0.0, 0.0], [1.0, 0.0, 0.0, 1.0],
             [0.3, 0.7] ],
           [ [0.0, 0.0], [1.0, 0.0, 0.0, 1.0], 
             [0.0, 0.0], [1.0, 0.0, 0.0, 1.0],
             [0.3, 0.7] ],
           ]

    return ghmm.HMMFromMatrices(account,ghmm.MultivariateGaussianDistribution(account),
                                transMatr, stateParams, pi)

def init_hmm(nStates, nMix, dim):
    account = ghmm.Float()   # emission domain of this model
    transMatr = [[(0.5 if i==j or i==j-1 else 0.0) for j in range(nStates) ] for i in range(nStates)]
    transMatr[-1][-1] = 1.0

    pi = [1.0] + [0.0] * (nStates - 1)

    mu = [0.0] * dim
    sigma = [(1.0 if i % (dim+1) == 0 else 0.0) for i in range(dim*dim)]
    coeffs = [1.0 / nMix] * nMix
    stateParam = sum([[mu, sigma]] * nMix, []) + [coeffs]
    stParams = [stateParam for stInd in range(nStates)]

    return ghmm.HMMFromMatrices(account,ghmm.MultivariateGaussianDistribution(account), transMatr, stParams, pi)
        

class HMMClassifier(object):
    def __init__(self, nStates, nMix):
        self.nStates = nStates
        self.nMix = nMix
        self.modelsDict = {}
    def train(self, trData, labels):
        for phoneme in set(labels):
            hmm = self._init_hmm(self.nStates, self.nMix, dim=len(trData[0][0]))
            seq_set = ghmm.SequenceSet(ghmm.Float(), [sum(phSample, []) for phSample, lab in zip(trData, labels) if lab == phoneme])
            hmm.baumWelch(seq_set)
            self.modelsDict[phoneme] = hmm

    def _init_hmm(self, nStates, nMix, dim):
        account = ghmm.Float()   # emission domain of this model
        transMatr = [[(0.5 if i==j or i==j-1 else 0.0) for j in range(nStates) ] for i in range(nStates)]
        transMatr[-1][-1] = 1.0

        pi = [1.0] + [0.0] * (nStates - 1)

        mu = [0.0] * dim
        sigma = [(1.0 if i % (dim+1) == 0 else 0.0) for i in range(dim*dim)]
        coeffs = [1.0 / nMix] * nMix
        stateParam = sum([[mu, sigma]] * nMix, []) + [coeffs]
        stParams = [stateParam for stInd in range(nStates)]

        return ghmm.HMMFromMatrices(account,ghmm.MultivariateGaussianDistribution(account), transMatr, stParams, pi)
