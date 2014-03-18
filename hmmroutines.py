import numpy as np
import ghmm, os

import sys
sys.path.append("/home/kuzaleks/Projects/NetBeansProjects/viterby_algorithm/src")
from hmmbuilder import HMMFromGHMMConverter, hmm_built_from, HmmFromGHMMBuilder, HmmFromHTKBuilder



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
        


class HMMInitializer(object):
    def __init__(self, nStates):
        self.nStates = nStates


class HHMSphericalInitializer(HMMInitializer):
    def __init__(self, nStates, nMix, dim):
        HMMInitializer.__init__(self, nStates)
        self.dim = dim
        self.nMix = nMix

    def get_hmm(self):
        account = ghmm.Float()   # emission domain of this model
        transMatr = [[(0.5 if i==j or i==j-1 else 0.0) for j in range(nStates) ] for i in range(nStates)]
        transMatr[-1][-1] = 1.0

        pi = [1.0] + [0.0] * (self.nStates - 1)

        mu = [0.0] * self.dim
        sigma = [1.0] * self.dim
        coeffs = [1.0 / self.nMix] * self.nMix
        stateParam = [mu, sigma, coeffs]
        stParams = [stateParam for stInd in range(self.nStates)]

        return ghmm.HMMFromMatrices(account,ghmm.MultivariateGaussianDistribution(account), transMatr, stParams, pi)

class HMMClassifier(object):
    def __init__(self, nStates, nMix):
        self.nStates = nStates
        self.nMix = nMix
        self.modelsDict = {}
        self.target = set()

    def train(self, trData, labels):
        self.target = set(labels)
        self.dim = len(trData[0][0])
        for phoneme in self.target:
            hmm = self._init_hmm(self.nStates, self.nMix, dim=len(trData[0][0]))
            seq_set = ghmm.SequenceSet(ghmm.Float(), [sum(phSample, []) for phSample, lab in zip(trData, labels) if lab == phoneme])
            hmm.baumWelch(seq_set)
            self.modelsDict[phoneme] = hmm

    def load(self, hmmDefFileList, pathToHmm='recsystem/hmm'):
        for hmmDeffn in hmmDefFileList:
            ph = os.path.basename(hmmDeffn)
            #print os.path.join(pathToHmm, hmmDeffn)
            self.modelsDict[ph] = hmm_built_from(HmmFromGHMMBuilder, os.path.join(pathToHmm, hmmDeffn))

    def refine_cov_matrix(self, threshold=0.5):
#        newSigma = [(threshold if i % (self.dim+1) == 0 else 0.0) for i in range(self.dim*self.dim)]
        
        for ph in self.modelsDict:
#            for stInd in range(self.modelsDict[ph].N):
#                mu, sigma = self.modelsDict[ph].getEmission(stInd, 0)
#                stateParam = [mu, newSigma]
#                self.modelsDict[ph].setEmission(stInd, 0, stateParam)
#            self.modelsDict[ph].normalize()
            self.modelsDict[ph] = HMMClassifier.reassigned_ghmm_object(self.modelsDict[ph], threshold)


    def predict(self, testData):
        seq_set = ghmm.SequenceSet(ghmm.Float(), [sum(phSeq, []) for phSeq in testData])
        res = []
        for seq in seq_set:
            loglikelihoods = [(model, self.modelsDict[model].loglikelihood(seq)) for model in self.modelsDict]
            res.append(max(loglikelihoods, key=lambda el: el[1])[0])
        return res

    def _init_hmm(self, nStates, nMix, dim):
        account = ghmm.Float()   # emission domain of this model
        transMatr = [[(0.5 if i==j or i==j-1 else 0.0) for j in range(nStates) ] for i in range(nStates)]
        transMatr[-1][-1] = 1.0

        pi = [1.0] + [0.0] * (nStates - 1)

        mu = [0.0] * dim
        sigma = [(4.0 if i % (dim+1) == 0 else 0.0) for i in range(dim*dim)]
        coeffs = [1.0 / nMix] * nMix
        stateParam = sum([[mu, sigma]] * nMix, []) + [coeffs]
        stParams = [stateParam for stInd in range(nStates)]

        return ghmm.HMMFromMatrices(account,ghmm.MultivariateGaussianDistribution(account), transMatr, stParams, pi)

    def reassigned_ghmm_object(genObj, threshold):
        B = []
        for stInd in range(genObj.N):
            mu, sigma = genObj.getEmission(stInd, 0)
            dim = len(mu)
            sigma = [(threshold if i % (dim+1) == 0 else 0.0) for i in range(dim*dim)]
            B.append([mu, sigma, [1.0]])
        A = [[genObj.getTransition(fr, to) for to in range(genObj.N)] for fr in range(genObj.N)]
        pi = [1.0] + [0.0] * (genObj.N - 1)
        return ghmm.HMMFromMatrices(ghmm.Float(), ghmm.MultivariateGaussianDistribution(ghmm.Float()), A, B, pi)

    reassigned_ghmm_object = staticmethod(reassigned_ghmm_object)
