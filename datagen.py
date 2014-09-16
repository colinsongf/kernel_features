#! usr/bin/python

import sys, os, yaml, json
import numpy as np

vitProjPath = "/home/kuzaleks/Projects/NetBeansProjects/viterby_algorithm/src"
sfeProjPath = "/home/kuzaleks/Projects/NetBeansProjects/signal_feature_experiment/src"
sys.path.append(vitProjPath)
sys.path.append(sfeProjPath)

from param_file_routines import HTKParamPhonemeReader, SampleCollector
from labmanip import labs_dict_from_file, labs_dict_from_db
from kernelrout import all_samples
from functools import partial

def gen_circle_data(radAve, total):
    np.random.seed(0)
    np.random.seed(0)
    x = np.zeros((total, 2))
    theta = np.random.normal(0, np.pi, 50)
    r = np.random.normal(radAve, 0.1, 50)
    x[0:total, 0] = r * np.cos(theta)
    x[0:total, 1] = r * np.sin(theta)
    return x

def gen_train_data(clTotals):
    x = np.array([])
    y = []
    rads = iter([0, 2, 5])
    for clTot in clTotals:
        if x.any(): x = np.vstack((x, gen_circle_data(next(rads), clTot[1])))
        else: x = gen_circle_data(next(rads), clTot[1])
        y.extend([clTot[0]] * clTot[1])
    return x, y

def gen_test_data():
    tSamplesTotal = 50
    return gen_circle_data(4.9, tSamplesTotal)

def phoneme_dict(paramDBPath, recSysDir, labsfn="", phsfn='monophones_full.json', verbose=False):
    pfr = HTKParamPhonemeReader()
    f = open(os.path.join(recSysDir, phsfn))
    monophones = json.load(f)
    f.close()

    if verbose:
        print 'Before:', monophones
    monophones = [ph for ph in monophones if not ph == 'sil']
    if verbose:
        print 'After:', monophones

    if (labsfn):
        labsDictReader = partial(labs_dict_from_file, os.path.join(recSysDir, labsfn))
    else:
        labsDictReader = labs_dict_from_db

    labsDict = labsDictReader()

    samples = {}
    sc = SampleCollector(pfr)
    for ph in monophones:
        sc.store_corpus(ph, paramDBPath, labsDict)
        samples[ph] = sc.corpus[:]
    return samples

def get_kernel_data(mvaDir):
    """
    returns original kernel data 
    """
    with open(os.path.join(mvaDir, "kernelprof.json")) as f:
        kernelProf = json.load(f)
    return kernelProf

class KernelProfile(object):
    def __init__(self, median, kernelFuncType, **args):
        """
        """
        self.median = median
        self.kernelFuncType = kernelFuncType
        self.args = args
    def __repr__(self):
        return " median: {0}\n kernel: \
        {1}\n args: {2}\n".format(self.median,self.kernelFuncType, self.args)

class CorpusFilesProfile(object):
    def __init__(self, files, subCorpLen=0):
        """
        """
        self.files = files
        self.subCorpLen = subCorpLen

class ClassifierProfile(object):
    def __init__(self):
        """
        """
        pass

class HMMProfile(ClassifierProfile):
    def __init__(self, covtype, nStates=3, nMix=1):
        """
        """
        self.covtype = covtype
        self.nStates = nStates
        self.nMix = nMix


def wraped_exp_entry(phAcc, sigFeatures, dictorsTotal, corpus,
                     trProf, testProf, mvaProf, testType,
                     clId, phones, kernelProf=None):
    recResEntry = {}
    
    if kernelProf is not None:
        kernel = {}
        kernel['median'] = kernelProf.median
        kernel['kerntype'] = str(kernelProf.kernelFuncType)
        kernel['args'] = kernelProf.args
        recResEntry['kernel'] = kernel

    recResEntry['accuracy'] = phAcc
    
    recResEntry['features'] = sigFeatures
    recResEntry['dicts_total'] = dictorsTotal
    recResEntry['corpus'] = corpus
    
    recResEntry['trainvol'] = trProf.subCorpLen
    recResEntry['testvol'] = testProf.subCorpLen
    recResEntry['trainfiles'] = trProf.files
    recResEntry['testfiles'] = trProf.files
    recResEntry['mvafiles'] = mvaProf.files
    recResEntry['test_type'] = testType
    
    recResEntry['classifier_id'] = clId
    recResEntry['phonemes'] = phones

    return recResEntry
