#! usr/bin/python

import sys, os, yaml
import numpy as np

sys.path.append("/home/kuzaleks/Projects/NetBeansProjects/viterby_algorithm/src")
from param_file_routines import HTKParamPhonemeReader, SampleCollector


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
    return gen_circle_data(3, tSamplesTotal)

def phoneme_dict(paramDBPath, labDBPath, recSysDir):
    pfr = HTKParamPhonemeReader()
    f = open(os.path.join(recSysDir, 'monophones.yaml'))
    monophones = yaml.load(f)
    f.close()

    print monophones
    samples = {}
    sc = SampleCollector(pfr)
    for ph in monophones:
        sc.store_corpus(ph, paramDBPath, labDBPath)
        samples[ph] = sc.corpus[:]
    return samples
