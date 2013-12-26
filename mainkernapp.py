#! /usr/bin/python

import sys, os

import numpy as np
import numpy, pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mlpy
from functools import partial
from kernelmethods import kPCA, kPLS, kOPLS, polynomial_closure, rbf_closure, KernelRbf, distance_prop
from datagen import gen_train_data, gen_test_data, phoneme_dict
import yaml, pickle, ghmm
from vistools import plotGHMMEmiss
from hmmroutines import init_hmm, HMMClassifier, HMMFromGHMMConverter, hmm_built_from, \
    HmmFromGHMMBuilder, HmmFromHTKBuilder
 

def line(stDot, finDot, res=100):
    assert len(stDot) == len(finDot)
    return np.array([alpha*stDot + (1 - alpha)*finDot for alpha in np.linspace(0, 1, res)])

def draw_data(data, clabs, kernel_func, testData):
    fig = plt.figure(1) # plot

    print data.shape
    ax1 = plt.subplot(121)
    plot1 = plt.scatter(data[:, 0], data[:, 1], c=clabs)
    plot1_5 = plt.scatter(testData[:, 0], testData[:, 1])

    Kx = np.mat([[kernel_func(xi, xj) for xj in data] for xi in data])
    KxCopy = Kx
    Kx = np.mat(mlpy.kernel_center(Kx, Kx))
    print "KxCopy == Kx", np.allclose(KxCopy, Kx)
    vals, vecs = np.linalg.eig(Kx)

    print "mult before normalization", vals[0]*vecs[:, 0].T*vecs[:, 0]

    norm_mat = np.mat(np.diag([1.0/np.sqrt(vals[i]) for i in range(len(vals))]))
    vecs = vecs * norm_mat
    print 'mult:', vals[0] * vecs[:, 0].T * vecs[:, 0]

#    gK = mlpy.kernel_gaussian(data, data, sigma=2)
#    gaussian_pca = mlpy.KPCA(mlpy.KernelGaussian(2.0))
#    gaussian_pca.learn(data)

    data_k_trans = np.real(Kx.T * vecs[:, :2])
    data_k_trans = np.array(data_k_trans)
    ax2 = plt.subplot(122)
    plot2 = plt.scatter(data_k_trans[:, 0], data_k_trans[:, 1], c=clabs)

    Kt = np.mat([[kernel_func(xi, xj) for xj in data] for xi in testData])

    Kt = np.mat(mlpy.kernel_center(Kt, KxCopy))
    kTransTestData = np.real(Kt * vecs[:, :2])
    kTransTestData = np.array(kTransTestData)
    
    plot2_5 = plt.scatter(kTransTestData[:, 0], kTransTestData[:, 1])

    plt.show()

def draw_kmva_obj(data, clabs, kMVA, testData):
    ax1 = plt.subplot(121)
    plot1 = plt.scatter(data[:, 0], data[:, 1], c=clabs)
    plot1_5 = plt.scatter(testData[:, 0], testData[:, 1])

    kMVA.estim_kbasis(data, clabs, int(len(data) * 0.5))
    print kMVA.__class__.__name__ + " created"
    ax2 = plt.subplot(122)

    data_k_trans = kMVA.transform(data, k=2)
    plot2 = plt.scatter(data_k_trans[:, 0], data_k_trans[:, 1], c=clabs)

    kTransData = kMVA.transform(testData, 2)
    plot2_5 = plt.scatter(kTransData[:, 0], kTransData[:, 1])

    plt.show()

    stream = open(kMVA.__class__.__name__ + ".pkl", 'w')
    pickle.dump(kMVA, stream)
    stream.close()

def draw_mlpy_example(data, clabs, testData):
    gK = mlpy.kernel_gaussian(data, data, sigma=2) # gaussian kernel matrix
    gaussian_pca = mlpy.KPCA(mlpy.KernelGaussian(2.0))
    gaussian_pca.learn(data)
    gz = gaussian_pca.transform(data, k=2)

    fig = plt.figure(1)
    ax1 = plt.subplot(121)
    plot1 = plt.scatter(data[:, 0], data[:, 1], c=clabs)
    plot1_5 = plt.scatter(testData[:, 0], testData[:, 1])
    title1 = ax1.set_title('Original data')
    trTestData = gaussian_pca.transform(testData, k=2)
    ax2 = plt.subplot(122)
    plot2 = plt.scatter(gz[:, 0], gz[:, 1], c=clabs)
    plot2_5 = plt.scatter(trTestData[:, 0], trTestData[:, 1])
    title2 = ax2.set_title('Gaussian kernel')
    plt.show()


def apply_hmm(data, clabs, kMVA):
         
    kMVA.estim_kbasis(data, clabs)
    data = kMVA.transform(data, k=2)

    model = init_hmm(3, 3, 2)

    data = [list(x) for x in data]
    dt, T = 1, 2
    seq_set = ghmm.SequenceSet(ghmm.Float(), [sum(list(data[i:i+T]), []) for i in range(0, len(data) - T, dt)])
    model.baumWelch(seq_set, 10, 0.01)

    plotGHMMEmiss(model, stInd=0, dimInd=1)

def check_particular_phoneme(phData):
    syspath = 'recsystem'
    somePh = phData.keys()[0]
    print somePh, len(phData[somePh][0]), len(phData[somePh][0][0])
    for sample in phData[somePh][0]:
        p = plt.plot(sample)
    plt.grid(True)
    plt.show()
    hmm = init_hmm(nStates=3, nMix=1, dim=len(phData[somePh][0][0]))
    seq_set = ghmm.SequenceSet(ghmm.Float(), [sum(phSample, []) for phSample in phData[somePh]])
    print 'Let us train it!'
    hmm.baumWelch(seq_set)
#    print os.path.join(syspath, 'hmm', somePh)
#    hmm = hmm_built_from(HmmFromGHMMBuilder, os.path.join(syspath, 'hmm', somePh))
    hmmReloaded = HMMClassifier.reassigned_ghmm_object(hmm, 0.5)
    loglikel = [hmmReloaded.loglikelihood(seq) for seq in seq_set]

    pl = plt.plot(loglikel)
    plt.show()
#    print len(seq), hmm.loglikelihood(seq), hmm.viterbi(seq)


def apply_hmm_to_phonemes(trPhData, testPhData):
    syspath = 'recsystem'
    
    train = []
    trTarget = []
    for ph in trPhData:
        train += trPhData[ph]
        trTarget += [ph] * len(trPhData[ph])

    print len(train), len(trTarget)

    hmmcl = HMMClassifier(nStates=3, nMix=1)
    hmmcl.train(train, trTarget)
    #hmmcl.load(phData.keys())
    hmmcl.refine_cov_matrix()
    print len(hmmcl.modelsDict)

    test = []
    testTarget = []
    for ph in testPhData:
        test += testPhData[ph]
        testTarget += [ph] * len(testPhData[ph])

    print len(train), len(testTarget)

    predRes = hmmcl.predict(test)
    print sum([1 if t == p else 0 for t, p in zip(testTarget, predRes)]) / float(len(testTarget))

    f = open('results', 'w')
    yaml.dump(zip(testTarget, predRes), f, default_flow_style=False)
    f.close()


def main():
    args = sys.argv[1:]
    print args
    prompt = ('\t' * 1).join(["--drawdata", "--custkmva", "--applyhmm"])    
    if not args:
        print prompt
        sys.exit(0)
    recSysDir = 'recsystem'
    x, y = gen_train_data([(0, 50), (1, 50), (2, 50)])
    testData = gen_test_data()
#    kernel_func = polynomial_closure(2)

    sigma = 20.0#distance_prop(x, np.mean)
    print "median:", sigma
    kernel_func = KernelRbf(sigma)
    kMVA = kOPLS(kernel_func)
    if args[0] == "--drawdata":
#        kernel_func = partial(mlpy.kernel_gaussian, sigma=2.0)
        draw_data(x, y, kernel_func, testData)
        draw_mlpy_example(x, y, testData)
    elif args[0] == "--custkmva":
        draw_kmva_obj(x, y, kMVA, testData)
    elif args[0] == "--applyhmm":
        apply_hmm(x, y, kMVA)
    elif args[0] == "--applyphhmm":
        if len(args) == 4:
            trDir, testDir, labDir = args[1:]
            trSamples = phoneme_dict(trDir, labDir, recSysDir)
            print 'len(trSamples) =', len(trSamples)
            for smpl in trSamples:
                print smpl, ': ', len(trSamples[smpl]), np.mean([len(phSmpl) for phSmpl in trSamples[smpl]])

            testSamples = phoneme_dict(testDir, labDir, recSysDir)
            print 'len(testSamples) =', len(testSamples)
            for smpl in testSamples:
                print smpl, ': ', len(testSamples[smpl]), np.mean([len(phSmpl) for phSmpl in testSamples[smpl]])

            apply_hmm_to_phonemes(trSamples, testSamples)
        else:
            print "--applyphhmm paramdbpath labdbpath"
    else:
        print "Wrong command! Choose:\n", prompt


if __name__ == "__main__":
    main()
