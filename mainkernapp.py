#! /usr/bin/python

"""

example:
./mainkernapp.py --applyphhmm recsystem/kOPLS/hmm/diag recsystem/mfcc/train/ recsystem/mfcc/test recsys/rodrech/mva rr_file_dict.json

./mainkernapp.py --custkmva

--applyphhmm args: hmmDir, trDir, testDir[, labfn]

"""

import sys, os, json

import numpy as np
import numpy, pickle, mlpy, yaml, ghmm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from persist import MongoPreserver
from bson import ObjectId
from functools import partial
from kernelmethods import kPCA, kPLS, kOPLS, polynomial_closure, rbf_closure, KernelRbf, distance_prop
from datagen import gen_train_data, gen_test_data, phoneme_dict, all_samples, \
    get_kernel_data, CorpusFilesProfile, KernelProfile, wraped_exp_entry
from vistools import plotGHMMEmiss
from hmmroutines import init_hmm, HMMClassifier, HMMFromGHMMConverter, hmm_built_from, \
    HmmFromGHMMBuilder, HmmFromHTKBuilder, covmatr_type
 

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
    plot1_5 = plt.scatter(testData[:, 0], testData[:, 1], color='0.5')

    kMVA.estim_kbasis(data, clabs, int(len(data) * 0.3))
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


def phoneme_rec_accuracy_hmm(acModelDir, trPhData, testPhData, debug=True):
    syspath = 'recsystem'
    
    train = []
    trTarget = []
    for ph in trPhData:
        train += trPhData[ph]
        trTarget += [ph] * len(trPhData[ph])

    print len(train), len(trTarget)

    hmmcl = HMMClassifier(nStates=3, nMix=1)
    #hmmcl.train(train, trTarget)
    hmmcl.load(trPhData.keys(), pathToHmm=acModelDir)
#    hmmcl.refine_cov_matrix()
    print len(hmmcl.modelsDict)

    test = []
    testTarget = []
    for ph in testPhData:
        test += testPhData[ph]
        testTarget += [ph] * len(testPhData[ph])

    print len(test), len(testTarget)

    predRes = hmmcl.predict(test)

    if debug:
        f = open('results', 'w')
        yaml.dump(zip(testTarget, predRes), f, default_flow_style=False)
        f.close()

    return sum([1 if t == p else 0 for t, p in zip(testTarget, predRes)]) / float(len(testTarget))


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

#    print "median:", distance_prop(x, np.median)
    sigma = 40.0
    kernel_func = KernelRbf(sigma)
    kMVA = kOPLS(kernel_func)
    if args[0] == "--drawdata":
#        kernel_func = partial(mlpy.kernel_gaussian, sigma=2.0)
        draw_data(x, y, kernel_func, testData)
#        draw_mlpy_example(x, y, testData)
    elif args[0] == "--custkmva":
        draw_kmva_obj(x, y, kMVA, testData)
    elif args[0] == "--applyhmm":
        apply_hmm(x, y, kMVA)
    elif args[0] == "--applyphhmm":
        if len(args) == 6 or len(args) == 5:
            verbose = True
            corpus = "rodrech"
            labFiles = ""
            if len(args) == 6:
                hmmDir, trDir, testDir, mvaDir, labFiles = args[1:]
            else:
                hmmDir, trDir, testDir, mvaDir = args[1:]
            phFileName = 'rr_phones_short.json'
            trSamples = phoneme_dict(trDir, recSysDir,
                                     labsfn=labFiles, phsfn=phFileName)
            if verbose:
                print "Train:"
                for smpl in trSamples:
                    print smpl, ': ', len(trSamples[smpl]), \
                          np.mean([len(phSmpl) for phSmpl in trSamples[smpl]])
            testSamples = phoneme_dict(testDir, recSysDir,
                                       labsfn=labFiles, phsfn=phFileName)
            if verbose:
                print "Test:"
                for smpl in testSamples:
                    print smpl, ': ', len(testSamples[smpl]), \
                          np.mean([len(phSmpl) for phSmpl in testSamples[smpl]])

            testType = os.path.basename(testDir)
            with open(os.path.join(recSysDir, corpus, 
                                   "mva", "kernelprof.pkl"), 'r') as f:
                kernelProf = pickle.load(f)

            mvafiles = [fn.split(".")[0] for fn in os.listdir(mvaDir)
                        if fn.endswith(".mfc")]
            mvaProf = CorpusFilesProfile(mvafiles)

            phAcc = phoneme_rec_accuracy_hmm(hmmDir, trSamples, testSamples)
#            hmmFile = os.listdir(hmmDir)[0]
#            covtype = covmatr_type(os.path.join(hmmDir, hmmFile))
            trLen = sum([len(trSamples[ph]) for ph in trSamples])
            trainfiles = [fn.split(".")[0] for fn in os.listdir(trDir)
                          if fn.endswith(".mfc")]
            trProf = CorpusFilesProfile(trainfiles, trLen)

            testLen = sum([len(testSamples[ph]) for ph in testSamples])
            testfiles = [fn.split(".")[0] for fn in os.listdir(testDir)
                         if fn.endswith(".mfc")]
            testProf = CorpusFilesProfile(testfiles, testLen)
                      
            f = open(os.path.join(recSysDir, phFileName))
            phones = json.load(f)
            f.close()
            dictorsTotal = 10
            clId = ObjectId("5417e810c25c0e582698190a") # Simple HMM 3, 1, Diag
            recResEntry = wraped_exp_entry(phAcc,
                                           "mfcc_kopls", dictorsTotal, corpus,
                                           trProf, testProf, mvaProf, testType,
                                           clId, phones)

            if verbose:
                for key in recResEntry:
                    if not key.endswith('files'):
                        print key, " --> ", recResEntry[key]

            saveToDB = False
            if saveToDB:
                pres = MongoPreserver()
                pres.persist(recResEntry)
        else:
            print "--applyphhmm hmmDir trainDir testDir mvaDict [labs_dict_file]"
    else:
        print "Wrong command! Choose:\n", prompt


if __name__ == "__main__":
    main()
