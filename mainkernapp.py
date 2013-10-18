#! /usr/bin/python

import sys

import numpy as np
import numpy
import matplotlib.pyplot as plt
import mlpy
from functools import partial
from kernelmethods import kPCA, kPLS, polynomial_closure, rbf_closure, KernelRbf
from datagen import gen_train_data, gen_test_data
import yaml, pickle

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

    gK = mlpy.kernel_gaussian(data, data, sigma=2)
    gaussian_pca = mlpy.KPCA(mlpy.KernelGaussian(2.0))
    gaussian_pca.learn(data)

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

    kMVA.estim_kbasis(data, clabs)

    ax2 = plt.subplot(122)

    data_k_trans = kMVA.transform(data, k=2)
    plot2 = plt.scatter(data_k_trans[:, 0], data_k_trans[:, 1], c=clabs)

    kTransData = kMVA.transform(testData, 2)
    plot2_5 = plt.scatter(kTransData[:, 0], kTransData[:, 1])

    plt.show()
    stream = open("kMVA.pkl", 'w')
    pickle.dump(kTransData, stream)
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

    

def main():
    args = sys.argv[1:]
    prompt = "\t\t\t--drawdata\n\t\t\t--custmva\n"    
    if not args:
        print prompt
        sys.exit(0)

    x, y = gen_train_data([(0, 50), (1, 50), (2, 50)])
    testData = gen_test_data()
#    kernel_func = polynomial_closure(2)
    kernel_func = KernelRbf(2.0)
    if args[0] == "--drawdata":
#        kernel_func = partial(mlpy.kernel_gaussian, sigma=2.0)
        draw_data(x, y, kernel_func, testData)
        draw_mlpy_example(x, y, testData)
    elif args[0] == "--custkmva":
        kMVA = kPLS(kernel_func)
        draw_kmva_obj(x, y, kMVA, testData)
    else:
        print "Wrong command! Choose:\n", prompt


if __name__ == "__main__":
    print "Hello!"
    main()
