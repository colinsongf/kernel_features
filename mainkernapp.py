#! /usr/bin/python

import sys

import numpy as np
import numpy
import matplotlib.pyplot as plt
import mlpy
from functools import partial
import yaml

def line(stDot, finDot, res=100):
    assert len(stDot) == len(finDot)
    return np.array([alpha*stDot + (1 - alpha)*finDot for alpha in np.linspace(0, 1, res)])

def rbf_closure(sigma):
    def rbf(x1, x2):
        assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
        return np.exp(-np.dot((x1 - x2), (x1 - x2)) / (2 * sigma**2))
    return rbf

def polynomial_closure(ext):
    def polynomial(x1, x2):
        assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
        return np.dot(x1, x2) ** ext
    return polynomial


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

#    vals = np.mat(gaussian_pca.evals())
#    vecs = np.mat(gaussian_pca.coeff())
#    Kx = gK
#    Kx = np.mat(mlpy.kernel_center(Kx, Kx))
    

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
    if not args:
        print "--drawdata"    
        sys.exit(0)

    if args[0] == "--drawdata":
        np.random.seed(0)
        np.random.seed(0)
        x = np.zeros((150, 2))
        y = np.empty(150, dtype=np.int)
        theta = np.random.normal(0, np.pi, 50)
        r = np.random.normal(0, 0.1, 50)
        x[0:50, 0] = r * np.cos(theta)
        x[0:50, 1] = r * np.sin(theta)
        y[0:50] = 0
        theta = np.random.normal(0, np.pi, 50)
        r = np.random.normal(2, 0.1, 50)
        x[50:100, 0] = r * np.cos(theta)
        x[50:100, 1] = r * np.sin(theta)
        y[50:100] = 1
        theta = np.random.normal(0, np.pi, 50)
        r = np.random.normal(5, 0.1, 50)
        x[100:150, 0] = r * np.cos(theta)
        x[100:150, 1] = r * np.sin(theta)
        y[100:150] = 2

        tSamplesTotal = 50
        testData = np.zeros((tSamplesTotal, 2))
        theta = np.random.normal(0, np.pi, tSamplesTotal)
        r = np.random.normal(3, 0.1, tSamplesTotal)

        testData[:, 0] = r * np.sin(theta)
        testData[:, 1] = r * np.cos(theta)
#        kernel_func = partial(mlpy.kernel_gaussian, sigma=2.0)
        kernel_func = rbf_closure(2.0)
        draw_data(x, y, kernel_func, testData)
        draw_mlpy_example(x, y, testData)


if __name__ == "__main__":
    print "Hello!"
    main()
