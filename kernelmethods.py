#! usr/bin/python

import numpy as np
import mlpy

def one_of_c(clabs):
    labsInds = {l: ind for ind, l in enumerate(set(clabs))}
    print labsInds
    Y = np.zeros(shape=(len(clabs), len(set(clabs))))
    for ind in range(len(clabs)):
        Y[ind, len(set(clabs)) - labsInds[clabs[ind]] - 1] = 1.0
    return np.mat(Y)

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

class KernelFunc:
    def __call__(self, x1, x2):
        pass

class KernelRbf(KernelFunc):
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x1, x2):
        assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
        return np.exp(-np.dot((x1 - x2), (x1 - x2)) / (2 * self.sigma**2))


class KernelMethod:
    def __init__(self, kernel_func):
        self.kernel_func = kernel_func
    def estim_kbasis(self, trData):
        pass
    def transform(self, data, trDim):
        pass

class kPCA(KernelMethod):
    def estim_kbasis(self, trData):
        self.trData = trData
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kx = np.mat(mlpy.kernel_center(Kx, Kx))
        vals, vecs = np.linalg.eig(Kx)
        norm_mat = np.mat(np.diag([1.0/np.sqrt(vals[i]) for i in range(len(vals))]))
        self.vecs = vecs * norm_mat
        print 'mult:', vals[0] * vecs[:, 0].T * vecs[:, 0]
    def transform(self, data, k=2):
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kt = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in data])
        Kt = np.mat(mlpy.kernel_center(Kt, Kx))
        kTransTestData = np.real(Kt * self.vecs[:, :k])
        return np.array(kTransTestData)

class kPLS(KernelMethod):
    def estim_kbasis(self, trData, labels):
        self.trData = trData
        Y = one_of_c(labels)
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kx = np.mat(mlpy.kernel_center(Kx, Kx))
        matForEig = np.mat(np.vstack([np.hstack([np.zeros(Kx.shape), Kx * Y]), np.hstack([Y.T * Kx, np.zeros((Y.shape[1], Y.shape[1]))])]))
        vals, vecs = np.linalg.eig(matForEig)
        norm_mat = np.mat(np.diag([1.0/np.sqrt(vals[i]) for i in range(len(vals))]))
        print 'mult:', vals[0] * vecs[:, 0].T * vecs[:, 0]
        self.vecs = (vecs * norm_mat)[:Kx.shape[0]]

    def transform(self, data, k=2):
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kt = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in data])
        Kt = np.mat(mlpy.kernel_center(Kt, Kx))
        kTransTestData = np.real(Kt * self.vecs[:, :k])
        return np.array(kTransTestData)
