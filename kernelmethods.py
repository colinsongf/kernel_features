#! usr/bin/python

import numpy as np
import mlpy

class KernelMethod:
    def __init__(self, kernel_func):
        self.kernel_func = kernel_func
    def estim_kbasis(self, trData):
        pass
    def transform(self, data, trDim):
        pass

class kPCA(KernelMethod):
    def estim_kbasis(self, trData):
        Kx = np.mat([[self.kernel_func(xi, xj) for xj in trData] for xi in trData])
        Kx = np.mat(mlpy.kernel_center(Kx, Kx))
        vals, vecs = np.linalg.eig(Kx)
        norm_mat = np.mat(np.diag([1.0/np.sqrt(vals[i]) for i in range(len(vals))]))
        vecs = vecs * norm_mat
        print 'mult:', vals[0] * vecs[:, 0].T * vecs[:, 0]
    def transform(self, data):
        pass
