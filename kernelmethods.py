#! usr/bin/python

import numpy as np
import mlpy, pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eig
from itertools import combinations

def distance_prop(data, prop=np.mean):
    return prop([np.linalg.norm(np.array(x) - np.array(y)) for x, y in combinations(data, 2)])

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
#        plt.imshow(Kx, cmap = cm.Greys_r)
#        plt.show()
        vals, vecs = np.linalg.eig(matForEig)
        vals = vals[:Kx.shape[0]]
        #print vals[:20]
        vals = np.real(np.array([vals[i] for i in range(1, len(vals), 2)]))
        vecs = vecs[:Kx.shape[0], :Kx.shape[0]]
        self.vecs = vecs[:, 1]
        for i in range(3, vecs.shape[1], 2):
            self.vecs = np.hstack([self.vecs, vecs[:, i]])
        self.vecs = np.mat(self.vecs)
        norm_mat = np.mat(np.diag([np.sqrt(2)/np.sqrt(vals[i]) for i in range(len(vals))]))
        #plobj1 = pylab.plot(vecs[:, 0])
        self.vecs = self.vecs * norm_mat
        print 'mult:', vals[0] * self.vecs[:, 0].T * self.vecs[:, 0]
        #plobj2 = pylab.plot(vecs[:, 0])
        #pylab.show()
        #self.vecs = vecs

    def transform(self, data, k=2):
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kt = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in data])
        Kt = np.mat(mlpy.kernel_center(Kt, Kx))
        kTransTestData = np.real(Kt * self.vecs[:, :k])
        return np.array(kTransTestData)

class kOPLS(KernelMethod):
    def estim_kbasis(self, trData, labels):
        self.trData = trData
        self.vecs = []
        Y = one_of_c(labels)
        print "rank Y", np.linalg.matrix_rank(Y)  
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kx = np.mat(mlpy.kernel_center(Kx, Kx))
        self.Kx = Kx
        Ky = Y * Y.T
        print Y.shape, Ky.shape
        print "Ky rankf", np.linalg.matrix_rank(Ky)
        Ky = np.mat(mlpy.kernel_center(Ky, Ky))
        for k in range(2):

#        plt.imshow(Kx, cmap = cm.Greys_r)
#        plt.show()
            vals, vecs = eig(Kx*Ky*Kx, Kx*Kx)
            vals = np.array([np.real(v) for v in vals])
            alpha = vecs[:, vals.argmax()]
            lamb = vals.max()
            plt.plot(vals)


            alpha = np.mat(alpha).T
            normTerm = (alpha.T * Kx) * (Kx * alpha)

        #print 'mult before norm:', normTerm
            alpha = alpha / np.sqrt(normTerm)
            print 'mult after norm:', (alpha.T * Kx) * (Kx * alpha)
        
            self.vecs.append(alpha)
#        self.vecs = self.vecs * norm_mat
            print np.linalg.matrix_rank(Kx*Ky*Kx)
            Ky = Ky - lamb * (Kx * alpha) * (alpha.T * Kx)
            print np.linalg.matrix_rank(Kx*Ky*Kx)

        plt.show()

    def transform(self, data, k=2):
        Kx = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in self.trData])
        Kt = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in data])
        Kt = np.mat(mlpy.kernel_center(Kt, Kx))
        kTransTestData = np.real(Kt * self.vecs[:, :k])
        return np.array(kTransTestData)


class kCCAWrapper(KernelMethod):
    def _read_dat(self, fn):
        f = open(fn, 'r')
        data = []
        for line in f:
            if not line.startswith('#') and len(line.split()) > 0:
                data.append([float(numb) for numb in line.split()])
        f.close()
        return data
        
    def load_basis(self, Kxfn, Wxfn, trainfn):
        self.Kx = np.mat(self._read_dat(Kxfn))
        self.vecs = np.mat(self._read_dat(Wxfn))
        self.trData = self._read_dat(trainfn)

    def transform(self, data, k=2):
        Kt = np.mat([[self.kernel_func(np.array(xi), np.array(xj)) for xj in self.trData] for xi in data])
        Kt = np.mat(mlpy.kernel_center(Kt, self.Kx))
        kTransTestData = np.real(Kt * self.vecs[:, :k])
        return np.array(kTransTestData)
