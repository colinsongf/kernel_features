#! /usr/bin python

import numpy as np
import ghmm
from scipy.stats import norm
import matplotlib.pyplot as plt

def multinormal_pdf(r, mean, cov):
    """ Probability density function for a multidimensional Gaussian distribution."""
    dim  = r.shape[-1]
    dev  = r - mean
    maha = np.einsum('...k,...kl,...l->...', dev, np.linalg.pinv(cov), dev)
    return (2 * np.pi)**(-0.5 * dim) * np.linalg.det(cov)**(0.5) * np.exp(-0.5 * maha)

def get_weight(state, mixtCompInd):
    return ghmm.ghmmwrapper.double_array_getitem(state.c, mixtCompInd)

def gauss_func(x, mu, sigma):
    return (1.0 / (2 * np.pi * sigma)) * np.exp(-((x - mu) ** 2) / sigma)

def plotGHMMEmiss(model, stInd, dimInd):
    state = ghmm.ghmmwrapper.cstate_array_getRef(model.cmodel.s, stInd) 
    model.getEmission(0, 0) 
    
    emission = state.getEmission(0)
    mean0 = ghmm.ghmmwrapper.double_array2list(emission.mean.vec,emission.dimension)[dimInd]
    sigma0 = ghmm.ghmmwrapper.double_array2list(emission.variance.mat,emission.dimension*emission.dimension)[dimInd * emission.dimension + dimInd]
    x = np.linspace(mean0 -4 * sigma0, mean0 + 4 * sigma0, 100)
    dx = 6 * sigma0 / 100
    p = np.zeros(len(x))
    for mixtCompInd in range(state.M):
        emission = state.getEmission(mixtCompInd)
        mean = ghmm.ghmmwrapper.double_array2list(emission.mean.vec,emission.dimension)[dimInd]
        print mean, 
        sigma = ghmm.ghmmwrapper.double_array2list(emission.variance.mat,emission.dimension*emission.dimension)[dimInd * emission.dimension + dimInd]
        print sigma,
        weight = get_weight(state, mixtCompInd)
        print weight
        p += weight * gauss_func(x, mean, sigma)

    plot1 = plt.plot(x, p)
    plt.show()


#    delta = 0.01
#    xRange = np.arange(-2, 2, delta)
#    yRange = np.arange(-2, 2, delta)
#    X, Y = np.meshgrid(xRange, yRange)
#    for y in yRange:
#        for x in xRange:
#            P = [multinormal_pdf(np.array([x, y]), np.array(mean), np.array(cov).reshape(2, 2)) for mean, cov in model.getEmission ]
#    mean, cov = model.getEmission(0, 0)
#    mean, cov = np.array(mean), np.array(cov).reshape(2, 2)
#    P = [[multinormal_pdf(np.array([x, y]), mean, cov) for x in xRange] for y in yRange]
#    plt.figure()
#    CS = plt.contour(X, Y, P)
#    plt.show()
