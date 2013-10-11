#! usr/bin/python

import numpy as np

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
#    y = np.empty(sum([clTot for cl, clTot in clTotals]), dtype=np.int)
    y = []
    rads = iter([0, 2, 5])
    for clTot in clTotals:
        if x.any(): x = np.vstack((x, gen_circle_data(next(rads), clTot[1])))
        else: x = gen_circle_data(next(rads), clTot[1])
#        y[0:clTot[1]] = clTot[0]
        y.extend([clTot[0]] * clTot[1])
    return x, y

def gen_test_data():
    tSamplesTotal = 50
    return gen_circle_data(3, tSamplesTotal)
