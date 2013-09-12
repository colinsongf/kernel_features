#! /usr/bin/python

import sys

import numpy as np
import numpy
import matplotlib.pyplot as plt
import mlpy

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


def draw_data(data, kernel_func):
    fig = plt.figure(1) # plot

    plot1 = plt.plot(data[:, 0], data[:, 1], 'o', label='Original Data')

    x_axis = line(np.array([0, 0]), np.array([3, 0]))
#    x_axis = np.array([np.array([val, val**2]) for val in np.linspace(1, 4, 100)])
    x_axis = x_axis - np.mean(x_axis)
    plot2 = plt.plot(x_axis[:, 0], x_axis[:, 1], label='x')

    y_axis = line(np.array([0, 0]), np.array([0, 3]))
    y_axis = y_axis - np.mean(y_axis)
    plot3 = plt.plot(y_axis[:, 0], y_axis[:, 1], label='y')

    xx = plt.xlim(-10, 10)
    yy = plt.ylim(-10, 10)

    emp_cov = np.cov(data.T)
    alpha, psi = np.linalg.eig(emp_cov)
#    x_axis_trans = np.array([psi * np.mat(x_axis[i]).T for i in range(len(x_axis))])

#    x_axis_trans = np.array((psi * np.mat(x_axis).T).T)
    x_axis_trans = np.mat(x_axis) * psi.T
#    plot4 = plt.plot(x_axis_trans[:, 0], x_axis_trans[:, 1], label='x_trans')

    y_axis_trans = np.mat(y_axis) * psi.T
#    plot5 = plt.plot(y_axis_trans[:, 0], y_axis_trans[:, 1], label='y_trans')

    data_trans = np.mat(data) * psi.T
    plot51 = plt.plot(data_trans[:, 0], data_trans[:, 1], 'o', label='Lin trans data')


    Kx = np.mat([[kernel_func(xi, xj) for xj in data] for xi in data])
    
    vals, vecs = np.linalg.eig((1.0 / len(data))*Kx)
    norm_mat = np.mat(np.diag([1.0/np.sqrt(vals[i] * (vecs[i]*vecs[i].T)[0,0]) for i in range(len(vals))]))
    vecs = norm_mat * vecs
    print 'mult:', vals[98] * vecs[98] * vecs[98].T

    data_k_trans = (vecs[:2] * Kx).T
    plot6 = plt.plot(data_k_trans[:, 0], data_k_trans[:, 1], 'o', label='Kern trans data')

    plt.legend()
    plt.show()


def draw_pylab_example(data):
    pass

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

        kernel_func = polynomial_closure(1)
        draw_data(x, kernel_func)


if __name__ == "__main__":
    print "Hello!"
    main()
