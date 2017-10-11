import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process.kernels


def get_gaussian_kernel(x, length_scales=(.5, 1, 2)):
    series = [np.exp(-(x * x) / (2 * l * l)) for l in length_scales]
    labels = [r'$\sigma = {}$'.format(l) for l in length_scales]
    return series, labels


def get_rational_quadratic_kernel(x, length_scales=(1,), shapes=(.5, 1, 2)):
    series = [(1 + x * x / (2 * l * l)) ** -s for l in length_scales for s in shapes]
    labels = [r'$\sigma = {}$, $\beta = {}$'.format(l, s) for l in length_scales for s in shapes]
    return series, labels


def get_matern_kernel(x, length_scales=(1,), nus=(.5, 1.5, 2.5)):
    series = [sklearn.gaussian_process.kernels.Matern(nu=n)(np.array([[0]]), np.array([x]).T / l)[0] for l in length_scales for n in nus]
    labels = [r'$\sigma = {}$, $\nu = {}$'.format(l, n) for l in length_scales for n in nus]
    return series, labels


def get_exponential_kernel(x, length_scales=(.5, 1, 2)):
    series = [np.exp(-np.abs(x) / (2 * l)) for l in length_scales]
    labels = [r'$\sigma = {}$'.format(l) for l in length_scales]
    return series, labels


def plot(fun, x):
    series, labels = fun(x)
    for s in series:
        plt.plot(x, s)
    plt.legend(labels)
    plt.show()

plot(get_matern_kernel, np.linspace(-3, 3, 1000))
