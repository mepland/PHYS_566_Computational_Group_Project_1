import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *

global x, rho, plots_path, data_path


def make_path(path):
    """
    Create the output diretory.
    :param path: path of the directory.
    :return: null.
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)


def func(x, sigma):
    """
    Model function f(x) = 1D normal distribution.
    """
    return 1.0 / (sqrt(2 * pi) * sigma) * np.exp(- x ** 2 / (2.0 * sigma ** 2))


def plot(file_name, sigma, t):
    """
    Plot the figure.
    :param file_name: file name.
    :param sigma: sigma.
    :param t: time.
    :return: null.
    """
    global x, rho
    guess = np.array([1.0])  # initial guess
    popt, pcov = curve_fit(func, x, rho, guess)  # curve fitting
    y = 1.0 / (sqrt(2 * pi) * sigma) * np.exp(- x ** 2 / (2 * sigma ** 2))
    # plot
    plt.plot(x, rho, marker='o', color='b', label='Numeric result')
    plt.plot(x, y, color='r', label=r'Gaussian fit with $\sigma(t) = %.5f$' % sigma)
    plt.title('Diffusion in one dimension at t = %.2f' % t)
    plt.xlabel('position ' + r'$x$')
    plt.ylabel('density ' + r'$\rho(x)$')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 10, forward=True)
    plt.savefig(plots_path + '/' + file_name + '.pdf')
    plt.close()


# initialization
D = 2.0           # diffusion constant
dx = 0.05          # spatial step
dt = 0.0001       # time step
size = int(10.0 / dx + 1)     # size of 1D grid
rho = np.zeros(size)          # density
x = np.linspace(-5, 5, size)  # position
t = 0.0           # time
count = 0         # number of time steps
steps = [300, 600, 1200, 2400, 4800]  # five different time
fitting_data = []       # save fitting data
theoretical_data = []   # save theoretical data
time_data = []          # save time
count_data = []         # save count
plots_path = '../output/plots_for_paper/problem_2'
data_path = '../output/data/problem_2'

# make path
make_path(plots_path)
make_path(data_path)

# initial condition (initial density peaks around x = 0)
rho[size / 2 - 1: size / 2 + 2] = 1.0 / (3.0 * dx)

for i in range(5000):
    for j in range(1, size - 1):
        rho[j] += D * dt * (rho[j + 1] - 2.0 * rho[j] + rho[j - 1]) / dx ** 2
    t += dt
    count += 1
    if steps.__contains__(count):
        guess = np.array([1.0])  # initial guess
        popt, pcov = curve_fit(func, x, rho, guess)  # curve fitting
        fitting_data.append(popt[0])
        time_data.append(t)
        theoretical_data.append(sqrt(2.0 * D * t))
        plot('part_b_%d' % count, popt[0], t)  # plot the figure

# save data | number of time steps | time | theoretical data | fitting data |
np.savetxt(data_path + '/data.txt', np.transpose(np.array([steps, time_data, theoretical_data, fitting_data])))