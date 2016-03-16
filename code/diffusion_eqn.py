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

# initialization
D = 2.0           # diffusion constant
dx = 0.05          # spatial step
dt = 0.0001       # time step
size = int(10.0 / dx + 1)     # size of 1D grid
rho = np.zeros(size)          # density
x = np.linspace(-5, 5, size)  # position
t = 0.0           # time
count = 0         # number of time steps
plots_path = './output/plots_for_paper'
data_path = './output/data/part_b'

# make path
make_path(plots_path)
make_path(data_path)

# initial condition (initial density peaks around x = 0)
rho[size / 2 - 1: size / 2 + 2] = 1.0 / (3.0 * dx)

for i in range(1000):
    for j in range(1, size - 1):
        rho[j] += D * dt * (rho[j + 1] - 2.0 * rho[j] + rho[j - 1]) / dx ** 2
    t += dt

guess = np.array([1.0])  # initial guess
popt, pcov = curve_fit(func, x, rho, guess)  # curve fitting
y = 1.0 / (sqrt(2 * pi) * popt[0]) * np.exp(- x ** 2 / (2 * popt[0] ** 2))

# plot
plt.plot(x, rho, 'b', marker='o')
plt.plot(x, y, 'r')
plt.show()