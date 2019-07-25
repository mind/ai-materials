#!/usr/bin/env python3
import mxnet as mx
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    X = np.arange(-1, 1, 0.1)
    Y = np.arange(-1, 1, 0.1)
    print(X, Y)
    X, Y = np.meshgrid(X, Y)
    print(X, Y)
    Z = X**2 - Y**2

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
    ax.text(0, 0, 0, 'Saddle Point')

    plt.show()


if __name__ == '__main__':
    main()
