import matplotlib.pyplot as plt
import numpy


def plot(x, ys, legs=['Training', 'CV'], show=True, xlab='Data Size', ylab='Accuracy'):
    plt.subplot(111)
    for index, y in enumerate(ys):
        plt.plot(x, y, label=legs[index])
    plt.grid()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    if show:
        plt.show()
