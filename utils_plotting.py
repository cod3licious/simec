from __future__ import division
from __future__ import print_function
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def get_colors(N=100):
    HSV_tuples = [(x * 1. / (N+1), 1., 0.8) for x in range(N)]
    return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]


def plot2d(X, Y_plot, X_test=None, Y_plot_test=None, title='original'):
    plt.figure()
    if (X_test is not None) and (Y_plot_test is not None):
        plt.scatter(X[:, 0], X[:, 1], c=Y_plot.flatten(), alpha=1)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_plot_test.flatten(), alpha=0.3)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=Y_plot.flatten(), alpha=1)
    plt.title(title)


def plot3d(X, Y_plot, X_test=None, Y_plot_test=None, title='original'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if (X_test is not None) and (Y_plot_test is not None):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_plot.flatten(), alpha=1)
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=Y_plot_test.flatten(), alpha=0.3)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y_plot.flatten(), alpha=1)
    plt.title(title)


def plot_digits(X, digits, title=None, plot_box=True):
    colorlist = get_colors(10)
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=colorlist[digits.target[i]],
                 fontdict={'weight': 'medium', 'size': 'smaller'})

    if plot_box and hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-2:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    if title is not None:
        plt.title(title)


def plot_mnist(X, y, X_test=None, y_test=None, title=None):
    plt.figure()
    colorlist = get_colors(10)
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    if (X_test is not None) and (y_test is not None):
        x_min, x_max = np.min(np.array([x_min, np.min(X_test, 0)]), 0), np.max(np.array([x_max, np.max(X_test, 0)]), 0)
        X_test = (X_test - x_min) / (x_max - x_min)
    X = (X - x_min) / (x_max - x_min)
    if (X_test is not None) and (y_test is not None):
        for i in range(X_test.shape[0]):
            plt.text(X_test[i, 0], X_test[i, 1], str(y_test[i]),
                     color=colorlist[y_test[i]],
                     fontdict={'weight': 'medium', 'size': 'smaller'},
                     alpha=0.4)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colorlist[y[i]],
                 fontdict={'weight': 'medium', 'size': 'smaller'},
                 alpha=1.)
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    if title is not None:
        plt.title(title)


def plot_mnist2(X, y, X_test=None, y_test=None, X_original=None, title=None):
    plt.figure()
    ax = plt.subplot(111)
    colorlist = get_colors(10)
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    if (X_test is not None) and (y_test is not None):
        x_min, x_max = np.min(np.array([x_min, np.min(X_test, 0)]), 0), np.max(np.array([x_max, np.max(X_test, 0)]), 0)
        X_test = (X_test - x_min) / (x_max - x_min)
    X = (X - x_min) / (x_max - x_min)
    if (X_test is not None) and (y_test is not None):
        for i in range(X_test.shape[0]):
            plt.text(X_test[i, 0], X_test[i, 1], str(y_test[i]),
                     color=colorlist[y_test[i]],
                     fontdict={'weight': 'medium', 'size': 'smaller'},
                     alpha=0.4)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colorlist[y[i]],
                 fontdict={'weight': 'medium', 'size': 'smaller'},
                 alpha=1.)
    # plot some images on top
    if X_original is not None:
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X_original[i].reshape(28, 28), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    if title is not None:
        plt.title(title, fontdict={'weight': 'medium', 'size': 25})


def plot_20news(X, y, target_names, X_test=None, y_test=None, title=None, legend=False):
    colorlist = get_colors(len(target_names))

    def plot_scatter(X, y, alpha=1):
        y = np.array(y)
        for i, l in enumerate(target_names):
            plt.scatter(X[y == i, 0], X[y == i, 1], c=colorlist[i], alpha=alpha,
                        edgecolors='none', label=l if alpha >= 0.5 else None)  # , rasterized=True)
    # plot scatter plot
    plt.figure()
    if (X_test is not None) and (y_test is not None):
        plot_scatter(X_test, y_test, 0.4)
        plot_scatter(X, y, 1.)
    else:
        plot_scatter(X, y, 0.6)
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_words(X, word_list, title=None):
    # Scale and visualize the embedding vectors
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], word_list[i],
                 color='k',
                 fontdict={'weight': 'medium', 'size': 'smaller'})
    plt.xticks([]), plt.yticks([])
    plt.xlim(-0.05, 1.2)
    plt.ylim(-0.05, 1.05)
    if title is not None:
        plt.title(title)