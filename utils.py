from __future__ import division
from __future__ import print_function
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.datasets import make_circles, make_blobs, make_swiss_roll, make_s_curve
from sklearn.utils import check_random_state


def get_colors(N=100):
    HSV_tuples = [(x * 1. / (N+1), 1., 0.8) for x in range(N)]
    return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]


def make_3_circles(n_samples, random_state=1):
    random_state = check_random_state(random_state)
    X = np.ones((3 * n_samples, 3))
    Y_plot = np.ones((3 * n_samples, 1))
    X[:n_samples, :2], _ = make_circles(n_samples=n_samples, noise=0.05, factor=.01, random_state=random_state)
    X[:n_samples, 2] *= -1
    Y_plot[:n_samples, 0] = 1
    X[n_samples:2 * n_samples, :2], _ = make_circles(n_samples=n_samples,
                                                     noise=0.05, factor=.01, random_state=random_state)
    X[n_samples:2 * n_samples, 2] = 0
    Y_plot[n_samples:2 * n_samples, 0] = 2
    X[2 * n_samples:, :2], _ = make_circles(n_samples=n_samples, noise=0.05, factor=.01, random_state=random_state)
    Y_plot[2 * n_samples:, 0] = 3
    # shuffle examples
    idx = random_state.permutation(list(range(3 * n_samples)))
    X, Y_plot = X[idx, :], Y_plot[idx, :]
    # cut to actual size
    X, Y_plot = X[:n_samples, :], Y_plot[:n_samples, :]
    return X, Y_plot


def make_sphere(n_samples, random_state=1):
    # Create our sphere.
    random_state = check_random_state(random_state)
    p = random_state.rand(int(n_samples * 1.5)) * (2 * np.pi - 0.5)
    t = random_state.rand(int(n_samples * 1.5)) * np.pi

    # Sever the poles from the sphere.
    indices = ((t < (np.pi - (np.pi / 10))) & (t > ((np.pi / 10))))
    colors = p[indices]
    x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
        np.sin(t[indices]) * np.sin(p[indices]), \
        np.cos(t[indices])
    sphere_data = np.array([x, y, z]).T
    return sphere_data[:n_samples, :], colors[:n_samples]


def make_broken_swiss_roll(n_samples, random_state=1):
    # get original swiss roll
    X, Y_plot = make_swiss_roll(2 * n_samples, random_state=random_state)
    # cut off a part
    X, Y_plot = X[X[:, 0] > -5, :], Y_plot[X[:, 0] > -5]
    # get desired number of samples
    X, Y_plot = X[:n_samples, :], Y_plot[:n_samples]
    return X, Y_plot


def make_peaks(n_samples, random_state=1):
    # get randomly sampled 2d grid
    random_state = check_random_state(random_state)
    X = 10. * random_state.rand(n_samples, 3)
    # have as 3rd dimension some peaks
    X[X[:, 0] <= 5, 2] = np.cos(0.9 * (X[X[:, 0] <= 5, 1] - 2))
    X[X[:, 0] > 5, 2] = np.cos(0.5 * (X[X[:, 0] > 5, 1] - 5))
    # 3rd dim is also the color
    Y_plot = X[:, 2]
    return X, Y_plot


def load_dataset(dataset, n_samples, random_state=1, n_features=3):
    # wrapper function to load one of the 3d datasets
    if dataset == 's_curve':
        return make_s_curve(n_samples, random_state=random_state)
    elif dataset == 'swiss_roll':
        return make_swiss_roll(n_samples, random_state=random_state)
    elif dataset == 'broken_swiss_roll':
        return make_broken_swiss_roll(n_samples, random_state=random_state)
    elif dataset == 'sphere':
        return make_sphere(n_samples, random_state=random_state)
    elif dataset == '3_circles':
        return make_3_circles(n_samples, random_state=random_state)
    elif dataset == 'peaks':
        return make_peaks(n_samples, random_state=random_state)
    elif dataset == 'blobs':
        return make_blobs(n_samples, n_features=n_features, centers=3, random_state=random_state)
    else:
        print("unknown dataset")


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


def check_embed_match(X_embed1, X_embed2):
    """
    Check whether the two embeddings are almost the same by computing their normalized euclidean distances
    in the embedding space and checking the correlation.
    Inputs:
        - X_embed1, X_embed2: two Nxd matrices with coordinates in the embedding space
    Returns:
        - msq, r^2, rho: mean squared error, R^2, and Spearman correlation coefficient between the distance matrices of
                         both embeddings (mean squared error is more exact, corrcoef a more relaxed error measure)
    """
    D_emb1 = pdist(X_embed1, 'euclidean')
    D_emb2 = pdist(X_embed2, 'euclidean')
    D_emb1 /= D_emb1.max()
    D_emb2 /= D_emb2.max()
    # compute mean squared error
    msqe = np.mean((D_emb1 - D_emb2) ** 2)
    # compute Spearman correlation coefficient
    rho = spearmanr(D_emb1.flatten(), D_emb2.flatten())[0]
    # compute Pearson correlation coefficient
    r = pearsonr(D_emb1.flatten(), D_emb2.flatten())[0]
    return msqe, r**2, rho


def check_similarity_match(X_embed, S, X_embed_is_S_approx=False, norm=False):
    """
    Since SimEcs are supposed to project the data into an embedding space where the target similarities
    can be linearly approximated; check if X_embed*X_embed^T = S
    (check mean squared error, R^2, and Spearman correlation coefficient)
    Inputs:
        - X_embed: Nxd matrix with coordinates in the embedding space
        - S: NxN matrix with target similarities (do whatever transformations were done before using this
             as input to the SimEc, e.g. centering, etc.)
    Returns:
        - msq, r^2, rho: mean squared error, R^2, and Spearman correlation coefficient between linear kernel of embedding
                         and target similarities (mean squared error is more exact, corrcoef a more relaxed error measure)
    """
    if X_embed_is_S_approx:
        S_approx = X_embed
    else:
        # compute linear kernel as approximated similarities
        S_approx = X_embed.dot(X_embed.T).real
    # to get results that are comparable across similarity measures, we have to normalize them somehow,
    # in this case by dividing by the absolute max value of the target similarity matrix
    if norm:
        S_norm = S / np.max(np.abs(S))
        S_approx /= np.max(np.abs(S_approx))
    else:
        S_norm = S
    # compute mean squared error
    msqe = np.mean((S_norm - S_approx) ** 2)
    # compute Spearman correlation coefficient
    rho = spearmanr(S_norm.flatten(), S_approx.flatten())[0]
    # compute Pearson correlation coefficient
    r = pearsonr(S_norm.flatten(), S_approx.flatten())[0]
    return msqe, r**2, rho
