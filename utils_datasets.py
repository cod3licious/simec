from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range
import numpy as np
from sklearn.datasets import make_circles, make_blobs, make_swiss_roll, make_s_curve
from sklearn.utils import check_random_state


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
