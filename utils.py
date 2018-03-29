from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr


def center_K(K):
    """
    Center the given square (and symmetric) kernel matrix

    Input:
        - K: square (and symmetric) kernel (similarity) matrix
    Returns:
        - centered kernel matrix (like if you had subtracted the mean from the input data)
    """
    n, m = K.shape
    assert n == m, "Kernel matrix needs to be square"
    H = np.eye(n) - np.tile(1. / n, (n, n))
    B = np.dot(np.dot(H, K), H)
    return (B + B.T) / 2


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
