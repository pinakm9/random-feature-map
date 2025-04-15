import numpy as np
import utility as ut
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
@ut.timer
def sinkhorn_div(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    """
    Computes the Sinkhorn divergence between two empirical distributions.

    Args:
        x (np.array): The first empirical distribution with shape (n, d).
        y (np.array): The second empirical distribution with shape (m, d).
        alpha (np.array, optional): Weights for the first distribution. Defaults to uniform weights.
        beta (np.array, optional): Weights for the second distribution. Defaults to uniform weights.
        epsilon (float, optional): Regularization parameter for the Sinkhorn algorithm. Defaults to 0.01.
        num_iters (int, optional): Maximum number of iterations for the Sinkhorn algorithm. Defaults to 200.
        p (int, optional): The norm degree for computing pairwise distance. Defaults to 2.

    Returns:
        float: The computed Sinkhorn divergence between the two distributions.
    """

    c = cdist(x, y)
 
    if alpha is None:
        alpha = np.ones(x.shape[0], dtype=x.dtype) / x.shape[0]

    if beta is None:
        beta = np.ones(y.shape[0], dtype=y.dtype) / y.shape[0]

    log_alpha = np.expand_dims(np.log(alpha), 1)
    log_beta = np.log(beta)

    f, g = 0. * alpha, 0. * beta
    f_, iter = 1. * alpha, 0
    while np.linalg.norm(f - f_, ord=1) / np.linalg.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = f
        f = - epsilon * logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * logsumexp(log_alpha + (np.expand_dims(f, 1) - c) / epsilon, axis=0)
        iter += 1
    #print(iter)

    OT_alpha_beta = np.sum(f * alpha) + np.sum(g * beta)
    
    c = cdist(x, x)
    f = 0. * alpha
    f_, iter = 1. * alpha, 0
    log_alpha = np.squeeze(log_alpha)
    while np.linalg.norm(f - f_, ord=1) / np.linalg.norm(f_, ord=1) > 1e-3 and iter < num_iters:
        f_ = f
        f = 0.5 * (f - epsilon * logsumexp(log_alpha + (f - c) / epsilon, axis=1) )
        iter += 1
    #print(iter)

    c = cdist(y, y)
    g = 0. * beta
    g_, iter = 1. * beta, 0
    while np.linalg.norm(g - g_, ord=1) / np.linalg.norm(g_, ord=1) > 1e-3 and iter < num_iters:
        g_ = g
        g = 0.5 * (g - epsilon * logsumexp(log_beta + (g - c) / epsilon, axis=1) )
        iter += 1
    return OT_alpha_beta - np.sum(f * alpha) - np.sum(g * beta)
