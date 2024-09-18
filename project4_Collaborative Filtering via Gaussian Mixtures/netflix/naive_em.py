"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    def gaussian_mix_pdf(x: np.ndarray, mixture: GaussianMixture):
        pdf_val = 0
        for i in range(len(mixture.p)):
            pdf_val += mixture.p[i] * multivariate_normal.pdf(x, mixture.mu[i], mixture.var[i])
        return pdf_val

    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    ll = 0

    for i in range(n):
        enumerator_list = mixture.p * np.array(list(map(lambda mu, sigma_s: multivariate_normal.pdf(X[i, :], mu, sigma_s), mixture.mu, mixture.var)))
        denominator = gaussian_mix_pdf(X[i, :], mixture)
        post[i, :] = enumerator_list/denominator
        ll += np.log(denominator)

    return post, ll
    


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    ll = None
    while (prev_ll is None or ll - prev_ll > 0.000001 * abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll
