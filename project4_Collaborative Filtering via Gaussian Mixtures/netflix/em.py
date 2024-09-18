"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    def log_md_gaussian(x: np.ndarray, mu: np.ndarray, var: float):
        # Taking the log of multi d gaussian pdf with diagonal
        # covariance matrix yields:
        return (-1/2)*(len(x)*np.log(2*np.pi*var) + ((1/var)*(x - mu)) @ (x - mu).T)

    soft_counts = np.zeros((X.shape[0], len(mixture.p)))
    ll = 0

    for i in range(X.shape[0]):
        Cu = np.where(X[i] != 0)[0]
        X_Cu = X[i, Cu]
        if len(Cu) != 0:
            f_row = np.log(mixture.p + 1e-16) + np.array(list(map(lambda mu, sigma_s: log_md_gaussian(X_Cu, mu[Cu], sigma_s), mixture.mu, mixture.var)))
        else:
            f_row = np.log(mixture.p + 1e-16)
        soft_counts[i, :] = f_row - logsumexp(f_row)
        # By proxy LL formula:
        ll += np.exp(soft_counts[i, :]) @ (f_row - soft_counts[i, :]).T

    return np.exp(soft_counts), ll


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    delta_mat = np.where(X == 0, X, 1)
    Cu_count = delta_mat.sum(axis=1)

    for k in range(post.shape[1]):
        post_k = post[:, k]
        for l in range(X.shape[1]):
            denominator = delta_mat[:, l].T @ post_k
            if denominator >= 1:
                mixture.mu[k, l] = (X[:, l].T @ post_k) / denominator

        norm_vec = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            Cu = X[i] != 0
            X_Cu = X[i, Cu]
            norm_vec[i] = np.linalg.norm(X_Cu - mixture.mu[k, Cu]) ** 2

        mixture.var[k] = max((norm_vec @ post_k) / (Cu_count.T @ post_k), min_variance)
        mixture.p[k] = (1 / X.shape[0]) * sum(post_k)

    return mixture


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
        mixture = mstep(X, post, mixture)

    return mixture, post, ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """

    def log_md_gaussian(x: np.ndarray, mu: np.ndarray, var: float):
        # Taking the log of multi d gaussian pdf with diagonal
        # covariance matrix yields:
        return (-1 / 2) * (len(x) * np.log(2 * np.pi * var) + ((1 / var) * (x - mu)) @ (x - mu).T)

    X_pred = X.copy()

    soft_counts = np.zeros((X.shape[0], len(mixture.p)))

    for i in range(X.shape[0]):
        Cu = np.where(X[i] != 0)[0]
        X_Cu = X[i, Cu]
        if len(Cu) != 0:
            f_row = np.log(mixture.p + 1e-16) + np.array(list(map(lambda mu, sigma_s: log_md_gaussian(X_Cu, mu[Cu], sigma_s), mixture.mu, mixture.var)))
        else:
            f_row = np.log(mixture.p + 1e-16)
        soft_counts[i, :] = f_row - logsumexp(f_row)

    soft_counts = np.exp(soft_counts)

    for i in range(X.shape[0]):
        for l in np.where(X[i] == 0)[0]:
            X_pred[i, l] = soft_counts[i] @ mixture.mu[:, l]
    return X_pred

