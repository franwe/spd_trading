import numpy as np
from sklearn.neighbors import KernelDensity

from ..utils.smoothing import bspline


def density_estimation(sample, X, h, kernel="epanechnikov"):
    """Kernel Density Estimation over the sample in domain X.

    Routine for `sklearn.neighbors.KernelDensity`.

    Args:
        sample (np.array): Sample of observations. shape: (n_samples, n_features) List of n_features-dimensional data
            points. Each row corresponds to a single data point.
        X (np.array): Domain in which the density is estimated. An array of points to query. Last dimension should match
            dimension of training data. shape: (n_estimates, n_features)
        h (float): Bandwidth of the kernel. Needs to be chosen wisely or estimated. Sensitive parameter.
        kernel (str, optional): The kernel to use for the estimation, so far only the Epanechnikov kernel is
            implemented. Defaults to "epanechnikov".

    Returns:
        [np.array]: The array of log(density) evaluations. These are normalized to be probability densities, so values
        will be low for high-dimensional data. shape: (n_estimates,)
    """
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(sample.reshape(-1, 1))
    log_dens = kde.score_samples(X.reshape(-1, 1))
    density = np.exp(log_dens)
    return density


def pointwise_density_trafo_K2M(K, q_K, S_vals, M_vals):
    """Pointwise density transformation from K (Strike Price) to M (Moneyness) domain. M = S/K

    First, a spline has to be fitted to q_K, so that it is possible to extract the q_K-value at every point of
    interest, not just at the known points K.
    Then, it is iterated through the (M, S)-tuples and the density q_K is transformed to q_M.

    Args:
        K (np.array): Strike Price values for which the density q_K is know.
        q_K (np.array): Density values in Strike Price domain.
        S_vals (array-like): Prices of underlying for the density points.
        M_vals (array-like): Moneyness values for the density point.

    Returns:
        [np.array]: Density values in Moneyness domain.
    """

    _, q_K, _ = bspline(K, q_K, 15)  # fit spline to q_K

    num = len(M_vals)
    q_pointsM = np.zeros(num)

    # loop through (M, S)-tuples and calculate the q_M value at this point
    for i, m, s in zip(range(num), M_vals, S_vals):
        q_pointsM[i] = s / (m ** 2) * q_K(s / m)
    return q_pointsM


def hd_rnd_domain(HD, RND, interval=[0.5, 1.5]):
    """Interpolates HD and RND densities (q_M) to the same interval, especially same interval values!

    Args:
        HD (class): hd.Calculator class, must contain attributes M and q_M
        RND (class): rnd.Calculator class, must contain attributes M and q_M
        interval (list, optional): Interval in which the densities are interpolated. Defaults to [0.5, 1.5].

    Returns:
        [tuple of np.arrays]: Interpolated densities hd, rnd and their common M-values.
    """
    _, HD_spline, _ = bspline(HD.M, HD.q_M, sections=15, degree=2)
    _, RND_spline, _ = bspline(RND.M, RND.q_M, sections=15, degree=2)
    M = np.linspace(interval[0], interval[1], 100)

    hd = HD_spline(M)
    rnd = RND_spline(M)
    return hd, rnd, M
