'''
This module samples from a general distribution.
'''
import numpy as np
from scipy.integrate import cumtrapz

__all__ = [
    'sampler',
    'statistics'
]

def sampler(distribution, x_min, x_max, resolution=100, size=None, scale=1.):
    """Sample from the Black Body distribution.
    The Black Body distribution follows a Planck's law function, described in [1]_.

    Parameters
    ----------
    distribution : float or int
        Probability distribution function to sample from.
    x_min, x_max : array_like
        Lower and upper bounds for the random variable x.
    resolution : int
        Resolution of the inverse transform sampling spline. Default is 100.
    size: int, optional
        Output shape of samples. If size is None and scale is a scalar, a
        single sample is returned. If size is None and scale is an array, an
        array of samples is returned with the same shape as scale.
    scale: array-like, optional
        Scale factor for the returned samples. Default is 1.

    Returns
    -------
    x_sample : array_like
        Samples drawn from the Black Body spectrum.


    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Planck%27s_law

    """

    if size is None:
        size = np.broadcast(x_min, x_max, scale).shape or None

    x = np.linspace(np.min(x_min), np.max(x_max), resolution)
    # bb_dist = BlackBody(temperature=temperature)

    pdf = distribution(x) # bb_dist(x)
    CDF = cumtrapz(pdf, x, initial=0)
    CDF = CDF / CDF[-1]
    n_uniform = np.random.uniform(size=size)
    return np.interp(n_uniform, CDF, x)


def statistics(realisations, nbins):
    """Data from the Black Body distribution.
    This function returns the average number counts and standard deviation
    from different realisations sampled from the same distribution.

    Parameters
    ----------
    realisations: list
        List of samples.
    nbins: int
        Number of bins.

    Returns
    -------
    average:
        Average sample.
    bin_center:
        Centers of the bins.
    average_counts:
        Average number counts at the bin centers.
    std:
        Standard deviation at the bin centers.

    References
    ----------
    .. [1] Averaging over realisations.

    """
    # Read the data of length n over nmocks mocks
    data = realisations
    nmocks = len(data)

    counts = []
    for d in data:
        counts.append(np.histogram(d, bins=nbins)[0])

    # Downsample to n / nmocks length
    n = [len(d) for d in data]
    fn = [np.int(np.round(np.divide(N, nmocks))) for N in n]

    subsample = []
    for i, d in enumerate(data):
        rows_numbers_to_keep = np.random.choice(n[i], fn[i], replace=False)
        subsample.append(d[rows_numbers_to_keep])

    # Merge subsamples into a single average catalog of length
    # n = nmocks * n / nmocks
    average = np.hstack(subsample)

    # Center values: counts, bins and standard deviation
    average_counts, bin_edges = np.histogram(average, bins=nbins)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts_center = []
    for c in counts:
        counts_center.append(0.5 * (c[:-1] + c[1:]))
    std = np.std(counts, axis=0)

    return average, bin_center, average_counts, std
