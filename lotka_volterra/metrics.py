import numpy as np


def rmse(simulated, observed):
    """
    Root-mean-square error between simulated and observed values.

    Parameters
    ----------
    simulated, observed : array-like
        Equal-length arrays of simulated and observed values.

    Returns
    -------
    float
    """
    simulated = np.asarray(simulated, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return np.sqrt(np.mean((simulated - observed) ** 2))
