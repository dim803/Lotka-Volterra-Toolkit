"""
estimation.py

Estimates Lotka-Volterra parameters (alpha, beta, delta, gamma) from
a time series of prey and predator population data, using the
per-capita growth rate linearisation:

    (1/Prey)(dPrey/dt)         = alpha - beta*Predator
    (1/Predator)(dPredator/dt) = delta*Prey - gamma

Both are linear in the parameters, so each can be recovered by
ordinary least squares once the derivatives are estimated numerically
via finite differences.

Works for any evenly-spaced time series (not specific to any species
or dataset) supplied as three arrays: t, prey, predator.
"""

import numpy as np
import statsmodels.api as sm


def estimate_derivatives(t, y):
    """
    Estimates dy/dt at each point in an evenly-spaced time series
    using finite differences: forward difference at the first point,
    backward difference at the last, and central differences
    elsewhere.

    Parameters
    ----------
    t : array-like
        Evenly-spaced time points.
    y : array-like
        Values of y at each time point.

    Returns
    -------
    ndarray
        Estimated dy/dt at each point in t.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    dy_dt = np.zeros(n)

    dt = t[1] - t[0]  # assumes even spacing

    for i in range(n):
        if i == 0:
            dy_dt[i] = (y[i + 1] - y[i]) / dt
        elif i == n - 1:
            dy_dt[i] = (y[i] - y[i - 1]) / dt
        else:
            dy_dt[i] = (y[i + 1] - y[i - 1]) / (2 * dt)

    return dy_dt


def estimate_parameters_regression(t, prey, predator, verbose=False):
    """
    Estimates alpha, beta, delta, gamma from a time series of prey
    and predator populations, via two linear regressions on the
    per-capita growth rates.

    Parameters
    ----------
    t : array-like
        Evenly-spaced time points (e.g. years).
    prey : array-like
        Prey population at each time point.
    predator : array-like
        Predator population at each time point.
    verbose : bool
        If True, prints the full statsmodels regression summaries.

    Returns
    -------
    dict
        Keys: 'alpha', 'beta', 'delta', 'gamma' (point estimates),
        'alpha_se', 'beta_se', 'delta_se', 'gamma_se' (standard
        errors), 'r2_prey', 'r2_predator' (R^2 of each regression),
        'dprey_dt', 'dpredator_dt' (the estimated derivatives),
        'prey_pcr', 'predator_pcr' (the per-capita rates used for
        the regressions).
    """
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)

    dprey_dt = estimate_derivatives(t, prey)
    dpredator_dt = estimate_derivatives(t, predator)

    prey_pcr = dprey_dt / prey          # = alpha - beta*predator
    predator_pcr = dpredator_dt / predator  # = delta*prey - gamma

    # Regression 1: prey_pcr = alpha - beta*predator
    X1 = sm.add_constant(predator)
    model1 = sm.OLS(prey_pcr, X1).fit()
    alpha = model1.params[0]
    beta = -model1.params[1]
    alpha_se = model1.bse[0]
    beta_se = model1.bse[1]
    r2_prey = model1.rsquared

    # Regression 2: predator_pcr = delta*prey - gamma
    X2 = sm.add_constant(prey)
    model2 = sm.OLS(predator_pcr, X2).fit()
    delta = model2.params[1]
    gamma = -model2.params[0]
    delta_se = model2.bse[1]
    gamma_se = model2.bse[0]
    r2_predator = model2.rsquared

    if verbose:
        print(model1.summary())
        print(model2.summary())

    return {
        "alpha": alpha, "alpha_se": alpha_se,
        "beta": beta, "beta_se": beta_se,
        "delta": delta, "delta_se": delta_se,
        "gamma": gamma, "gamma_se": gamma_se,
        "r2_prey": r2_prey, "r2_predator": r2_predator,
        "dprey_dt": dprey_dt, "dpredator_dt": dpredator_dt,
        "prey_pcr": prey_pcr, "predator_pcr": predator_pcr,
    }
