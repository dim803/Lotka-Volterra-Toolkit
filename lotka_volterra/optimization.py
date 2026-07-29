"""
optimization.py

Refines Lotka-Volterra parameters by directly minimising the error
between a simulated trajectory and observed data, rather than fitting
per-capita growth rates (see estimation.py). This targets the actual
quantity of interest -- trajectory accuracy -- at the cost of
requiring a reasonable starting guess and a numerical ODE solve at
every optimisation step.

Two approaches are provided:

    refine_parameters_local  : local search (Levenberg-Marquardt /
                                trust-region reflective), fast, but
                                requires a starting guess and can in
                                principle converge to a poor local
                                minimum.
    refine_parameters_global : global search (differential evolution),
                                does not require a starting guess, but
                                is slower and needs parameter bounds.

Both rely on model.lotka_volterra_rhs and numerical.heun_method to
generate candidate trajectories, so the equations themselves define
the entire space of curves being searched -- this is not a free-form
curve fit.
"""

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from .model import lotka_volterra_rhs
from .numerical import heun_method


def _simulate(params, y0, t0, t_end, h, record_times):
    alpha, beta, delta, gamma = params
    rhs = lotka_volterra_rhs(alpha, beta, delta, gamma)
    return heun_method(rhs, y0, t0, t_end, h, record_times=record_times)


def _residuals(params, t, prey_data, predator_data, y0, h):
    results = _simulate(params, y0, t[0], t[-1], h, record_times=list(t))
    prey_sim = np.array([results[ti][0] for ti in t])
    predator_sim = np.array([results[ti][1] for ti in t])
    return np.concatenate([prey_sim - prey_data, predator_sim - predator_data])


def refine_parameters_local(t, prey_data, predator_data, initial_guess,
                              y0=None, h=0.01):
    """
    Refines (alpha, beta, delta, gamma) by minimising the sum of
    squared errors between a simulated trajectory and the observed
    data, using a local (gradient-based) optimiser starting from
    `initial_guess`.

    Parameters
    ----------
    t : array-like
        Time points at which data was observed (e.g. years).
    prey_data, predator_data : array-like
        Observed population values at each time in `t`.
    initial_guess : tuple of float
        Starting guess (alpha, beta, delta, gamma), e.g. from
        estimation.estimate_parameters_regression.
    y0 : tuple of float, optional
        Initial condition (prey_0, predator_0) for the simulation.
        Defaults to (prey_data[0], predator_data[0]).
    h : float
        Step size used for the ODE solve at each optimisation
        iteration. Should be small enough for an accurate simulation
        but not so small that optimisation becomes impractically
        slow (each iteration requires a full simulation).

    Returns
    -------
    dict
        Keys: 'alpha', 'beta', 'delta', 'gamma' (refined parameters),
        'result' (the raw scipy.optimize.OptimizeResult object).
    """
    t = np.asarray(t, dtype=float)
    prey_data = np.asarray(prey_data, dtype=float)
    predator_data = np.asarray(predator_data, dtype=float)

    if y0 is None:
        y0 = (prey_data[0], predator_data[0])

    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    result = least_squares(
        _residuals, initial_guess, bounds=bounds,
        args=(t, prey_data, predator_data, y0, h)
    )

    alpha, beta, delta, gamma = result.x
    return {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma,
            "result": result}


def refine_parameters_global(t, prey_data, predator_data, bounds,
                               y0=None, h=0.02, seed=42, maxiter=100,
                               popsize=15):
    """
    Refines (alpha, beta, delta, gamma) by minimising the sum of
    squared errors between a simulated trajectory and the observed
    data, using a global (differential evolution) search that does
    not require a starting guess. Useful for confirming that a local
    optimisation has not converged to a poor local minimum.

    Parameters
    ----------
    t : array-like
        Time points at which data was observed.
    prey_data, predator_data : array-like
        Observed population values at each time in `t`.
    bounds : list of (float, float)
        Search bounds for each of [alpha, beta, delta, gamma], e.g.
        [(0.01, 2.0), (0.001, 0.2), (0.001, 0.2), (0.01, 3.0)].
    y0 : tuple of float, optional
        Initial condition. Defaults to (prey_data[0], predator_data[0]).
    h : float
        Step size used for the ODE solve at each iteration.
    seed : int
        Random seed, for reproducibility.
    maxiter, popsize : int
        Passed directly to scipy.optimize.differential_evolution.

    Returns
    -------
    dict
        Keys: 'alpha', 'beta', 'delta', 'gamma', 'result'.
    """
    t = np.asarray(t, dtype=float)
    prey_data = np.asarray(prey_data, dtype=float)
    predator_data = np.asarray(predator_data, dtype=float)

    if y0 is None:
        y0 = (prey_data[0], predator_data[0])

    def sse(params):
        try:
            res = _residuals(params, t, prey_data, predator_data, y0, h)
            if not np.all(np.isfinite(res)):
                return 1e12
            return np.sum(res ** 2)
        except (OverflowError, FloatingPointError):
            return 1e12

    result = differential_evolution(
        sse, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-8, polish=True
    )

    alpha, beta, delta, gamma = result.x
    return {"alpha": alpha, "beta": beta, "delta": delta, "gamma": gamma,
            "result": result}
