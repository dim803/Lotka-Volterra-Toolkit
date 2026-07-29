"""
numerical.py

General-purpose numerical ODE solvers (Euler's method and Heun's
method) for any coupled two-variable system

    dy1/dt = f1(y1, y2)
    dy2/dt = f2(y1, y2)

expressed as a single function f(y1, y2) -> (dy1_dt, dy2_dt). These
are not specific to the Lotka-Volterra equations and can be reused
for any two-species (or more generally, two-state-variable) ODE
system by supplying a different `rhs` function.

Also includes a step-size convergence test, applicable to either
solver, which repeatedly halves the step size until the solution at
the final time stops changing beyond a chosen tolerance.
"""

import numpy as np


def euler_method(rhs, y0, t0, t_end, h, record_times=None):
    """
    Solves dy/dt = rhs(y1, y2) using Euler's method.

    Parameters
    ----------
    rhs : callable
        rhs(y1, y2) -> (dy1_dt, dy2_dt)
    y0 : tuple of float
        Initial condition (y1_0, y2_0).
    t0, t_end : float
        Start and end time.
    h : float
        Step size.
    record_times : list of float, optional
        Specific times at which to record (y1, y2). Defaults to
        every integer time step in [t0, t_end].

    Returns
    -------
    dict
        Mapping {time: (y1, y2)} for each requested recorded time.
    """
    if record_times is None:
        record_times = list(range(int(t0), int(t_end) + 1))

    n_steps = round((t_end - t0) / h)
    record_steps = {round((t - t0) / h): t for t in record_times}

    y1, y2 = y0
    results = {}
    if 0 in record_steps:
        results[record_steps[0]] = (y1, y2)

    for n in range(1, n_steps + 1):
        dy1, dy2 = rhs(y1, y2)
        y1 += h * dy1
        y2 += h * dy2
        if n in record_steps:
            results[record_steps[n]] = (y1, y2)

    return results


def heun_method(rhs, y0, t0, t_end, h, record_times=None):
    """
    Solves dy/dt = rhs(y1, y2) using Heun's method (Improved Euler):
    a predictor-corrector scheme that averages the gradient at the
    start and predicted end of each interval.

    Parameters
    ----------
    rhs : callable
        rhs(y1, y2) -> (dy1_dt, dy2_dt)
    y0 : tuple of float
        Initial condition (y1_0, y2_0).
    t0, t_end : float
        Start and end time.
    h : float
        Step size.
    record_times : list of float, optional
        Specific times at which to record (y1, y2). Defaults to
        every integer time step in [t0, t_end].

    Returns
    -------
    dict
        Mapping {time: (y1, y2)} for each requested recorded time.
    """
    if record_times is None:
        record_times = list(range(int(t0), int(t_end) + 1))

    n_steps = round((t_end - t0) / h)
    record_steps = {round((t - t0) / h): t for t in record_times}

    y1, y2 = y0
    results = {}
    if 0 in record_steps:
        results[record_steps[0]] = (y1, y2)

    for n in range(1, n_steps + 1):
        dy1_1, dy2_1 = rhs(y1, y2)
        y1_pred = y1 + h * dy1_1
        y2_pred = y2 + h * dy2_1
        dy1_2, dy2_2 = rhs(y1_pred, y2_pred)
        y1 += h / 2 * (dy1_1 + dy1_2)
        y2 += h / 2 * (dy2_1 + dy2_2)
        if n in record_steps:
            results[record_steps[n]] = (y1, y2)

    return results


def euler_method_full(rhs, y0, t0, t_end, h):
    """
    Same as euler_method, but returns the full trajectory at every
    step (rather than only at requested times), for plotting purposes.

    Returns
    -------
    t : ndarray
    y1 : ndarray
    y2 : ndarray
    """
    n_steps = round((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y1_arr = np.zeros(n_steps + 1)
    y2_arr = np.zeros(n_steps + 1)
    y1_arr[0], y2_arr[0] = y0

    y1, y2 = y0
    for n in range(n_steps):
        dy1, dy2 = rhs(y1, y2)
        y1 += h * dy1
        y2 += h * dy2
        y1_arr[n + 1], y2_arr[n + 1] = y1, y2

    return t, y1_arr, y2_arr


def heun_method_full(rhs, y0, t0, t_end, h):
    """
    Same as heun_method, but returns the full trajectory at every
    step (rather than only at requested times), for plotting purposes.

    Returns
    -------
    t : ndarray
    y1 : ndarray
    y2 : ndarray
    """
    n_steps = round((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y1_arr = np.zeros(n_steps + 1)
    y2_arr = np.zeros(n_steps + 1)
    y1_arr[0], y2_arr[0] = y0

    y1, y2 = y0
    for n in range(n_steps):
        dy1_1, dy2_1 = rhs(y1, y2)
        y1_pred = y1 + h * dy1_1
        y2_pred = y2 + h * dy2_1
        dy1_2, dy2_2 = rhs(y1_pred, y2_pred)
        y1 += h / 2 * (dy1_1 + dy1_2)
        y2 += h / 2 * (dy2_1 + dy2_2)
        y1_arr[n + 1], y2_arr[n + 1] = y1, y2

    return t, y1_arr, y2_arr


def find_convergent_step(solver, rhs, y0, t0, t_end,
                          h_start=1.0, tol=1e-4, min_h=1e-8, verbose=False):
    """
    Repeatedly halves the step size h and compares the solution at
    t_end between successive halvings, using the given solver
    (euler_method or heun_method). Returns the step size at which
    the change first falls below the given tolerance, i.e. the
    solution is considered to have numerically converged.

    Parameters
    ----------
    solver : callable
        Either euler_method or heun_method (or any function with the
        same signature).
    rhs : callable
        rhs(y1, y2) -> (dy1_dt, dy2_dt)
    y0 : tuple of float
        Initial condition.
    t0, t_end : float
        Start and end time.
    h_start : float
        Initial step size to begin halving from.
    tol : float
        Convergence tolerance: the algorithm stops once the change
        in solution between successive halvings is below this value.
    min_h : float
        A floor on how small h is allowed to shrink, to prevent an
        infinite loop if convergence is never reached.
    verbose : bool
        If True, prints the convergence table as it is computed.

    Returns
    -------
    float
        The converged step size.
    """
    h = h_start
    prev = solver(rhs, y0, t0, t_end, h, record_times=[t_end])[t_end]

    if verbose:
        print(f"{'h':>12} {'y1(t_end)':>10} {'y2(t_end)':>10} {'change':>10}")

    while h > min_h:
        h_next = h / 2
        y1_next, y2_next = solver(rhs, y0, t0, t_end, h_next,
                                    record_times=[t_end])[t_end]

        if not (np.isfinite(y1_next) and np.isfinite(y2_next)):
            if verbose:
                print(f"{h_next:12.6f} {'diverged':>10} {'':>10} {'':>10}")
            h = h_next
            prev = (y1_next, y2_next)
            continue

        change = np.sqrt((y1_next - prev[0]) ** 2 + (y2_next - prev[1]) ** 2)
        if verbose:
            print(f"{h_next:12.6f} {y1_next:10.4f} {y2_next:10.4f} {change:10.6f}")

        if change < tol:
            return h_next

        h = h_next
        prev = (y1_next, y2_next)

    return h
