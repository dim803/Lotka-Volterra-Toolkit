"""
model.py

The Lotka-Volterra predator-prey equations:

    dPrey/dt     = alpha * Prey - beta * Prey * Predator
    dPredator/dt = delta * Prey * Predator - gamma * Predator

This module defines the right-hand side of the system, its two
equilibria, and its conserved quantity , for any
choice of the four parameters.
"""

import numpy as np


def lotka_volterra_rhs(alpha, beta, delta, gamma):
    """
    Returns a function f(prey, predator) -> (d_prey/dt, d_predator/dt)
    for the Lotka-Volterra system with the given parameters.

    Parameters
    ----------
    alpha : float
        Intrinsic per-capita growth rate of the prey species.
    beta : float
        Rate at which predator-prey encounters remove prey.
    delta : float
        Efficiency with which consumed prey are converted into new
        predators.
    gamma : float
        Intrinsic per-capita death rate of the predator species.

    Returns
    -------
    f : callable
        f(prey, predator) -> (d_prey_dt, d_predator_dt)
    """
    def f(prey, predator):
        d_prey_dt = alpha * prey - beta * prey * predator
        d_predator_dt = delta * prey * predator - gamma * predator
        return d_prey_dt, d_predator_dt
    return f


def equilibria(alpha, beta, delta, gamma):
    """
    Returns the two equilibrium points of the Lotka-Volterra system:
    the trivial equilibrium (extinction of both species) and the
    coexistence equilibrium.

    Returns
    -------
    (prey_extinct, predator_extinct) : tuple
        Always (0.0, 0.0).
    (prey_star, predator_star) : tuple
        The coexistence equilibrium (gamma/delta, alpha/beta).
    """
    extinction = (0.0, 0.0)
    coexistence = (gamma / delta, alpha / beta)
    return extinction, coexistence


def conserved_quantity(prey, predator, alpha, beta, delta, gamma):
    """
    Evaluates the conserved quantity

        V(Prey, Predator) = delta*Prey - gamma*ln(Prey)
                             + beta*Predator - alpha*ln(Predator)

    which is constant along any true trajectory of the Lotka-Volterra
    system. Useful for checking how well a numerical simulation
    conserves this quantity (a proxy for numerical accuracy).

    Parameters
    ----------
    prey, predator : float or array-like
        Population values at which to evaluate V. Must be > 0.

    Returns
    -------
    V : float or array-like
    """
    prey = np.asarray(prey, dtype=float)
    predator = np.asarray(predator, dtype=float)
    return (delta * prey - gamma * np.log(prey)
            + beta * predator - alpha * np.log(predator))
