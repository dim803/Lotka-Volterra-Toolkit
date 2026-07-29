"""
visualization.py

Phase-plane and time-series plotting for a Lotka-Volterra model,
generalised to any prey/predator pair via configurable axis labels
and units (not hardcoded to any specific species or dataset).
"""

import numpy as np
import matplotlib.pyplot as plt

from .model import lotka_volterra_rhs, equilibria


def plot_phase_plane(alpha, beta, delta, gamma,
                      prey_sim, predator_sim,
                      prey_data=None, predator_data=None,
                      prey_label="Prey", predator_label="Predator",
                      units="", grid_range=None, ax=None):
    """
    Plots the direction field, a simulated trajectory, and
    (optionally) observed data, in the prey-predator phase plane.

    Parameters
    ----------
    alpha, beta, delta, gamma : float
        The Lotka-Volterra parameters.
    prey_sim, predator_sim : array-like
        The simulated trajectory to plot.
    prey_data, predator_data : array-like, optional
        Observed data to overlay, if available.
    prey_label, predator_label : str
        Axis labels (e.g. "Hares", "Lynx").
    units : str
        Units string appended to axis labels (e.g. "thousands").
    grid_range : tuple of (float, float, float, float), optional
        (prey_min, prey_max, predator_min, predator_max) for the
        direction field and plot limits. Defaults to a range that
        comfortably contains the simulated (and observed, if given)
        data.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Creates a new figure if not provided.

    Returns
    -------
    matplotlib.axes.Axes
    """
    prey_sim = np.asarray(prey_sim, dtype=float)
    predator_sim = np.asarray(predator_sim, dtype=float)

    if grid_range is None:
        all_prey = list(prey_sim)
        all_predator = list(predator_sim)
        if prey_data is not None:
            all_prey += list(prey_data)
            all_predator += list(predator_data)
        pad_prey = 0.1 * (max(all_prey) - min(all_prey) + 1)
        pad_predator = 0.1 * (max(all_predator) - min(all_predator) + 1)
        grid_range = (
            max(0, min(all_prey) - pad_prey), max(all_prey) + pad_prey,
            max(0, min(all_predator) - pad_predator), max(all_predator) + pad_predator
        )

    prey_min, prey_max, predator_min, predator_max = grid_range

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Direction field
    prey_grid = np.linspace(max(prey_min, 0.01), prey_max, 20)
    predator_grid = np.linspace(max(predator_min, 0.01), predator_max, 20)
    P, Q = np.meshgrid(prey_grid, predator_grid)
    dP = alpha * P - beta * P * Q
    dQ = delta * P * Q - gamma * Q
    magnitude = np.sqrt(dP ** 2 + dQ ** 2)
    magnitude[magnitude == 0] = 1
    ax.quiver(P, Q, dP / magnitude, dQ / magnitude, magnitude,
              cmap='Greys', alpha=0.5, pivot='mid')

    # Simulated trajectory
    ax.plot(prey_sim, predator_sim, color='C0', linewidth=1.5,
             label='Simulated trajectory')

    # Observed data
    if prey_data is not None:
        ax.plot(prey_data, predator_data, 'o-', color='C1',
                 markersize=4, linewidth=1, label='Observed data')

    # Equilibria
    extinction, coexistence = equilibria(alpha, beta, delta, gamma)
    ax.plot(*coexistence, 'k*', markersize=12,
             label=f'Equilibrium ({coexistence[0]:.1f}, {coexistence[1]:.1f})')
    ax.plot(*extinction, 'ks', markersize=6, label='Equilibrium (0, 0)')

    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"{prey_label}{unit_str}")
    ax.set_ylabel(f"{predator_label}{unit_str}")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(prey_min, prey_max)
    ax.set_ylim(predator_min, predator_max)

    return ax


def plot_time_series(t_sim, prey_sim, predator_sim,
                       t_data=None, prey_data=None, predator_data=None,
                       prey_label="Prey", predator_label="Predator",
                       time_label="Time", units="", ax=None):
    """
    Plots simulated prey and predator populations over time, with
    observed data overlaid as discrete points if provided.

    Parameters
    ----------
    t_sim : array-like
        Time points of the simulated trajectory.
    prey_sim, predator_sim : array-like
        Simulated population values.
    t_data, prey_data, predator_data : array-like, optional
        Observed data to overlay.
    prey_label, predator_label : str
        Legend/series labels (e.g. "Hares", "Lynx").
    time_label : str
        X-axis label (e.g. "Year").
    units : str
        Units string appended to the y-axis label.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(t_sim, prey_sim, color='C0', label=f'Simulated {prey_label}')
    ax.plot(t_sim, predator_sim, color='C1', label=f'Simulated {predator_label}')

    if t_data is not None:
        ax.scatter(t_data, prey_data, color='C0', marker='o', s=25,
                    label=f'Observed {prey_label}', zorder=5)
        ax.scatter(t_data, predator_data, color='C1', marker='o', s=25,
                    label=f'Observed {predator_label}', zorder=5)

    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(time_label)
    ax.set_ylabel(f"Population{unit_str}")
    ax.legend()

    return ax
