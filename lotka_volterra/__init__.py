"""
lotka-volterra-toolkit
=======================

A general-purpose toolkit for fitting, simulating, and evaluating
Lotka-Volterra predator-prey models against real time series data.

Modules
-------
model          : the LV equations, equilibria, and the conserved quantity
numerical      : general-purpose Euler's and Heun's method ODE solvers,
                 usable for any system of ODEs, not just LV
estimation     : parameter estimation via per-capita linear regression
optimization   : trajectory-based parameter refinement (local + global)
metrics        : error metrics (RMSE)
visualization  : phase-plane and time-series plotting
"""

from .model import lotka_volterra_rhs, equilibria, conserved_quantity
from .numerical import (
    euler_method,
    heun_method,
    euler_method_full,
    heun_method_full,
    find_convergent_step,
)
from .estimation import estimate_parameters_regression
from .optimization import refine_parameters_local, refine_parameters_global
from .metrics import rmse
from .visualization import plot_phase_plane, plot_time_series

__all__ = [
    "lotka_volterra_rhs",
    "equilibria",
    "conserved_quantity",
    "euler_method",
    "heun_method",
    "euler_method_full",
    "heun_method_full",
    "find_convergent_step",
    "estimate_parameters_regression",
    "refine_parameters_local",
    "refine_parameters_global",
    "rmse",
    "plot_phase_plane",
    "plot_time_series",
]

__version__ = "0.1.0"
