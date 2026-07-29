"""
hare_lynx_example.py

A full worked example of the lotka-volterra-toolkit pipeline, using
the classic Hudson's Bay Company hare-lynx dataset (1900-1920).

Run from the repository root with:
    python examples/hare_lynx_example.py
"""

import pandas as pd
import numpy as np

from lotka_volterra import (
    lotka_volterra_rhs,
    equilibria,
    heun_method,
    heun_method_full,
    find_convergent_step,
    estimate_parameters_regression,
    refine_parameters_local,
    refine_parameters_global,
    rmse,
    plot_phase_plane,
    plot_time_series,
)
import matplotlib.pyplot as plt


def main():
    # ---- 1. Load data ----
    df = pd.read_csv("examples/data/hare_lynx.csv")
    t = df["Year"].to_numpy()
    prey_data = df["Hare"].to_numpy()
    predator_data = df["Lynx"].to_numpy()

    # ---- 2. Estimate parameters via per-capita regression ----
    reg = estimate_parameters_regression(t, prey_data, predator_data, verbose=False)
    print("Regression-based parameters:")
    print(f"  alpha = {reg['alpha']:.4f} +/- {reg['alpha_se']:.4f}")
    print(f"  beta  = {reg['beta']:.4f} +/- {reg['beta_se']:.4f}")
    print(f"  delta = {reg['delta']:.4f} +/- {reg['delta_se']:.4f}")
    print(f"  gamma = {reg['gamma']:.4f} +/- {reg['gamma_se']:.4f}")
    print(f"  R^2 (prey)     = {reg['r2_prey']:.4f}")
    print(f"  R^2 (predator) = {reg['r2_predator']:.4f}\n")

    initial_guess = (reg["alpha"], reg["beta"], reg["delta"], reg["gamma"])

    # ---- 3. Find a suitable step size for Heun's method ----
    rhs = lotka_volterra_rhs(*initial_guess)
    y0 = (prey_data[0], predator_data[0])
    best_h = find_convergent_step(heun_method, rhs, y0, t[0], t[-1],
                                    h_start=1.0, tol=1e-4)
    print(f"Converged step size (Heun's method): h = {best_h:.6f}\n")

    # ---- 4. Simulate with the regression-based parameters ----
    results = heun_method(rhs, y0, t[0], t[-1], best_h, record_times=list(t))
    prey_sim = np.array([results[ti][0] for ti in t])
    predator_sim = np.array([results[ti][1] for ti in t])

    rmse_prey = rmse(prey_sim, prey_data)
    rmse_predator = rmse(predator_sim, predator_data)
    print(f"RMSE with regression parameters: prey={rmse_prey:.2f}, predator={rmse_predator:.2f}\n")

    # ---- 5. Refine parameters by directly minimising trajectory RMSE ----
    refined = refine_parameters_local(t, prey_data, predator_data, initial_guess, y0=y0, h=0.01)
    print("Refined parameters (local optimisation):")
    print(f"  alpha = {refined['alpha']:.4f}")
    print(f"  beta  = {refined['beta']:.4f}")
    print(f"  delta = {refined['delta']:.4f}")
    print(f"  gamma = {refined['gamma']:.4f}\n")

    refined_rhs = lotka_volterra_rhs(refined["alpha"], refined["beta"],
                                       refined["delta"], refined["gamma"])
    results_refined = heun_method(refined_rhs, y0, t[0], t[-1], best_h, record_times=list(t))
    prey_sim_refined = np.array([results_refined[ti][0] for ti in t])
    predator_sim_refined = np.array([results_refined[ti][1] for ti in t])

    rmse_prey_r = rmse(prey_sim_refined, prey_data)
    rmse_predator_r = rmse(predator_sim_refined, predator_data)
    print(f"RMSE with refined parameters: prey={rmse_prey_r:.2f}, predator={rmse_predator_r:.2f}\n")

    # ---- 6. Confirm with a global search (optional, slower) ----
    bounds = [(0.01, 2.0), (0.001, 0.2), (0.001, 0.2), (0.01, 3.0)]
    global_result = refine_parameters_global(t, prey_data, predator_data, bounds, y0=y0)
    print("Refined parameters (global optimisation, for comparison):")
    print(f"  alpha = {global_result['alpha']:.4f}")
    print(f"  beta  = {global_result['beta']:.4f}")
    print(f"  delta = {global_result['delta']:.4f}")
    print(f"  gamma = {global_result['gamma']:.4f}\n")

    # ---- 7. Plot results ----
    t_full, prey_full, predator_full = heun_method_full(
        refined_rhs, y0, t[0], t[-1], best_h
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    plot_time_series(
        t_full, prey_full, predator_full,
        t_data=t, prey_data=prey_data, predator_data=predator_data,
        prey_label="Hares", predator_label="Lynx",
        time_label="Year", units="thousands", ax=axes[0]
    )
    axes[0].set_title("Time Series: Refined Model vs. Observed Data")

    plot_phase_plane(
        refined["alpha"], refined["beta"], refined["delta"], refined["gamma"],
        prey_full, predator_full,
        prey_data=prey_data, predator_data=predator_data,
        prey_label="Hares", predator_label="Lynx", units="thousands",
        ax=axes[1]
    )
    axes[1].set_title("Phase Plane: Refined Model vs. Observed Data")

    plt.tight_layout()
    plt.savefig("examples/hare_lynx_results.png", dpi=150)
    print("Saved plot to examples/hare_lynx_results.png")
    plt.show()


if __name__ == "__main__":
    main()
