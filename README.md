# lotka-volterra-toolkit

A general-purpose Python toolkit for fitting, simulating, and
evaluating Lotka-Volterra predator-prey models against real time
series data. Built for an IB Mathematics AA HL Internal Assessment
modelling the Hudson's Bay Company lynx-hare population cycle, and
generalised here for reuse with any predator-prey dataset.

## Features

- **Model**: the Lotka-Volterra equations, their two equilibria, and
  the conserved quantity `V(prey, predator)`.
- **Numerical methods**: general-purpose Euler's method and Heun's
  method (Improved Euler) ODE solvers, usable for any two-variable
  coupled system, not just Lotka-Volterra.
- **Convergence testing**: a step-size convergence test that
  repeatedly halves `h` until the solution stops changing beyond a
  chosen tolerance.
- **Parameter estimation**: recovers `(alpha, beta, delta, gamma)`
  from time series data via the per-capita growth rate linearisation,
  reducing the nonlinear estimation problem to two ordinary linear
  regressions.
- **Trajectory refinement**: refines parameters by directly
  minimising RMSE between the simulated trajectory and observed data,
  via both local (gradient-based) and global (differential evolution)
  optimisation.
- **Visualisation**: phase-plane plots (direction field + simulated
  trajectory + observed data) and time-series comparison plots, with
  configurable species names and units.

## Installation

```bash
git clone https://github.com/<your-username>/lotka-volterra-toolkit.git
cd lotka-volterra-toolkit
pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
import numpy as np
from lotka_volterra import (
    lotka_volterra_rhs, heun_method, find_convergent_step,
    estimate_parameters_regression, refine_parameters_local, rmse,
)

# Load your own data: columns of time, prey population, predator population
df = pd.read_csv("your_data.csv")
t = df["Year"].to_numpy()
prey = df["Prey"].to_numpy()
predator = df["Predator"].to_numpy()

# 1. Estimate initial parameters via regression
reg = estimate_parameters_regression(t, prey, predator)
initial_guess = (reg["alpha"], reg["beta"], reg["delta"], reg["gamma"])

# 2. Find a stable, accurate step size
rhs = lotka_volterra_rhs(*initial_guess)
y0 = (prey[0], predator[0])
h = find_convergent_step(heun_method, rhs, y0, t[0], t[-1])

# 3. Simulate
results = heun_method(rhs, y0, t[0], t[-1], h, record_times=list(t))
prey_sim = np.array([results[ti][0] for ti in t])
predator_sim = np.array([results[ti][1] for ti in t])
print("RMSE:", rmse(prey_sim, prey), rmse(predator_sim, predator))

# 4. Refine parameters against the trajectory directly
refined = refine_parameters_local(t, prey, predator, initial_guess, y0=y0)
print(refined)
```

See `examples/hare_lynx_example.py` for a complete worked example
using the classic 1900-1920 Hudson's Bay Company hare-lynx dataset,
including plotting.

## Package structure

```
lotka_volterra/
    model.py          # LV equations, equilibria, conserved quantity
    numerical.py       # Euler's and Heun's method solvers, convergence testing
    estimation.py       # per-capita regression parameter estimation
    optimization.py     # local and global trajectory-based refinement
    metrics.py           # RMSE
    visualization.py     # phase-plane and time-series plotting
examples/
    hare_lynx_example.py  # full worked example
    data/hare_lynx.csv     # the classic 1900-1920 dataset
```

## Mathematical background

The Lotka-Volterra system models a predator-prey pair as

```
d(prey)/dt     = alpha * prey - beta * prey * predator
d(predator)/dt = delta * prey * predator - gamma * predator
```

with two equilibria, `(0, 0)` and `(gamma/delta, alpha/beta)`, and a
conserved quantity

```
V(prey, predator) = delta*prey - gamma*ln(prey) + beta*predator - alpha*ln(predator)
```

that is constant along any true trajectory, which is what makes
orbits in the phase plane closed loops around the coexistence
equilibrium. The system has no known closed-form solution for
`prey(t)` and `predator(t)` in terms of elementary functions, which is
why numerical methods (Euler's, Heun's) are necessary rather than a
convenience.

## Notes on parameter estimation

Two estimation strategies are provided, and they answer genuinely
different questions:

- `estimate_parameters_regression` minimises error in the *per-capita
  growth rates*. This is a well-posed linear regression problem with
  a closed-form solution, but small errors here can compound
  substantially once integrated forward over a long time period,
  since the model is nonlinear.
- `refine_parameters_local` / `refine_parameters_global` minimise
  error in the *simulated trajectory* directly, which is usually the
  quantity of actual interest, at the cost of requiring iterative
  numerical optimisation (and, for the local method, a reasonable
  starting guess -- typically the output of the regression above).

## License

MIT
