# Matematické modelování – Final Project

Minimal Python project for simulating a short-term glucose–insulin model and producing plots/tables.

## Project structure
.
├─ .vscode/                # Editor settings
├─ results_extras/         # Extra figures/CSV outputs from optional analyses
├─ results_modular/        # Main results folder (plots, CSVs)
├─ analytics.py            # Post-processing & summary metrics over simulation outputs
├─ experiments.py          # Batch runs (e.g., parameter sweeps, scenario loops)
├─ inputs.py               # External inputs / stimuli (e.g., injections, pulses)
├─ model.py                # Core ODE/dynamics and model runner
├─ params.py               # Central place for default parameters (constants, steps)
├─ plotting.py             # Standard plotting helpers (time series, comparisons)
├─ plotting_extras.py      # Additional / fancier figures (e.g., flux stacks, planes)
├─ processes.py            # Physiological sub-process computations (fluxes, rates)
├─ run_ivgtt.py            # Main entry point to run a single simulation
├─ scenarios.py            # Predefined scenario builders (baseline, perturbed, etc.)
├─ state.py                # State representation & initial conditions
└─ utils.py                # Small shared utilities (I/O, paths, interpolation)




