#!/usr/bin/env python
"""Mathematical verification using the exact optimizer imported by relocation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "HSSO-CG-Paper" / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "HSSO-Core" / "src"))
from swarm_core import SwarmConfig, optimize

METHODS = ["PSO", "LPSO", "HSSO-no-helix", "HSSO-E"]
SHIFT2 = np.array([1.35, -2.25])
SHIFT10 = np.array([1.35, -2.25, .75, -1.50, 2.10, -.90, 1.80, -2.70, .40, -1.10])

def rastrigin(x, shift):
    y = x - shift
    return 10 * y.shape[-1] + np.sum(y*y - 10*np.cos(2*np.pi*y), axis=-1)

def ackley(x, shift):
    y = x - shift; d = y.shape[-1]
    return -20*np.exp(-.2*np.sqrt(np.sum(y*y, axis=-1)/d))-np.exp(np.sum(np.cos(2*np.pi*y), axis=-1)/d)+20+np.e

def camel(x, _shift):
    a, b = x[..., 0], x[..., 1]
    return (4-2.1*a*a+a**4/3)*a*a+a*b+(-4+4*b*b)*b*b

PROBLEMS = {
    "Rastrigin-2D": (rastrigin, SHIFT2, (-5.12, 5.12), 2, 0.0),
    "Rastrigin-10D": (rastrigin, SHIFT10, (-5.12, 5.12), 10, 0.0),
    "Ackley-10D": (ackley, SHIFT10, (-5.12, 5.12), 10, 0.0),
    "Six-Hump-Camel-2D": (camel, np.array([0.089842, -0.712656]), ((-3, 3), (-2, 2)), 2, -1.031628453),
}

def bounds_array(bounds, d):
    return np.array(bounds, float) if isinstance(bounds[0], tuple) else np.tile(np.array(bounds, float), (d, 1))


def main():
    rows = []
    config = SwarmConfig(particles=30, iterations=300)
    for problem, (fun, optimum, bounds, dimension, fstar) in PROBLEMS.items():
        b = bounds_array(bounds, dimension)
        objective = lambda x, f=fun, o=optimum: np.asarray(f(x, o), float)
        for seed in range(30):
            for method in METHODS:
                run = optimize(method, objective, b, seed, config, record_history=False)
                error_f = abs(run["fitness"] - fstar)
                if problem == "Six-Hump-Camel-2D":
                    error_x = min(np.linalg.norm(run["best"] - optimum), np.linalg.norm(run["best"] + optimum))
                else:
                    error_x = np.linalg.norm(run["best"] - optimum)
                rows.append({"problem": problem, "method": method, "seed": seed,
                             "fitness": run["fitness"], "error_f": error_f, "error_x": error_x,
                             "success": int(error_f < 1.0e-6), "evaluations": run["evaluations"]})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "mathematical_validation_runs.csv", index=False)
    summary = raw.groupby(["problem", "method"]).agg(
        successes=("success", "sum"), median_error_f=("error_f", "median"),
        iqr_error_f=("error_f", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        best_error_f=("error_f", "min"), worst_error_f=("error_f", "max"),
        median_error_x=("error_x", "median")).reset_index()
    summary.to_csv(OUT / "mathematical_validation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
