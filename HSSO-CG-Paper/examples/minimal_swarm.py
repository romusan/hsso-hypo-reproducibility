#!/usr/bin/env python
"""Minimal executable example for the canonical PSO/HSSO implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HSSO-Core" / "src"))
from swarm_core import SwarmConfig, optimize


def shifted_sphere(positions: np.ndarray) -> np.ndarray:
    target = np.array([1.25, -0.75, 2.0])
    return np.sum((positions - target) ** 2, axis=1)


def main() -> None:
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]])
    config = SwarmConfig(particles=30, iterations=100)
    for method in ("PSO", "HSSO-E"):
        result = optimize(method, shifted_sphere, bounds, seed=20260712, config=config)
        print(
            f"{method}: fitness={result['fitness']:.3e}, "
            f"best={np.round(result['best'], 6).tolist()}, "
            f"evaluations={result['evaluations']}"
        )
        if result["fitness"] > 1.0e-6:
            raise SystemExit(f"{method} did not reach the example tolerance")


if __name__ == "__main__":
    main()
