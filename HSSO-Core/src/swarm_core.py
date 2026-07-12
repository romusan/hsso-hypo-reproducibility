"""Canonical swarm optimizers shared by benchmarks and hypocenter relocation.

The same implementation is imported by every experiment.  Objectives accept
an ``(n_particles, n_dimensions)`` array and return one value per particle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SwarmConfig:
    particles: int = 30
    iterations: int = 300
    neighbors: int = 6
    amp_scale: float = 0.010
    amp_exp: float = 5.0
    stall_strength: float = 1.5
    stall_window: int = 25


def _neighbor_best(pos: Array, pbest: Array, pscore: Array, k: int) -> Array:
    out = np.empty_like(pos)
    for i in range(len(pos)):
        nn = np.argsort(np.linalg.norm(pos - pos[i], axis=1))[1 : min(k + 1, len(pos))]
        out[i] = pbest[nn[np.argmin(pscore[nn])]]
    return out


def optimize(
    method: str,
    objective: Callable[[Array], Array],
    bounds: Array,
    seed: int,
    config: SwarmConfig = SwarmConfig(),
    record_history: bool = True,
    target_fitness: float | None = None,
) -> dict:
    """Minimize an objective with one of the four preregistered swarm methods."""
    valid = {"PSO", "LPSO", "HSSO-no-helix", "HSSO-E"}
    if method not in valid:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(valid)}")
    b = np.asarray(bounds, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must have shape (dimensions, 2)")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    n, d = config.particles, len(b)
    span = b[:, 1] - b[:, 0]
    pos = rng.uniform(b[:, 0], b[:, 1], (n, d))
    vel_scale = 0.10 if method in {"PSO", "LPSO"} else 0.08
    vel = rng.uniform(-vel_scale, vel_scale, (n, d)) * span
    val = np.asarray(objective(pos), dtype=float)
    pbest, pscore = pos.copy(), val.copy()
    j = int(np.argmin(pscore))
    gbest, gscore = pbest[j].copy(), float(pscore[j])
    if method.startswith("HSSO"):
        phase = rng.uniform(0.0, 2.0 * np.pi, n)
        spin = rng.uniform(0.75, 1.25, n)
    else:
        phase = spin = None
    stagnation = 0
    history: list[dict] = []

    last_iteration = 0
    for k in range(config.iterations + 1):
        last_iteration = k
        centroid = pos.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((pos - centroid) ** 2, axis=1))))
        if record_history:
            history.append(
                {"iteration": k, "best_objective": gscore, "median_objective": float(np.median(val)),
                 "radius": radius, "positions": pos.copy(), "best_position": gbest.copy()}
            )
        if (target_fitness is not None and gscore <= target_fitness) or k == config.iterations:
            break
        frac = 1.0 - (k + 1) / config.iterations
        if method == "PSO":
            inertia = 0.9 - 0.5 * k / config.iterations
            vel = (inertia * vel + 1.7 * rng.random((n, d)) * (pbest - pos)
                   + 1.7 * rng.random((n, d)) * (gbest - pos))
        elif method == "LPSO":
            lbest = _neighbor_best(pos, pbest, pscore, config.neighbors)
            inertia = 0.9 - 0.5 * k / config.iterations
            vel = (inertia * vel + 1.7 * rng.random((n, d)) * (pbest - pos)
                   + 1.7 * rng.random((n, d)) * (lbest - pos))
        else:
            lbest = _neighbor_best(pos, pbest, pscore, config.neighbors)
            new_velocity = np.empty_like(vel)
            rho = (config.amp_scale * float(np.mean(span)) * frac ** config.amp_exp
                   / (1.0 + config.stall_strength * min(stagnation / config.stall_window, 1.0)))
            for i in range(n):
                u = rng.normal(size=d)
                u /= np.linalg.norm(u) + 1.0e-15
                z = rng.normal(size=d)
                z -= u * np.dot(u, z)
                z /= np.linalg.norm(z) + 1.0e-15
                theta = phase[i] + spin[i] * 2.0 * np.pi * (k + 1) * (0.12 + 0.55 * frac)
                direction = np.cos(theta) * u + np.sin(theta) * z
                if method == "HSSO-E":
                    scaled = span * direction
                    scaled /= np.linalg.norm(scaled) + 1.0e-15
                    helix = rho * scaled
                else:
                    helix = np.zeros(d)
                new_velocity[i] = (
                    0.58 * vel[i] + 1.45 * rng.random(d) * (pbest[i] - pos[i])
                    + 1.05 * rng.random(d) * (lbest[i] - pos[i])
                    + 1.15 * rng.random(d) * (gbest - pos[i]) + helix
                )
            vel = new_velocity
        vmax = 0.20 if method in {"PSO", "LPSO"} else 0.18
        vel = np.clip(vel, -vmax * span, vmax * span)
        pos = np.clip(pos + vel, b[:, 0], b[:, 1])
        val = np.asarray(objective(pos), dtype=float)
        improved = val < pscore
        pbest[improved], pscore[improved] = pos[improved], val[improved]
        j = int(np.argmin(pscore))
        if pscore[j] < gscore - 1.0e-14:
            gbest, gscore, stagnation = pbest[j].copy(), float(pscore[j]), 0
        else:
            stagnation += 1
    return {"best": gbest, "fitness": gscore, "evaluations": n * (last_iteration + 1),
            "iterations_completed": last_iteration,
            "elapsed_s": time.perf_counter() - started, "history": history}
