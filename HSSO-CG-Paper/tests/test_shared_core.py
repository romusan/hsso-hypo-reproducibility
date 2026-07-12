"""Minimal invariants for the shared optimizer implementation."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HSSO-Core/src"))
from swarm_core import SwarmConfig, optimize


def sphere(x):
    return np.sum(x*x, axis=1)


def test_shared_initial_positions():
    bounds = np.array([[-3., 3.], [-7., 7.], [0., 20.]])
    cfg = SwarmConfig(particles=12, iterations=2)
    starts = []
    for method in ["PSO", "LPSO", "HSSO-no-helix", "HSSO-E"]:
        starts.append(optimize(method, sphere, bounds, 42, cfg)["history"][0]["positions"])
    assert all(np.array_equal(starts[0], x) for x in starts[1:])


def test_energy_normalization_identity():
    rng = np.random.default_rng(9)
    span = np.array([1., 10., 100.])
    rho = 2.75
    for _ in range(100):
        q = rng.normal(size=3); q /= np.linalg.norm(q)
        scaled = span*q; scaled /= np.linalg.norm(scaled)
        h = rho*scaled
        assert np.isclose(np.dot(h, h), rho*rho, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    test_shared_initial_positions(); test_energy_normalization_identity()
    print("shared-core invariants: PASS")
