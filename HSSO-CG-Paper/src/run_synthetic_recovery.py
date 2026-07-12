#!/usr/bin/env python
"""Compact synthetic recovery test with real event geometries and cached FSM fields.

The experiment is intentionally diagnostic.  It uses the same velocity grid for
generation and inversion, but adds pick noise and unmodelled receiver delays so
that exact recovery is not guaranteed.  It therefore tests optimizer and
geometry recoverability, not velocity-model accuracy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CGDIR = ROOT / "HSSO-CG-Paper"
HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Core" / "src"))
from swarm_core import SwarmConfig, optimize


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hypo = load_module("synthetic_hypo", HYPODIR / "src" / "hsso_hypo_relocation.py")
comparison = load_module(
    "synthetic_comparison", HYPODIR / "src" / "run_shared_optimizer_comparison.py"
)
METHODS = ["PSO", "LPSO", "HSSO-no-helix", "HSSO-E"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth-window-km", type=float, default=20.0)
    parser.add_argument("--pick-noise-s", type=float, default=0.05)
    parser.add_argument("--receiver-delay-s", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--heldout-statics-csv",
        default=str(HYPODIR / "data" / "sgc_new_30" / "heldout_station_statics.csv"),
    )
    parser.add_argument(
        "--output-prefix", default=str(CGDIR / "results" / "synthetic_recovery")
    )
    return parser.parse_args()


def representative_events(catalog: pd.DataFrame) -> pd.DataFrame:
    """Select unique events nearest the 20th, 50th, and 80th depth percentiles."""
    depth = catalog["source_z_km"].to_numpy(float)
    chosen: list[int] = []
    for quantile in (0.2, 0.5, 0.8):
        target = float(np.quantile(depth, quantile))
        order = np.argsort(np.abs(depth - target))
        chosen.append(next(int(i) for i in order if int(i) not in chosen))
    return catalog.iloc[chosen].copy()


def search_bounds(event: pd.Series, receivers: np.ndarray, model, depth_window: float) -> np.ndarray:
    center = np.array([event.source_x_km, event.source_y_km, event.source_z_km], float)
    rxy, _ = hypo.event_search_radii(event, receivers)
    return np.array(
        [
            [max(model.xmin, center[0] - 2.5 * rxy), min(model.xmax, center[0] + 2.5 * rxy)],
            [max(model.ymin, center[1] - 2.5 * rxy), min(model.ymax, center[1] + 2.5 * rxy)],
            [max(model.zmin + 0.5, center[2] - depth_window), min(model.zmax, center[2] + depth_window)],
        ]
    )


def load_cached_fields(model, picks: pd.DataFrame):
    cache = hypo.FsmFieldCache(model, 8)
    cache_dir = HYPODIR / "data" / "fsm_field_cache_sweeps8"
    for rec in picks[["receiver_x_km", "receiver_y_km"]].drop_duplicates().itertuples(index=False):
        xyz = np.array([rec.receiver_x_km, rec.receiver_y_km, 0.0])
        key = hypo.coord_to_index_xyz(*xyz, model)
        cache_file = cache_dir / ("field_k%02d_j%02d_i%02d.npy" % key)
        if cache_file.exists():
            cache.fields[key] = np.load(cache_file)
        else:
            cache.fields[key] = cache.field_for_receiver(xyz)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, cache.fields[key])
    return cache


def main() -> None:
    args = arguments()
    blocks = []
    for dataset in ("stratified32", "new30"):
        catalog, picks = comparison.load_dataset(dataset)
        blocks.append((representative_events(catalog), picks))
    catalog = pd.concat([block[0] for block in blocks], ignore_index=True)
    picks = pd.concat([block[1] for block in blocks], ignore_index=True)

    statics = pd.read_csv(args.heldout_statics_csv)
    static_map = statics.drop_duplicates("station").set_index("station")["station_static_s"]
    new_mask = picks.dataset == "new30"
    picks.loc[new_mask, "station_static_s"] = (
        picks.loc[new_mask, "station"].map(static_map).fillna(0.0)
    )

    model = hypo.load_velocity_model()
    cache = load_cached_fields(model, picks)
    config = SwarmConfig(particles=args.particles, iterations=args.iterations)
    offsets = np.array(
        [[6.0, -4.0, 8.0], [-5.0, 6.0, -7.0], [7.0, 5.0, 10.0],
         [-6.0, -5.0, -8.0], [5.0, 7.0, 7.0], [-7.0, 4.0, -10.0]]
    )
    rows = []

    for case_index, event in catalog.iterrows():
        event_picks = picks[
            (picks.dataset == event.dataset) & (picks.event_id == event.event_id)
        ].drop_duplicates("station").copy()
        receivers = np.column_stack(
            [event_picks.receiver_x_km.to_numpy(float),
             event_picks.receiver_y_km.to_numpy(float),
             np.zeros(len(event_picks))]
        )
        known_statics = event_picks.station_static_s.to_numpy(float)
        operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
        bounds = search_bounds(event, receivers, model, args.depth_window_km)
        center = np.array([event.source_x_km, event.source_y_km, event.source_z_km], float)
        margin = np.minimum(0.5, 0.1 * (bounds[:, 1] - bounds[:, 0]))
        true_source = np.clip(center + offsets[case_index], bounds[:, 0] + margin, bounds[:, 1] - margin)

        case_seed = int((args.seed + zlib.crc32(str(event.event_id).encode())) % (2**32 - 1))
        rng = np.random.default_rng(case_seed)
        pick_noise = rng.normal(0.0, args.pick_noise_s, len(receivers))
        unmodelled_delay = rng.normal(0.0, args.receiver_delay_s, len(receivers))
        unmodelled_delay -= np.median(unmodelled_delay)
        observed = operator.predict(true_source) + known_statics + pick_noise + unmodelled_delay
        objective = lambda positions: comparison.evaluate_matrix(
            positions, operator, observed, known_statics, huber_delta=1.0
        )

        for method in METHODS:
            run = optimize(method, objective, bounds, case_seed, config)
            recovered = np.asarray(run["best"], float)
            horizontal_error = float(np.linalg.norm(recovered[:2] - true_source[:2]))
            depth_error = float(recovered[2] - true_source[2])
            error_3d = float(np.linalg.norm(recovered - true_source))
            _, rms, _, _, _ = hypo.residual_solution(
                recovered, observed, receivers, model, 2.0,
                operator=operator, station_static_s=known_statics
            )
            rows.append(
                {
                    "case": int(case_index + 1), "dataset": event.dataset,
                    "geometry_event_id": event.event_id, "method": method,
                    "seed": case_seed, "n_stations": len(receivers),
                    "pick_noise_sigma_s": args.pick_noise_s,
                    "unmodelled_receiver_delay_sigma_s": args.receiver_delay_s,
                    "true_x_km": true_source[0], "true_y_km": true_source[1],
                    "true_depth_km": true_source[2], "recovered_x_km": recovered[0],
                    "recovered_y_km": recovered[1], "recovered_depth_km": recovered[2],
                    "horizontal_error_km": horizontal_error, "depth_error_km": depth_error,
                    "absolute_depth_error_km": abs(depth_error), "error_3d_km": error_3d,
                    "recovered_within_5km": bool(error_3d <= 5.0),
                    "terminal_rms_s": rms, "evaluations": run["evaluations"],
                    "elapsed_s": run["elapsed_s"],
                }
            )
            print(
                f"case {case_index + 1} {event.dataset} {method}: "
                f"3-D error={error_3d:.2f} km, RMS={rms:.3f} s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(prefix.with_name(prefix.name + "_runs.csv"), index=False)
    summary = (
        results.groupby("method", sort=False)
        .agg(
            cases=("case", "size"),
            median_horizontal_error_km=("horizontal_error_km", "median"),
            median_absolute_depth_error_km=("absolute_depth_error_km", "median"),
            median_3d_error_km=("error_3d_km", "median"),
            max_3d_error_km=("error_3d_km", "max"),
            recovered_within_5km=("recovered_within_5km", "sum"),
            median_terminal_rms_s=("terminal_rms_s", "median"),
        )
        .reset_index()
    )
    summary.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    metadata = vars(args) | {
        "cases": 6,
        "selection": "three depth-quantile geometries from each seismic data set",
        "forward_inverse_model": "same fixed 3-D Vp grid and cached receiver-centred FSM fields",
        "interpretation": "conditional algorithmic recovery; not independent velocity-model validation",
    }
    prefix.with_name(prefix.name + "_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
