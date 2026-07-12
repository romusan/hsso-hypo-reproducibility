#!/usr/bin/env python
"""Fair FSM hypocenter comparison using the canonical shared swarm core."""
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
HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Core" / "src"))
from swarm_core import SwarmConfig, optimize

spec = importlib.util.spec_from_file_location("hypo", HYPODIR / "src" / "hsso_hypo_relocation.py")
hypo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hypo
spec.loader.exec_module(hypo)

METHODS = ["PSO", "LPSO", "HSSO-no-helix", "HSSO-E"]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["stratified32", "new30", "all"], default="all")
    p.add_argument("--particles", type=int, default=30)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--depth-window-km", type=float, default=20.0)
    p.add_argument("--fsm-sweeps", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--heldout-statics-csv", default="")
    p.add_argument("--output", default=str(HYPODIR / "results" / "shared_core_fsm"))
    return p.parse_args()


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name == "stratified32":
        catalog = pd.read_csv(HYPODIR / "data" / "selected_events_rms_lt1_for_hsso.csv")
        picks = pd.read_csv(HYPODIR / "data" / "selected_p_picks_for_hsso.csv")
        picks["station_static_s"] = 0.0
    else:
        base = HYPODIR / "data" / "sgc_new_30"
        catalog = pd.read_csv(base / "sgc_new_events_30_catalog_for_hsso.csv")
        picks = pd.read_csv(base / "sgc_new_events_30_p_picks_for_hsso.csv")
        picks["station_static_s"] = pd.to_numeric(picks.get("station_static_s", 0.0), errors="coerce").fillna(0.0)
    catalog["dataset"] = name
    picks["dataset"] = name
    return catalog, picks


def evaluate_matrix(positions, operator, obs, statics, huber_delta=1.0):
    pred = np.column_stack([hypo._interp_grid_field(positions, field, operator.model) for field in operator.fields])
    pred += statics[None, :]
    shifts = np.median(obs[None, :] - pred, axis=1)
    residual = obs[None, :] - pred - shifts[:, None]
    absolute = np.abs(residual)
    huber = np.where(absolute <= huber_delta, residual**2,
                     2.0 * huber_delta * absolute - huber_delta**2)
    return np.sqrt(np.mean(huber, axis=1))


def solve_event(event, event_picks, model, cache, method, config, depth_window, seed):
    event_picks = event_picks.drop_duplicates("station").copy()
    obs = event_picks["travel_time_s"].to_numpy(float)
    statics = event_picks["station_static_s"].to_numpy(float)
    receivers = np.column_stack([
        event_picks["receiver_x_km"].to_numpy(float),
        event_picks["receiver_y_km"].to_numpy(float),
        np.zeros(len(event_picks)),
    ])
    center = np.array([event.source_x_km, event.source_y_km, event.source_z_km], float)
    operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
    rxy, _ = hypo.event_search_radii(event, receivers)
    zlo = max(model.zmin + 0.5, center[2] - depth_window)
    zhi = min(model.zmax, center[2] + depth_window)
    bounds = np.array([
        [max(model.xmin, center[0] - 2.5 * rxy), min(model.xmax, center[0] + 2.5 * rxy)],
        [max(model.ymin, center[1] - 2.5 * rxy), min(model.ymax, center[1] + 2.5 * rxy)],
        [zlo, zhi],
    ])
    objective = lambda x: evaluate_matrix(x, operator, obs, statics)
    run = optimize(method, objective, bounds, seed, config)
    best = run["best"]
    obj, rms, origin_shift, residual, pred = hypo.residual_solution(
        best, obs, receivers, model, 2.0, operator=operator, station_static_s=statics
    )
    _, sgc_rms, _, _, _ = hypo.residual_solution(
        center, obs, receivers, model, 2.0, operator=operator, station_static_s=statics
    )
    lat, lon = hypo.xy_to_latlon(best[0], best[1])
    dh = float(np.linalg.norm(best[:2] - center[:2]))
    dz = float(best[2] - center[2])
    tol = max(0.25, 0.01 * (bounds[2, 1] - bounds[2, 0]))
    row = {
        "dataset": event.dataset, "event_id": event.event_id, "method": method, "seed": seed,
        "evaluations": run["evaluations"], "optimizer_elapsed_s": run["elapsed_s"], "n_stations": len(event_picks),
        "sgc_lat": event.event_lat, "sgc_lon": event.event_lon, "sgc_depth_km": event.event_depth_km,
        "relocated_lat": lat, "relocated_lon": lon, "relocated_depth_km": best[2],
        "origin_shift_s": origin_shift, "sgc_model_rms_s": sgc_rms, "relocated_rms_s": rms,
        "huber_objective_s": obj, "rms_reduction_s": sgc_rms - rms,
        "delta_horizontal_km": dh, "delta_depth_km": dz,
        "depth_lower_km": bounds[2, 0], "depth_upper_km": bounds[2, 1],
        "depth_boundary": bool(best[2] <= bounds[2, 0] + tol or best[2] >= bounds[2, 1] - tol),
        "large_horizontal_shift": bool(dh >= 25.0), "large_vertical_shift": bool(abs(dz) >= 15.0),
    }
    history = pd.DataFrame([{k: v for k, v in h.items() if k not in {"positions", "best_position"}}
                            for h in run["history"]])
    history.insert(0, "method", method); history.insert(0, "event_id", event.event_id)
    history.insert(0, "dataset", event.dataset)
    return row, history


def main():
    args = arguments()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset != "all" else ["stratified32", "new30"]
    blocks = [load_dataset(d) for d in datasets]
    catalog = pd.concat([b[0] for b in blocks], ignore_index=True)
    picks = pd.concat([b[1] for b in blocks], ignore_index=True)
    if args.heldout_statics_csv:
        statics = pd.read_csv(args.heldout_statics_csv)
        mapping = statics.set_index("station")["station_static_s"]
        mask = picks.dataset == "new30"
        picks.loc[mask, "station_static_s"] = picks.loc[mask, "station"].map(mapping).fillna(0.0)
    if args.max_events:
        catalog = catalog.iloc[: args.max_events]
    model = hypo.load_velocity_model()
    cache = hypo.FsmFieldCache(model, args.fsm_sweeps)
    cache_dir = HYPODIR / "data" / "fsm_field_cache_sweeps8"
    cache_dir.mkdir(parents=True, exist_ok=True)
    receiver_rows = picks[["receiver_x_km", "receiver_y_km"]].drop_duplicates()
    for rec in receiver_rows.itertuples(index=False):
        xyz = np.array([rec.receiver_x_km, rec.receiver_y_km, 0.0])
        key = hypo.coord_to_index_xyz(*xyz, model)
        cache_file = cache_dir / ("field_k%02d_j%02d_i%02d.npy" % key)
        if key in cache.fields:
            continue
        if cache_file.exists():
            cache.fields[key] = np.load(cache_file)
        else:
            cache.fields[key] = cache.field_for_receiver(xyz)
            np.save(cache_file, cache.fields[key])
    config = SwarmConfig(particles=args.particles, iterations=args.iterations)
    rows, histories = [], []
    for _, event in catalog.iterrows():
        ep = picks[(picks.dataset == event.dataset) & (picks.event_id == event.event_id)]
        shared_seed = int((args.seed + zlib.crc32(str(event.event_id).encode())) % (2**32 - 1))
        for method in METHODS:
            row, history = solve_event(event, ep, model, cache, method, config,
                                       args.depth_window_km, shared_seed)
            rows.append(row); histories.append(history)
            print(f"{event.dataset} {event.event_id} {method}: RMS={row['relocated_rms_s']:.4f} s, "
                  f"dH={row['delta_horizontal_km']:.1f} km, dZ={row['delta_depth_km']:.1f} km", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "relocation_comparison.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(out / "convergence_history.csv", index=False)
    config_payload = vars(args) | {"methods": METHODS, "optimizer_module": str(ROOT / "HSSO-Core/src/swarm_core.py"),
                                  "local_polish": False, "station_statics_new30": True,
                                  "station_statics_stratified32": False}
    (out / "run_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
