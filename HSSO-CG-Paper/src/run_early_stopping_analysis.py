#!/usr/bin/env python
"""Measure time to a tolerance around each method's full-budget terminal objective."""
from __future__ import annotations

import importlib.util
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Core/src"))
from swarm_core import SwarmConfig, optimize

spec = importlib.util.spec_from_file_location("comparison", HYPODIR / "src/run_shared_optimizer_comparison.py")
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)
hypo = comparison.hypo


def arguments():
    p=argparse.ArgumentParser(); p.add_argument("--full-results",default=str(HYPODIR/"results/shared_core_fsm/relocation_comparison.csv"))
    p.add_argument("--heldout-statics-csv",default=""); p.add_argument("--output-prefix",default="early_stopping")
    return p.parse_args()


def main():
    args=arguments(); full = pd.read_csv(args.full_results)
    blocks = [comparison.load_dataset("stratified32"), comparison.load_dataset("new30")]
    catalog = pd.concat([x[0] for x in blocks], ignore_index=True)
    picks = pd.concat([x[1] for x in blocks], ignore_index=True)
    if args.heldout_statics_csv:
        statics=pd.read_csv(args.heldout_statics_csv).set_index("station").station_static_s
        mask=picks.dataset=="new30"; picks.loc[mask,"station_static_s"]=picks.loc[mask,"station"].map(statics).fillna(0.0)
    model = hypo.load_velocity_model(); cache = hypo.FsmFieldCache(model, 8)
    cache_dir = HYPODIR / "data/fsm_field_cache_sweeps8"
    for f in cache_dir.glob("field_k*_j*_i*.npy"):
        key = tuple(map(int, f.stem.replace("field_k", "").replace("_j", " ").replace("_i", " ").split()))
        cache.fields[key] = np.load(f)
    cfg = SwarmConfig(particles=30, iterations=300)
    rows = []
    for _, event in catalog.iterrows():
        ep = picks[(picks.dataset == event.dataset) & (picks.event_id == event.event_id)].drop_duplicates("station")
        receivers = np.column_stack([ep.receiver_x_km, ep.receiver_y_km, np.zeros(len(ep))])
        obs = ep.travel_time_s.to_numpy(float); statics = ep.station_static_s.to_numpy(float)
        center = np.array([event.source_x_km, event.source_y_km, event.source_z_km])
        operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
        rxy, _ = hypo.event_search_radii(event, receivers)
        bounds = np.array([
            [max(model.xmin, center[0]-2.5*rxy), min(model.xmax, center[0]+2.5*rxy)],
            [max(model.ymin, center[1]-2.5*rxy), min(model.ymax, center[1]+2.5*rxy)],
            [max(model.zmin+0.5, center[2]-20), min(model.zmax, center[2]+20)]])
        objective = lambda x: comparison.evaluate_matrix(x, operator, obs, statics)
        seed = int((20260710 + zlib.crc32(str(event.event_id).encode())) % (2**32-1))
        for method in comparison.METHODS:
            terminal = full[(full.dataset == event.dataset) & (full.event_id == event.event_id) &
                            (full.method == method)].iloc[0]
            target = float(terminal.huber_objective_s) + 1.0e-4
            run = optimize(method, objective, bounds, seed, cfg, record_history=False, target_fitness=target)
            rows.append({"dataset": event.dataset, "event_id": event.event_id, "method": method,
                         "terminal_objective_s": terminal.huber_objective_s, "target_objective_s": target,
                         "reached_objective_s": run["fitness"], "iterations": run["iterations_completed"],
                         "evaluations": run["evaluations"], "elapsed_s": run["elapsed_s"],
                         "full_elapsed_s": terminal.optimizer_elapsed_s,
                         "time_fraction": run["elapsed_s"] / terminal.optimizer_elapsed_s})
    out = ROOT / "HSSO-CG-Paper/results"; out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows); raw.to_csv(out / f"{args.output_prefix}_runs.csv", index=False)
    summary = raw.groupby(["dataset", "method"]).agg(
        events=("event_id", "size"), median_iterations=("iterations", "median"),
        median_evaluations=("evaluations", "median"), median_time_s=("elapsed_s", "median"),
        median_full_time_s=("full_elapsed_s", "median"), median_time_fraction=("time_fraction", "median")
    ).reset_index()
    summary.to_csv(out / f"{args.output_prefix}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
