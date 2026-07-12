#!/usr/bin/env python
"""Repeated-seed audit for a balanced subset of QC-flagged HSSO-E events."""
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
PAPER = ROOT / "HSSO-CG-Paper"
HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Core/src"))
from swarm_core import SwarmConfig, optimize

spec = importlib.util.spec_from_file_location(
    "comparison", HYPODIR / "src/run_shared_optimizer_comparison.py"
)
comparison = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = comparison
spec.loader.exec_module(comparison)
hypo = comparison.hypo


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-dataset", type=int, default=6)
    parser.add_argument("--additional-seeds", type=int, default=9)
    parser.add_argument("--base-seed", type=int, default=20260710)
    parser.add_argument(
        "--heldout-statics-csv",
        default=str(HYPODIR / "data/sgc_new_30/heldout_station_statics.csv"),
    )
    parser.add_argument("--output-prefix", default="seed_sensitivity_flagged12")
    return parser.parse_args()


def select_events(flags: pd.DataFrame, n: int) -> pd.DataFrame:
    frame = flags.copy()
    frame["qc_excursion"] = np.maximum.reduce(
        [
            frame.delta_horizontal_km.to_numpy(float) / 25.0,
            frame.delta_depth_km.abs().to_numpy(float) / 15.0,
            frame.depth_boundary.astype(float).to_numpy(),
        ]
    )
    selected = []
    for dataset, block in frame.groupby("dataset", sort=True):
        chosen = block.sort_values(
            ["qc_excursion", "relocated_rms_s", "event_id"],
            ascending=[False, False, True],
        ).head(n)
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def load_problem_data(statics_csv: str):
    blocks = [comparison.load_dataset("stratified32"), comparison.load_dataset("new30")]
    catalog = pd.concat([block[0] for block in blocks], ignore_index=True)
    picks = pd.concat([block[1] for block in blocks], ignore_index=True)
    terms = pd.read_csv(statics_csv).set_index("station").station_static_s
    mask = picks.dataset == "new30"
    picks.loc[mask, "station_static_s"] = picks.loc[mask, "station"].map(terms).fillna(0.0)
    return catalog, picks


def load_cache(picks: pd.DataFrame):
    model = hypo.load_velocity_model()
    cache = hypo.FsmFieldCache(model, 8)
    cache_dir = HYPODIR / "data/fsm_field_cache_sweeps8"
    for path in cache_dir.glob("field_k*_j*_i*.npy"):
        key = tuple(
            map(
                int,
                path.stem.replace("field_k", "")
                .replace("_j", " ")
                .replace("_i", " ")
                .split(),
            )
        )
        cache.fields[key] = np.load(path)
    for rec in picks[["receiver_x_km", "receiver_y_km"]].drop_duplicates().itertuples(index=False):
        xyz = np.array([rec.receiver_x_km, rec.receiver_y_km, 0.0])
        key = hypo.coord_to_index_xyz(*xyz, model)
        if key not in cache.fields:
            cache.fields[key] = cache.field_for_receiver(xyz)
    return model, cache


def solve(event, event_picks, model, cache, seed: int, config: SwarmConfig) -> dict[str, object]:
    event_picks = event_picks.drop_duplicates("station").copy()
    obs = event_picks.travel_time_s.to_numpy(float)
    statics = event_picks.station_static_s.to_numpy(float)
    receivers = np.column_stack(
        [event_picks.receiver_x_km, event_picks.receiver_y_km, np.zeros(len(event_picks))]
    )
    center = np.array([event.source_x_km, event.source_y_km, event.source_z_km], float)
    operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
    rxy, _ = hypo.event_search_radii(event, receivers)
    bounds = np.array(
        [
            [max(model.xmin, center[0] - 2.5 * rxy), min(model.xmax, center[0] + 2.5 * rxy)],
            [max(model.ymin, center[1] - 2.5 * rxy), min(model.ymax, center[1] + 2.5 * rxy)],
            [max(model.zmin + 0.5, center[2] - 20.0), min(model.zmax, center[2] + 20.0)],
        ]
    )
    objective = lambda positions: comparison.evaluate_matrix(positions, operator, obs, statics, 1.0)
    run = optimize("HSSO-E", objective, bounds, seed, config, record_history=False)
    best = run["best"]
    _, rms, origin_shift, _, _ = hypo.residual_solution(
        best,
        obs,
        receivers,
        model,
        2.0,
        operator=operator,
        station_static_s=statics,
    )
    tolerance = max(0.25, 0.01 * (bounds[2, 1] - bounds[2, 0]))
    return {
        "dataset": event.dataset,
        "event_id": event.event_id,
        "seed": seed,
        "relocated_rms_s": rms,
        "delta_horizontal_km": float(np.linalg.norm(best[:2] - center[:2])),
        "delta_depth_km": float(best[2] - center[2]),
        "relocated_x_km": float(best[0]),
        "relocated_y_km": float(best[1]),
        "relocated_depth_km": float(best[2]),
        "origin_shift_s": origin_shift,
        "depth_boundary": bool(
            best[2] <= bounds[2, 0] + tolerance or best[2] >= bounds[2, 1] - tolerance
        ),
        "evaluations": run["evaluations"],
        "elapsed_s": run["elapsed_s"],
    }


def value_range(series: pd.Series) -> float:
    return float(series.max() - series.min())


def main() -> None:
    args = arguments()
    flags = pd.read_csv(PAPER / "results/supplementary_hsso_e_flagged_events.csv")
    selected = select_events(flags, args.events_per_dataset)
    catalog, picks = load_problem_data(args.heldout_statics_csv)
    selected_catalog = catalog.merge(
        selected[["dataset", "event_id", "qc_excursion"]],
        on=["dataset", "event_id"],
        how="inner",
        validate="one_to_one",
    )
    model, cache = load_cache(picks[picks.event_id.isin(selected.event_id)])
    config = SwarmConfig(particles=30, iterations=300)
    rows = []
    for _, event in selected_catalog.iterrows():
        event_picks = picks[(picks.dataset == event.dataset) & (picks.event_id == event.event_id)]
        event_base = int((args.base_seed + zlib.crc32(str(event.event_id).encode())) % (2**32 - 1))
        for offset in range(args.additional_seeds + 1):
            seed = int((event_base + offset * 1_000_003) % (2**32 - 1))
            row = solve(event, event_picks, model, cache, seed, config)
            row["seed_offset"] = offset
            row["reported_seed"] = offset == 0
            row["qc_excursion"] = event.qc_excursion
            rows.append(row)
            print(
                f"{event.dataset} {event.event_id} seed {offset}: "
                f"RMS={row['relocated_rms_s']:.4f} dH={row['delta_horizontal_km']:.2f} "
                f"dZ={row['delta_depth_km']:.2f}",
                flush=True,
            )
    runs = pd.DataFrame(rows)
    out = PAPER / "results"
    prefix = args.output_prefix
    runs.to_csv(out / f"{prefix}_runs.csv", index=False)
    per_event = (
        runs.groupby(["dataset", "event_id"], as_index=False)
        .agg(
            seeds=("seed", "size"),
            rms_median_s=("relocated_rms_s", "median"),
            rms_sd_s=("relocated_rms_s", "std"),
            rms_range_s=("relocated_rms_s", value_range),
            horizontal_median_km=("delta_horizontal_km", "median"),
            horizontal_sd_km=("delta_horizontal_km", "std"),
            horizontal_range_km=("delta_horizontal_km", value_range),
            depth_shift_median_km=("delta_depth_km", "median"),
            depth_shift_sd_km=("delta_depth_km", "std"),
            depth_shift_range_km=("delta_depth_km", value_range),
            boundary_frequency=("depth_boundary", "mean"),
        )
    )
    per_event.to_csv(out / f"{prefix}_event_summary.csv", index=False)
    pooled = pd.DataFrame(
        [
            {
                "events": len(per_event),
                "seeds_per_event": args.additional_seeds + 1,
                "additional_seeds": args.additional_seeds,
                "median_rms_sd_s": per_event.rms_sd_s.median(),
                "max_rms_sd_s": per_event.rms_sd_s.max(),
                "median_rms_range_s": per_event.rms_range_s.median(),
                "max_rms_range_s": per_event.rms_range_s.max(),
                "median_horizontal_sd_km": per_event.horizontal_sd_km.median(),
                "max_horizontal_sd_km": per_event.horizontal_sd_km.max(),
                "median_horizontal_range_km": per_event.horizontal_range_km.median(),
                "max_horizontal_range_km": per_event.horizontal_range_km.max(),
                "median_depth_shift_sd_km": per_event.depth_shift_sd_km.median(),
                "max_depth_shift_sd_km": per_event.depth_shift_sd_km.max(),
                "median_depth_shift_range_km": per_event.depth_shift_range_km.median(),
                "max_depth_shift_range_km": per_event.depth_shift_range_km.max(),
                "events_with_mixed_boundary_status": int(
                    ((per_event.boundary_frequency > 0) & (per_event.boundary_frequency < 1)).sum()
                ),
            }
        ]
    )
    pooled.to_csv(out / f"{prefix}_pooled_summary.csv", index=False)
    (out / f"{prefix}_config.json").write_text(
        json.dumps(
            vars(args)
            | {
                "selection": "six largest normalized QC excursions per data set among the 35 pre-flagged HSSO-E events",
                "selected_events": selected[["dataset", "event_id"]].to_dict("records"),
                "reported_seed_offset": 0,
                "seed_stride": 1_000_003,
                "optimizer": "HSSO-E",
                "particles": 30,
                "iterations": 300,
                "evaluations": 9030,
                "huber_delta_s": 1.0,
                "depth_window_km": 20.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nPooled summary\n" + pooled.to_string(index=False))


if __name__ == "__main__":
    main()
