#!/usr/bin/env python
"""Evaluate HSSO-E sensitivity to the Huber transition on all 62 events."""
from __future__ import annotations

import importlib.util
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Core/src"))
from swarm_core import SwarmConfig, optimize

spec = importlib.util.spec_from_file_location("comparison", HYPODIR / "src/run_shared_optimizer_comparison.py")
comparison = importlib.util.module_from_spec(spec); sys.modules[spec.name] = comparison; spec.loader.exec_module(comparison)
hypo = comparison.hypo


def score_best(best, operator, obs, statics, delta):
    pred = np.array([hypo._interp_grid_field(best[None, :], f, operator.model)[0] for f in operator.fields]) + statics
    shift = float(np.median(obs - pred)); residual = obs - pred - shift
    rms = float(np.sqrt(np.mean(residual**2))); a = np.abs(residual)
    objective = float(np.sqrt(np.mean(np.where(a <= delta, residual**2, 2*delta*a-delta**2))))
    return objective, rms, shift


def arguments():
    p=argparse.ArgumentParser(); p.add_argument("--full-results",default=str(HYPODIR/"results/shared_core_fsm/relocation_comparison.csv"))
    p.add_argument("--heldout-statics-csv",default=""); p.add_argument("--output-prefix",default="huber_sensitivity")
    return p.parse_args()


def main():
    args=arguments(); base = pd.read_csv(args.full_results)
    base = base[base.method == "HSSO-E"].copy(); base["huber_delta_s"] = 1.0
    base = base.rename(columns={"huber_objective_s":"objective_s"})
    rows = [base]
    blocks = [comparison.load_dataset("stratified32"), comparison.load_dataset("new30")]
    catalog = pd.concat([x[0] for x in blocks], ignore_index=True); picks = pd.concat([x[1] for x in blocks], ignore_index=True)
    if args.heldout_statics_csv:
        station_terms=pd.read_csv(args.heldout_statics_csv).set_index("station").station_static_s
        mask=picks.dataset=="new30"; picks.loc[mask,"station_static_s"]=picks.loc[mask,"station"].map(station_terms).fillna(0.0)
    model = hypo.load_velocity_model(); cache = hypo.FsmFieldCache(model, 8)
    for f in (HYPODIR / "data/fsm_field_cache_sweeps8").glob("field_k*_j*_i*.npy"):
        key = tuple(map(int, f.stem.replace("field_k", "").replace("_j", " ").replace("_i", " ").split())); cache.fields[key] = np.load(f)
    cfg = SwarmConfig(particles=30, iterations=300)
    alt = []
    for delta in [0.5, 1.5]:
        for _, event in catalog.iterrows():
            ep = picks[(picks.dataset == event.dataset) & (picks.event_id == event.event_id)].drop_duplicates("station")
            receivers = np.column_stack([ep.receiver_x_km, ep.receiver_y_km, np.zeros(len(ep))])
            obs = ep.travel_time_s.to_numpy(float); statics = ep.station_static_s.to_numpy(float)
            center = np.array([event.source_x_km, event.source_y_km, event.source_z_km]); operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
            rxy, _ = hypo.event_search_radii(event, receivers)
            bounds = np.array([[max(model.xmin,center[0]-2.5*rxy),min(model.xmax,center[0]+2.5*rxy)],
                               [max(model.ymin,center[1]-2.5*rxy),min(model.ymax,center[1]+2.5*rxy)],
                               [max(model.zmin+0.5,center[2]-20),min(model.zmax,center[2]+20)]])
            objective = lambda x, d=delta: comparison.evaluate_matrix(x, operator, obs, statics, d)
            seed = int((20260710 + zlib.crc32(str(event.event_id).encode())) % (2**32-1))
            run = optimize("HSSO-E", objective, bounds, seed, cfg, record_history=False)
            obj, rms, shift = score_best(run["best"], operator, obs, statics, delta)
            dh = float(np.linalg.norm(run["best"][:2]-center[:2])); dz = float(run["best"][2]-center[2])
            tol = max(.25,.01*(bounds[2,1]-bounds[2,0]))
            alt.append({"dataset":event.dataset,"event_id":event.event_id,"method":"HSSO-E","huber_delta_s":delta,
                        "objective_s":obj,"relocated_rms_s":rms,"origin_shift_s":shift,
                        "relocated_x_km":run["best"][0],"relocated_y_km":run["best"][1],"relocated_depth_km":run["best"][2],
                        "delta_horizontal_km":dh,"delta_depth_km":dz,
                        "depth_boundary":bool(run["best"][2]<=bounds[2,0]+tol or run["best"][2]>=bounds[2,1]-tol),
                        "elapsed_s":run["elapsed_s"]})
            print(event.event_id, delta, f"RMS={rms:.4f}", flush=True)
    alt = pd.DataFrame(alt)
    keep = ["dataset","event_id","method","huber_delta_s","objective_s","relocated_rms_s","origin_shift_s",
            "relocated_depth_km","delta_horizontal_km","delta_depth_km","depth_boundary"]
    combined = pd.concat([base[[c for c in keep if c in base.columns]], alt[keep]], ignore_index=True, sort=False)
    out = ROOT / "HSSO-CG-Paper/results"; combined.to_csv(out / f"{args.output_prefix}_runs.csv", index=False)
    pivot = alt.pivot(index=["dataset","event_id"], columns="huber_delta_s", values=["relocated_rms_s","relocated_depth_km","delta_horizontal_km"])
    ref = base.set_index(["dataset","event_id"])
    audit=[]
    for key,row in pivot.iterrows():
        for delta in [0.5,1.5]:
            b=ref.loc[key]
            audit.append({"dataset":key[0],"event_id":key[1],"huber_delta_s":delta,
                          "rms_change_s":row[("relocated_rms_s",delta)]-b.relocated_rms_s,
                          "depth_change_km":row[("relocated_depth_km",delta)]-b.relocated_depth_km,
                          "horizontal_shift_change_km":row[("delta_horizontal_km",delta)]-b.delta_horizontal_km})
    audit=pd.DataFrame(audit); audit.to_csv(out / f"{args.output_prefix}_audit.csv",index=False)
    summary=audit.groupby(["dataset","huber_delta_s"]).agg(events=("event_id","size"),median_abs_rms_change_s=("rms_change_s",lambda x:np.median(np.abs(x))),
        median_abs_depth_change_km=("depth_change_km",lambda x:np.median(np.abs(x))),max_abs_depth_change_km=("depth_change_km",lambda x:np.max(np.abs(x))),
        median_abs_horizontal_change_km=("horizontal_shift_change_km",lambda x:np.median(np.abs(x)))).reset_index()
    summary.to_csv(out / f"{args.output_prefix}_summary.csv",index=False); print(summary.to_string(index=False))


if __name__ == "__main__": main()
