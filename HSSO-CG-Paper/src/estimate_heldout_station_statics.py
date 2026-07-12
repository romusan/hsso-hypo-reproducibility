#!/usr/bin/env python
"""Estimate robust station terms without using any of the 30 evaluation events."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(HYPODIR / "src"))
import hsso_hypo_relocation as hypo

TRAIN = ROOT / "tomografia_localizacion_simultaneas/output_sgc_new_events_joint_vp_relocation_fmm_min4_plus_shallow_1d_current/data/sgc_new_events_vp_picks_selected_min4_non2700_nest300_for_joint_inversion.csv"
EVAL = HYPODIR / "data/sgc_new_30/sgc_new_events_30_catalog_for_hsso.csv"
OUT = HYPODIR / "data/sgc_new_30/heldout_station_statics.csv"


def main():
    picks = pd.read_csv(TRAIN); eval_ids = set(pd.read_csv(EVAL).event_id.astype(str))
    overlap = picks.event_id.astype(str).isin(eval_ids)
    excluded_rows = int(overlap.sum()); excluded_events = int(picks.loc[overlap, "event_id"].nunique())
    train = picks.loc[~overlap].copy()
    model = hypo.load_velocity_model(); cache = hypo.FsmFieldCache(model, 8)
    for f in (HYPODIR / "data/fsm_field_cache_sweeps8").glob("field_k*_j*_i*.npy"):
        key = tuple(map(int, f.stem.replace("field_k", "").replace("_j", " ").replace("_i", " ").split())); cache.fields[key] = np.load(f)
    residual_rows=[]
    for event_id, g in train.groupby("event_id"):
        g=g.drop_duplicates("station"); receivers=np.column_stack([g.receiver_x_km,g.receiver_y_km,np.zeros(len(g))])
        operator=hypo.FsmTravelTimeOperator(model,receivers,cache)
        src=g[["source_x_km","source_y_km","source_z_km"]].iloc[0].to_numpy(float)
        obs=g.travel_time_s.to_numpy(float); pred=operator.predict(src); shift=float(np.median(obs-pred)); residual=obs-pred-shift
        residual_rows.extend({"event_id":event_id,"station":sta,"residual_s":r} for sta,r in zip(g.station.astype(str),residual))
    residuals=pd.DataFrame(residual_rows).replace([np.inf,-np.inf],np.nan).dropna(subset=["residual_s"])
    grouped=residuals.groupby("station")
    statics=grouped.residual_s.agg(static_raw_s="median",training_picks="count").reset_index()
    statics["mad_s"] = statics.station.map(grouped.residual_s.apply(
        lambda x: float(np.median(np.abs(x.to_numpy()-np.median(x.to_numpy()))))))
    statics["training_events"] = statics.station.map(grouped.event_id.nunique())
    statics["static_supported"] = statics.training_events >= 20
    centre = float(np.median(statics.loc[statics.static_supported,"static_raw_s"]))
    statics["station_static_s"] = np.where(statics.static_supported,statics.static_raw_s-centre,0.0)
    statics["training_total_events"] = int(train.event_id.nunique())
    statics["evaluation_events_excluded"] = 30
    statics["overlap_events_removed"] = excluded_events
    statics["overlap_pick_rows_removed"] = excluded_rows
    statics["minimum_training_events"] = 20
    statics.to_csv(OUT,index=False)
    residuals.to_csv(HYPODIR / "data/sgc_new_30/heldout_station_static_training_residuals.csv",index=False)
    print(statics.to_string(index=False)); print(f"excluded overlap: {excluded_events} events, {excluded_rows} rows")


if __name__ == "__main__": main()
