#!/usr/bin/env python
"""Pilot HSSO-Hypo relocation for real SGC earthquakes.

The first implementation keeps the optimizer and data plumbing reproducible:
it selects SGC events with catalog RMS < 1 s, loads P picks already extracted
for the Vp/Q workflow, uses the final 3-D Vp model, and relocates hypocenters
with a helical/topological swarm. Travel times can be computed by straight-ray
integration or by a 3-D Fast Sweeping Method (FSM) grid-eikonal operator.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False
    plt = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
TOMO_DIR = WORKSPACE_DIR / "tomografia_Q"

PICKS_CSV = TOMO_DIR / "output_V2" / "sgc_final_q_extra_station_tomography" / "data" / "sgc_1000_vp_picks.csv"
CATALOG_CSV = TOMO_DIR / "output_V2" / "data" / "sgc_combined_area_2022_2026_depth20_170_rms_lt1_for_final.csv"
XML_PICKS_CSV = TOMO_DIR / "output_V2" / "data" / "vp_travel_times_xml_picks_v2_inside_area.csv"
XML_EVENTS_CSV = TOMO_DIR / "output_V2" / "data" / "catalog_events_from_xml_v2.csv"
GRID_NPZ = (
    TOMO_DIR
    / "output_V2"
    / "sgc_final_q_extra_station_fsm_tomography"
    / "models"
    / "sgc_final_fsm_grid.npz"
)
VP_NPY = (
    TOMO_DIR
    / "output_V2"
    / "sgc_final_q_extra_station_fsm_tomography"
    / "models"
    / "sgc_final_fsm_vp_final.npy"
)

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PROJECT_DIR / "figures"

LAT0 = 7.0
LON0 = -73.0


@dataclass(frozen=True)
class VelocityModel:
    xg: np.ndarray
    yg: np.ndarray
    zg: np.ndarray
    vp: np.ndarray  # shape: z, y, x

    @property
    def xmin(self) -> float:
        return float(self.xg[0])

    @property
    def xmax(self) -> float:
        return float(self.xg[-1])

    @property
    def ymin(self) -> float:
        return float(self.yg[0])

    @property
    def ymax(self) -> float:
        return float(self.yg[-1])

    @property
    def zmin(self) -> float:
        return float(self.zg[0])

    @property
    def zmax(self) -> float:
        return float(self.zg[-1])


def ensure_dirs() -> None:
    for folder in (DATA_DIR, RESULTS_DIR, FIG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSSO-Hypo on selected SGC events.")
    parser.add_argument("--picks-csv", default=str(PICKS_CSV))
    parser.add_argument("--catalog-csv", default=str(CATALOG_CSV))
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--rms-max", type=float, default=1.0)
    parser.add_argument("--min-stations", type=int, default=4)
    parser.add_argument("--selection-mode", choices=["top", "depth-stratified"], default="top")
    parser.add_argument("--depth-min-km", type=float, default=0.0)
    parser.add_argument("--depth-max-km", type=float, default=160.0)
    parser.add_argument("--depth-bin-km", type=float, default=10.0)
    parser.add_argument("--events-per-bin", type=int, default=2)
    parser.add_argument("--include-xml-picks", action="store_true")
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--ray-step-km", type=float, default=2.0)
    parser.add_argument("--travel-time-mode", choices=["straight", "fsm"], default="straight")
    parser.add_argument("--fsm-sweeps", type=int, default=8)
    parser.add_argument("--depth-window-km", type=float, default=0.0)
    parser.add_argument("--station-statics-csv", default="")
    parser.add_argument("--apply-station-statics", action="store_true")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--no-polish", action="store_true")
    return parser.parse_args()


def latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
    x = (float(lon) - LON0) * 111.32 * math.cos(math.radians(LAT0))
    y = (float(lat) - LAT0) * 111.32
    return float(x), float(y)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = float(y) / 111.32 + LAT0
    lon = float(x) / (111.32 * math.cos(math.radians(LAT0))) + LON0
    return lat, lon


def coord_to_index_1d(value: float, grid: np.ndarray) -> int:
    if len(grid) < 2:
        return 0
    idx = int(round((float(value) - float(grid[0])) / float(grid[1] - grid[0])))
    return int(np.clip(idx, 0, len(grid) - 1))


def coord_to_index_xyz(x: float, y: float, z: float, model: VelocityModel) -> tuple[int, int, int]:
    i = coord_to_index_1d(x, model.xg)
    j = coord_to_index_1d(y, model.yg)
    k = coord_to_index_1d(z, model.zg)
    return k, j, i


def event_code(value: object) -> str:
    text = str(value or "").strip()
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return text


def load_velocity_model() -> VelocityModel:
    grid = np.load(GRID_NPZ)
    vp = np.load(VP_NPY).astype(float)
    model = VelocityModel(
        xg=grid["xg"].astype(float),
        yg=grid["yg"].astype(float),
        zg=grid["zg"].astype(float),
        vp=vp,
    )
    expected = (len(model.zg), len(model.yg), len(model.xg))
    if model.vp.shape != expected:
        raise ValueError(f"Vp shape {model.vp.shape} does not match grid {expected}")
    return model


def read_inputs(picks_csv: Path = PICKS_CSV, catalog_csv: Path = CATALOG_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = pd.read_csv(picks_csv)
    catalog = pd.read_csv(catalog_csv)
    for col in [
        "rms_s",
        "phases",
        "gap_deg",
        "error_lat_km",
        "error_lon_km",
        "error_depth_km",
        "depth_km",
        "magnitude",
        "lat",
        "lon",
    ]:
        if col in catalog.columns:
            catalog[col] = pd.to_numeric(catalog[col], errors="coerce")
    for col in [
        "event_lat",
        "event_lon",
        "event_depth_km",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "station_lat",
        "station_lon",
        "receiver_x_km",
        "receiver_y_km",
        "travel_time_s",
    ]:
        if col in picks.columns:
            picks[col] = pd.to_numeric(picks[col], errors="coerce")
    picks["event_id"] = picks["event_id"].map(event_code)
    if "selection_source" not in picks.columns:
        picks["selection_source"] = "final_sgc_catalog"
    if "rms_source" not in picks.columns:
        picks["rms_source"] = "catalog_rms_s"
    catalog["event_id"] = catalog["event_id"].map(event_code)
    if "selection_source" not in catalog.columns:
        catalog["selection_source"] = "final_sgc_catalog"
    if "rms_source" not in catalog.columns:
        catalog["rms_source"] = "catalog_rms_s"
    return picks, catalog


def configure_station_statics(picks: pd.DataFrame, statics_csv: str, apply_statics: bool) -> pd.DataFrame:
    out = picks.copy()
    if not apply_statics:
        out["station_static_s"] = 0.0
        return out
    if statics_csv:
        statics = pd.read_csv(statics_csv)
        if not {"station", "static_s"}.issubset(statics.columns):
            raise ValueError(f"Station statics CSV must contain station/static_s columns: {statics_csv}")
        static_map = statics.drop_duplicates("station").set_index("station")["static_s"].astype(float).to_dict()
        out["station_static_s"] = out["station"].astype(str).map(static_map).fillna(0.0).astype(float)
        return out
    if "station_static_s" in out.columns:
        out["station_static_s"] = pd.to_numeric(out["station_static_s"], errors="coerce").fillna(0.0)
        return out
    out["station_static_s"] = 0.0
    return out


def read_xml_proxy_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read XML-derived picks and build a per-event catalog proxy.

    The XML table lacks the SGC catalog RMS column used by the final catalog,
    but includes per-pick arrival residuals.  For depth bins not covered by the
    final depth>20 km catalog, the event RMS is computed from those residuals.
    """
    picks = pd.read_csv(XML_PICKS_CSV)
    events = pd.read_csv(XML_EVENTS_CSV) if XML_EVENTS_CSV.exists() else pd.DataFrame()
    for col in [
        "event_lat",
        "event_lon",
        "event_depth_km",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "station_lat",
        "station_lon",
        "receiver_x_km",
        "receiver_y_km",
        "travel_time_s",
        "arrival_residual_s",
    ]:
        if col in picks.columns:
            picks[col] = pd.to_numeric(picks[col], errors="coerce")
    picks["event_id"] = picks["event_id"].map(event_code)
    picks["selection_source"] = "xml_arrival_residual_proxy"
    if not events.empty:
        events["event_id"] = events["event_id"].map(event_code)
        events = events.drop_duplicates("event_id")
        event_names = events.set_index("event_id")["event_name"].to_dict() if "event_name" in events else {}
    else:
        event_names = {}

    grouped = (
        picks.groupby("event_id")
        .agg(
            time_utc=("origin_time_utc", "first"),
            origin_time_utc=("origin_time_utc", "first"),
            event_lat=("event_lat", "first"),
            event_lon=("event_lon", "first"),
            event_depth_km=("event_depth_km", "first"),
            depth_km=("event_depth_km", "first"),
            lat=("event_lat", "first"),
            lon=("event_lon", "first"),
            source_x_km=("source_x_km", "first"),
            source_y_km=("source_y_km", "first"),
            source_z_km=("source_z_km", "first"),
            phases=("station", "size"),
            rms_s=("arrival_residual_s", lambda s: float(np.sqrt(np.nanmean(np.asarray(s, dtype=float) ** 2)))),
        )
        .reset_index()
    )
    grouped["magnitude"] = np.nan
    grouped["magnitude_type"] = ""
    grouped["gap_deg"] = np.nan
    grouped["error_lat_km"] = np.nan
    grouped["error_lon_km"] = np.nan
    grouped["error_depth_km"] = np.nan
    grouped["region"] = grouped["event_id"].map(event_names).fillna("SGC XML event")
    grouped["status"] = "xml_rms_proxy"
    grouped["source_depth_class"] = "xml_v2"
    grouped["quakeml_url"] = ""
    grouped["phases_url"] = ""
    grouped["map_url"] = ""
    grouped["selection_source"] = "xml_arrival_residual_proxy"
    grouped["rms_source"] = "xml_arrival_residual_rms_s"
    return picks, grouped


def combine_sources(
    picks: pd.DataFrame,
    catalog: pd.DataFrame,
    include_xml_picks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not include_xml_picks:
        return picks, catalog
    xml_picks, xml_catalog = read_xml_proxy_inputs()
    final_ids = set(catalog["event_id"].astype(str))
    xml_picks = xml_picks[~xml_picks["event_id"].astype(str).isin(final_ids)].copy()
    xml_catalog = xml_catalog[~xml_catalog["event_id"].astype(str).isin(final_ids)].copy()
    all_picks = pd.concat([picks, xml_picks], ignore_index=True, sort=False)
    all_catalog = pd.concat([catalog, xml_catalog], ignore_index=True, sort=False)
    return all_picks, all_catalog


def filter_picks_for_model(picks: pd.DataFrame, model: VelocityModel) -> pd.DataFrame:
    """Keep picks whose source and receiver endpoints are inside the 3-D grid."""
    required = [
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "receiver_x_km",
        "receiver_y_km",
        "travel_time_s",
    ]
    clean = picks.dropna(subset=required).copy()
    inside = (
        clean["source_x_km"].between(model.xmin, model.xmax)
        & clean["source_y_km"].between(model.ymin, model.ymax)
        & clean["source_z_km"].between(model.zmin, model.zmax)
        & clean["receiver_x_km"].between(model.xmin, model.xmax)
        & clean["receiver_y_km"].between(model.ymin, model.ymax)
    )
    return clean[inside].reset_index(drop=True)


def select_events(
    picks: pd.DataFrame,
    catalog: pd.DataFrame,
    model: VelocityModel,
    n_events: int,
    rms_max: float,
    min_stations: int,
    selection_mode: str = "top",
    depth_min_km: float = 0.0,
    depth_max_km: float = 160.0,
    depth_bin_km: float = 10.0,
    events_per_bin: int = 2,
) -> pd.DataFrame:
    catalog_one = catalog.drop_duplicates("event_id").copy()
    grouped = (
        picks.dropna(subset=["event_id", "station", "travel_time_s"])
        .groupby("event_id")
        .agg(
            origin_time_utc=("origin_time_utc", "first"),
            event_lat=("event_lat", "first"),
            event_lon=("event_lon", "first"),
            event_depth_km=("event_depth_km", "first"),
            source_x_km=("source_x_km", "first"),
            source_y_km=("source_y_km", "first"),
            source_z_km=("source_z_km", "first"),
            n_picks=("station", "size"),
            n_stations=("station", "nunique"),
            stations=("station", lambda s: " ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    events = grouped.merge(catalog_one, on="event_id", how="left", suffixes=("", "_catalog"))
    inside = (
        events["source_x_km"].between(model.xmin, model.xmax)
        & events["source_y_km"].between(model.ymin, model.ymax)
        & events["source_z_km"].between(model.zmin, model.zmax)
    )
    candidates = events[
        inside
        & (events["rms_s"] < rms_max)
        & (events["n_stations"] >= min_stations)
        & events["event_lat"].notna()
        & events["event_lon"].notna()
        & events["event_depth_km"].notna()
    ].copy()
    candidates = candidates.sort_values(
        by=["n_stations", "phases", "rms_s", "gap_deg", "time_utc"],
        ascending=[False, False, True, True, True],
    )
    if selection_mode == "depth-stratified":
        selected_parts = []
        bin_rows = []
        edges = np.arange(depth_min_km, depth_max_km + 0.5 * depth_bin_km, depth_bin_km, dtype=float)
        for lo, hi in zip(edges[:-1], edges[1:]):
            bin_df = candidates[(candidates["event_depth_km"] >= lo) & (candidates["event_depth_km"] < hi)].copy()
            take = bin_df.head(events_per_bin)
            if len(take):
                selected_parts.append(take)
            bin_rows.append(
                {
                    "depth_min_km": lo,
                    "depth_max_km": hi,
                    "candidate_events": int(len(bin_df)),
                    "selected_events": int(len(take)),
                    "target_events": int(events_per_bin),
                    "complete": bool(len(take) >= events_per_bin),
                }
            )
        bin_report = pd.DataFrame(bin_rows)
        bin_report.to_csv(DATA_DIR / "depth_stratified_selection_bins.csv", index=False)
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
        expected = int((len(edges) - 1) * events_per_bin)
        if len(selected) < expected:
            missing = bin_report[~bin_report["complete"]]
            missing_text = ", ".join(
                f"{r.depth_min_km:.0f}-{r.depth_max_km:.0f} km ({r.selected_events}/{r.target_events})"
                for r in missing.itertuples(index=False)
            )
            raise RuntimeError(
                f"Only {len(selected)} stratified events selected; expected {expected}. "
                f"Incomplete bins: {missing_text}"
            )
    else:
        selected = candidates.head(n_events)
        if len(selected) < n_events:
            raise RuntimeError(f"Only {len(selected)} events satisfy the filters; requested {n_events}.")
    keep_cols = [
        "event_id",
        "time_utc",
        "origin_time_utc",
        "event_lat",
        "event_lon",
        "event_depth_km",
        "source_x_km",
        "source_y_km",
        "source_z_km",
        "magnitude",
        "magnitude_type",
        "phases",
        "rms_s",
        "gap_deg",
        "error_lat_km",
        "error_lon_km",
        "error_depth_km",
        "region",
        "status",
        "source_depth_class",
        "n_picks",
        "n_stations",
        "stations",
        "selection_source",
        "rms_source",
        "quakeml_url",
        "phases_url",
        "map_url",
    ]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    if selection_mode == "depth-stratified":
        return selected[keep_cols].reset_index(drop=True)
    return selected.head(n_events)[keep_cols].reset_index(drop=True)


def _axis_frame(receivers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0])
    if receivers.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), z_axis
    xy = receivers[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(xy.T)
    if not np.all(np.isfinite(cov)):
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), z_axis
    vals, vecs = np.linalg.eigh(cov)
    e1xy = vecs[:, int(np.argmax(vals))]
    e1 = np.array([float(e1xy[0]), float(e1xy[1]), 0.0])
    e1 /= max(np.linalg.norm(e1), 1.0e-12)
    e2 = np.array([-e1[1], e1[0], 0.0])
    return e1, e2, z_axis


def _interp_vp(points: np.ndarray, model: VelocityModel) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    outside = (
        (x < model.xmin)
        | (x > model.xmax)
        | (y < model.ymin)
        | (y > model.ymax)
        | (z < model.zmin)
        | (z > model.zmax)
    )

    ix = np.searchsorted(model.xg, x, side="right") - 1
    iy = np.searchsorted(model.yg, y, side="right") - 1
    iz = np.searchsorted(model.zg, z, side="right") - 1
    ix = np.clip(ix, 0, len(model.xg) - 2)
    iy = np.clip(iy, 0, len(model.yg) - 2)
    iz = np.clip(iz, 0, len(model.zg) - 2)

    tx = (x - model.xg[ix]) / np.maximum(model.xg[ix + 1] - model.xg[ix], 1.0e-12)
    ty = (y - model.yg[iy]) / np.maximum(model.yg[iy + 1] - model.yg[iy], 1.0e-12)
    tz = (z - model.zg[iz]) / np.maximum(model.zg[iz + 1] - model.zg[iz], 1.0e-12)

    vp = model.vp
    c000 = vp[iz, iy, ix]
    c001 = vp[iz, iy, ix + 1]
    c010 = vp[iz, iy + 1, ix]
    c011 = vp[iz, iy + 1, ix + 1]
    c100 = vp[iz + 1, iy, ix]
    c101 = vp[iz + 1, iy, ix + 1]
    c110 = vp[iz + 1, iy + 1, ix]
    c111 = vp[iz + 1, iy + 1, ix + 1]

    c00 = c000 * (1.0 - tx) + c001 * tx
    c01 = c010 * (1.0 - tx) + c011 * tx
    c10 = c100 * (1.0 - tx) + c101 * tx
    c11 = c110 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    out = c0 * (1.0 - tz) + c1 * tz
    out[outside] = np.nan
    return out


def _interp_grid_field(points: np.ndarray, field: np.ndarray, model: VelocityModel) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    outside = (
        (x < model.xmin)
        | (x > model.xmax)
        | (y < model.ymin)
        | (y > model.ymax)
        | (z < model.zmin)
        | (z > model.zmax)
    )

    ix = np.searchsorted(model.xg, x, side="right") - 1
    iy = np.searchsorted(model.yg, y, side="right") - 1
    iz = np.searchsorted(model.zg, z, side="right") - 1
    ix = np.clip(ix, 0, len(model.xg) - 2)
    iy = np.clip(iy, 0, len(model.yg) - 2)
    iz = np.clip(iz, 0, len(model.zg) - 2)

    tx = (x - model.xg[ix]) / np.maximum(model.xg[ix + 1] - model.xg[ix], 1.0e-12)
    ty = (y - model.yg[iy]) / np.maximum(model.yg[iy + 1] - model.yg[iy], 1.0e-12)
    tz = (z - model.zg[iz]) / np.maximum(model.zg[iz + 1] - model.zg[iz], 1.0e-12)

    c000 = field[iz, iy, ix]
    c001 = field[iz, iy, ix + 1]
    c010 = field[iz, iy + 1, ix]
    c011 = field[iz, iy + 1, ix + 1]
    c100 = field[iz + 1, iy, ix]
    c101 = field[iz + 1, iy, ix + 1]
    c110 = field[iz + 1, iy + 1, ix]
    c111 = field[iz + 1, iy + 1, ix + 1]

    c00 = c000 * (1.0 - tx) + c001 * tx
    c01 = c010 * (1.0 - tx) + c011 * tx
    c10 = c100 * (1.0 - tx) + c101 * tx
    c11 = c110 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    out = c0 * (1.0 - tz) + c1 * tz
    out[outside] = np.nan
    return out


def _solve_local_fsm(t1: float, t2: float, t3: float, h1: float, h2: float, h3: float, slow: float) -> float:
    a = 1.0 / (h1 * h1)
    b = -2.0 * t1 / (h1 * h1)
    c = (t1 * t1) / (h1 * h1) - slow * slow
    disc = max(b * b - 4.0 * a * c, 0.0)
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    if t <= t2:
        return t

    a = 1.0 / (h1 * h1) + 1.0 / (h2 * h2)
    b = -2.0 * (t1 / (h1 * h1) + t2 / (h2 * h2))
    c = (t1 * t1) / (h1 * h1) + (t2 * t2) / (h2 * h2) - slow * slow
    disc = max(b * b - 4.0 * a * c, 0.0)
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    if t <= t3:
        return t

    a = 1.0 / (h1 * h1) + 1.0 / (h2 * h2) + 1.0 / (h3 * h3)
    b = -2.0 * (t1 / (h1 * h1) + t2 / (h2 * h2) + t3 / (h3 * h3))
    c = (t1 * t1) / (h1 * h1) + (t2 * t2) / (h2 * h2) + (t3 * t3) / (h3 * h3) - slow * slow
    disc = max(b * b - 4.0 * a * c, 0.0)
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    return max(t, t3)


def fast_sweeping_3d(slowness: np.ndarray, src_k: int, src_j: int, src_i: int, dx: float, dy: float, dz: float, n_sweeps: int, big: float = 1.0e20) -> np.ndarray:
    nz, ny, nx = slowness.shape
    times = np.full((nz, ny, nx), big, dtype=float)
    times[src_k, src_j, src_i] = 0.0
    for _ in range(n_sweeps):
        for kz in (1, -1):
            k_range = range(0, nz) if kz == 1 else range(nz - 1, -1, -1)
            for jy in (1, -1):
                j_range = range(0, ny) if jy == 1 else range(ny - 1, -1, -1)
                for ix in (1, -1):
                    i_range = range(0, nx) if ix == 1 else range(nx - 1, -1, -1)
                    for k in k_range:
                        for j in j_range:
                            for i in i_range:
                                if k == src_k and j == src_j and i == src_i:
                                    continue
                                tx = big
                                if i > 0 and times[k, j, i - 1] < tx:
                                    tx = times[k, j, i - 1]
                                if i < nx - 1 and times[k, j, i + 1] < tx:
                                    tx = times[k, j, i + 1]
                                ty = big
                                if j > 0 and times[k, j - 1, i] < ty:
                                    ty = times[k, j - 1, i]
                                if j < ny - 1 and times[k, j + 1, i] < ty:
                                    ty = times[k, j + 1, i]
                                tz = big
                                if k > 0 and times[k - 1, j, i] < tz:
                                    tz = times[k - 1, j, i]
                                if k < nz - 1 and times[k + 1, j, i] < tz:
                                    tz = times[k + 1, j, i]
                                vals = [(tx, dx), (ty, dy), (tz, dz)]
                                vals.sort(key=lambda item: item[0])
                                new_t = _solve_local_fsm(
                                    vals[0][0],
                                    vals[1][0],
                                    vals[2][0],
                                    vals[0][1],
                                    vals[1][1],
                                    vals[2][1],
                                    float(slowness[k, j, i]),
                                )
                                if new_t < times[k, j, i]:
                                    times[k, j, i] = new_t
    return times


class FsmFieldCache:
    def __init__(self, model: VelocityModel, n_sweeps: int):
        self.model = model
        self.n_sweeps = n_sweeps
        self.dx = float(model.xg[1] - model.xg[0])
        self.dy = float(model.yg[1] - model.yg[0])
        self.dz = float(model.zg[1] - model.zg[0])
        self.slowness = 1.0 / np.maximum(model.vp, 1.0e-9)
        self.fields: dict[tuple[int, int, int], np.ndarray] = {}

    def field_for_receiver(self, receiver_xyz: np.ndarray) -> np.ndarray:
        kji = coord_to_index_xyz(float(receiver_xyz[0]), float(receiver_xyz[1]), float(receiver_xyz[2]), self.model)
        if kji not in self.fields:
            k, j, i = kji
            print(f"  FSM field for receiver node kji={kji}", flush=True)
            self.fields[kji] = fast_sweeping_3d(self.slowness, k, j, i, self.dx, self.dy, self.dz, self.n_sweeps)
        return self.fields[kji]


class FsmTravelTimeOperator:
    def __init__(self, model: VelocityModel, receivers: np.ndarray, field_cache: FsmFieldCache):
        self.model = model
        self.fields = [field_cache.field_for_receiver(rec) for rec in receivers]

    def predict(self, src: np.ndarray) -> np.ndarray:
        point = np.asarray(src, dtype=float)[None, :]
        return np.array([float(_interp_grid_field(point, field, self.model)[0]) for field in self.fields], dtype=float)


def travel_time_straight_ray(src: np.ndarray, rec: np.ndarray, model: VelocityModel, step_km: float) -> float:
    vec = rec - src
    distance = float(np.linalg.norm(vec))
    if not np.isfinite(distance) or distance <= 0.0:
        return math.inf
    n_steps = max(2, int(math.ceil(distance / max(step_km, 0.25))))
    u = (np.arange(n_steps, dtype=float) + 0.5) / n_steps
    pts = src[None, :] + u[:, None] * vec[None, :]
    speeds = _interp_vp(pts, model)
    if np.any(~np.isfinite(speeds)) or np.any(speeds <= 0.0):
        return math.inf
    return float((distance / n_steps) * np.sum(1.0 / speeds))


def predict_times(
    src: np.ndarray,
    receivers: np.ndarray,
    model: VelocityModel,
    step_km: float,
    operator: FsmTravelTimeOperator | None = None,
) -> np.ndarray:
    if operator is not None:
        return operator.predict(src)
    return np.array([travel_time_straight_ray(src, rec, model, step_km) for rec in receivers], dtype=float)


def residual_solution(
    src: np.ndarray,
    obs_travel: np.ndarray,
    receivers: np.ndarray,
    model: VelocityModel,
    step_km: float,
    huber_s: float = 1.0,
    operator: FsmTravelTimeOperator | None = None,
    station_static_s: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    pred = predict_times(src, receivers, model, step_km, operator=operator)
    if np.any(~np.isfinite(pred)):
        n = len(obs_travel)
        return 1.0e9, 1.0e9, 0.0, np.full(n, np.nan), pred
    if station_static_s is not None:
        pred = pred + np.asarray(station_static_s, dtype=float)
    origin_shift_s = float(np.median(obs_travel - pred))
    residual = obs_travel - pred - origin_shift_s
    rms = float(np.sqrt(np.mean(residual**2)))
    abs_r = np.abs(residual)
    huber = np.where(abs_r <= huber_s, residual**2, 2.0 * huber_s * abs_r - huber_s**2)
    objective = float(np.sqrt(np.mean(huber)))
    return objective, rms, origin_shift_s, residual, pred


def event_search_radii(event: pd.Series, receivers: np.ndarray) -> tuple[float, float]:
    xy_span = 0.0
    if len(receivers) > 1:
        xy = receivers[:, :2]
        xy_span = float(np.max(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))
    err_lat = float(event.get("error_lat_km", np.nan))
    err_lon = float(event.get("error_lon_km", np.nan))
    err_z = float(event.get("error_depth_km", np.nan))
    err_h = math.sqrt(err_lat**2 + err_lon**2) if np.isfinite(err_lat) and np.isfinite(err_lon) else 0.0
    rxy = max(18.0, min(70.0, 0.22 * xy_span + 2.0 * err_h))
    rz = max(12.0, min(50.0, 2.0 * err_z if np.isfinite(err_z) else 20.0))
    return rxy, rz


def clip_positions(positions: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(positions, bounds[:, 0]), bounds[:, 1])


def score_positions(
    positions: np.ndarray,
    obs_travel: np.ndarray,
    receivers: np.ndarray,
    model: VelocityModel,
    step_km: float,
    operator: FsmTravelTimeOperator | None = None,
    station_static_s: np.ndarray | None = None,
) -> np.ndarray:
    scores = []
    for pos in positions:
        obj, _, _, _, _ = residual_solution(
            pos,
            obs_travel,
            receivers,
            model,
            step_km,
            operator=operator,
            station_static_s=station_static_s,
        )
        scores.append(obj)
    return np.asarray(scores, dtype=float)


def initialize_swarm(
    center: np.ndarray,
    bounds: np.ndarray,
    rxy: float,
    rz: float,
    e1: np.ndarray,
    e2: np.ndarray,
    population: int,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, population, endpoint=False)
    theta += rng.uniform(-np.pi / population, np.pi / population, size=population)
    radial = rxy * np.sqrt(rng.uniform(0.05, 1.0, size=population))
    axial = rz * rng.uniform(-1.0, 1.0, size=population)
    positions = center[None, :] + radial[:, None] * (
        np.cos(theta)[:, None] * e1[None, :] + np.sin(theta)[:, None] * e2[None, :]
    )
    positions[:, 2] += axial
    positions += rng.normal(0.0, [0.8, 0.8, 0.4], size=positions.shape)
    positions[0] = center
    return clip_positions(positions, bounds)


def topological_neighbor_mean(positions: np.ndarray, pbest: np.ndarray, k_neighbors: int) -> np.ndarray:
    n = len(positions)
    if n <= 1:
        return pbest.copy()
    k = max(1, min(k_neighbors, n - 1))
    out = np.empty_like(pbest)
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=1)
        order = np.argsort(d)
        nn = order[1 : k + 1]
        out[i] = pbest[nn].mean(axis=0)
    return out


def polish_solution(
    best: np.ndarray,
    bounds: np.ndarray,
    obs_travel: np.ndarray,
    receivers: np.ndarray,
    model: VelocityModel,
    step_km: float,
    rxy: float,
    rz: float,
    operator: FsmTravelTimeOperator | None = None,
    station_static_s: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    best = best.copy()
    best_obj = residual_solution(
        best,
        obs_travel,
        receivers,
        model,
        step_km,
        operator=operator,
        station_static_s=station_static_s,
    )[0]
    steps = np.array([max(0.75, 0.12 * rxy), max(0.75, 0.12 * rxy), max(0.75, 0.12 * rz)], dtype=float)
    min_steps = np.array([0.15, 0.15, 0.15], dtype=float)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    while np.any(steps > min_steps):
        improved = False
        for direction in directions:
            cand = clip_positions((best + direction * steps)[None, :], bounds)[0]
            obj = residual_solution(
                cand,
                obs_travel,
                receivers,
                model,
                step_km,
                operator=operator,
                station_static_s=station_static_s,
            )[0]
            if obj + 1.0e-10 < best_obj:
                best, best_obj = cand, obj
                improved = True
        if not improved:
            steps *= 0.5
    return best, best_obj


def relocate_event(
    event: pd.Series,
    event_picks: pd.DataFrame,
    model: VelocityModel,
    population: int,
    iterations: int,
    neighbors: int,
    step_km: float,
    rng: np.random.Generator,
    polish: bool = True,
    travel_time_mode: str = "straight",
    fsm_cache: FsmFieldCache | None = None,
    depth_window_km: float = 0.0,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    event_picks = event_picks.drop_duplicates("station").copy()
    obs_travel = event_picks["travel_time_s"].to_numpy(dtype=float)
    station_static_s = (
        event_picks["station_static_s"].to_numpy(dtype=float)
        if "station_static_s" in event_picks.columns
        else np.zeros(len(event_picks), dtype=float)
    )
    receivers = event_picks[["receiver_x_km", "receiver_y_km"]].to_numpy(dtype=float)
    receivers = np.column_stack([receivers, np.zeros(len(receivers), dtype=float)])
    center = np.array([event["source_x_km"], event["source_y_km"], event["source_z_km"]], dtype=float)
    operator = None
    if travel_time_mode == "fsm":
        if fsm_cache is None:
            raise ValueError("FSM travel-time mode requires an FsmFieldCache.")
        operator = FsmTravelTimeOperator(model, receivers, fsm_cache)

    rxy, rz = event_search_radii(event, receivers)
    if depth_window_km > 0:
        zmin = max(model.zmin + 0.5, center[2] - depth_window_km)
        zmax = min(model.zmax, center[2] + depth_window_km)
    else:
        zmin = max(model.zmin + 0.5, center[2] - 2.5 * rz)
        zmax = min(model.zmax, center[2] + 2.5 * rz)
    if zmax <= zmin:
        zmin = max(model.zmin + 0.5, min(model.zmax - 0.5, center[2] - 0.5))
        zmax = min(model.zmax, zmin + 1.0)
    bounds = np.array(
        [
            [max(model.xmin, center[0] - 2.5 * rxy), min(model.xmax, center[0] + 2.5 * rxy)],
            [max(model.ymin, center[1] - 2.5 * rxy), min(model.ymax, center[1] + 2.5 * rxy)],
            [zmin, zmax],
        ],
        dtype=float,
    )
    e1, e2, z_axis = _axis_frame(receivers)
    positions = initialize_swarm(center, bounds, rxy, rz, e1, e2, population, rng)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=population)
    spin = rng.uniform(0.65, 1.35, size=population)

    scores = score_positions(
        positions,
        obs_travel,
        receivers,
        model,
        step_km,
        operator=operator,
        station_static_s=station_static_s,
    )
    pbest = positions.copy()
    pbest_scores = scores.copy()
    ibest = int(np.argmin(pbest_scores))
    gbest = pbest[ibest].copy()
    gbest_score = float(pbest_scores[ibest])

    history = []
    for it in range(iterations):
        frac = max(0.0, 1.0 - (it + 1) / max(iterations, 1))
        shrink = 0.08 + 0.92 * frac**1.65
        rxy_k = max(0.4, rxy * shrink)
        rz_k = max(0.35, rz * shrink)
        topo = topological_neighbor_mean(positions, pbest, neighbors)
        theta_base = 2.0 * np.pi * (it + 1) * (0.18 + 0.72 * frac)
        new_positions = np.empty_like(positions)
        for i in range(population):
            theta = phase[i] + spin[i] * theta_base + rng.normal(0.0, 0.10 + 0.20 * frac)
            radial_scale = rxy_k * rng.uniform(0.25, 1.0)
            axial_scale = rz_k * math.sin(theta + phase[i]) * rng.uniform(0.35, 1.0)
            helix = (
                gbest
                + radial_scale * (math.cos(theta) * e1 + math.sin(theta) * e2)
                + axial_scale * z_axis
            )
            jitter = rng.normal(0.0, [0.22 * rxy_k, 0.22 * rxy_k, 0.18 * rz_k])
            guided = 0.56 * helix + 0.18 * pbest[i] + 0.16 * topo[i] + 0.10 * gbest + 0.18 * jitter
            if rng.random() < 0.06 * frac:
                guided = np.array(
                    [
                        rng.uniform(bounds[0, 0], bounds[0, 1]),
                        rng.uniform(bounds[1, 0], bounds[1, 1]),
                        rng.uniform(bounds[2, 0], bounds[2, 1]),
                    ],
                    dtype=float,
                )
            new_positions[i] = guided
        positions = clip_positions(new_positions, bounds)
        scores = score_positions(
            positions,
            obs_travel,
            receivers,
            model,
            step_km,
            operator=operator,
            station_static_s=station_static_s,
        )
        improved = scores < pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = scores[improved]
        ibest = int(np.argmin(pbest_scores))
        if float(pbest_scores[ibest]) < gbest_score:
            gbest = pbest[ibest].copy()
            gbest_score = float(pbest_scores[ibest])
        if (it == 0) or ((it + 1) % 5 == 0) or (it == iterations - 1):
            history.append(
                {
                    "event_id": event["event_id"],
                    "iteration": it + 1,
                    "best_objective_s": gbest_score,
                    "median_objective_s": float(np.median(scores)),
                    "rxy_km": rxy_k,
                    "rz_km": rz_k,
                }
            )

    if polish:
        gbest, gbest_score = polish_solution(
            gbest,
            bounds,
            obs_travel,
            receivers,
            model,
            step_km,
            rxy,
            rz,
            operator=operator,
            station_static_s=station_static_s,
        )

    sgc_obj, sgc_rms, sgc_shift, sgc_res, sgc_pred = residual_solution(
        center, obs_travel, receivers, model, step_km, operator=operator, station_static_s=station_static_s
    )
    obj, rms, origin_shift_s, residual, pred = residual_solution(
        gbest, obs_travel, receivers, model, step_km, operator=operator, station_static_s=station_static_s
    )
    hlat, hlon = xy_to_latlon(gbest[0], gbest[1])
    hdist = float(np.linalg.norm(gbest[:2] - center[:2]))
    dd = float(gbest[2] - center[2])
    dist3 = float(np.linalg.norm(gbest - center))
    origin_ts = pd.to_datetime(event["origin_time_utc"], utc=True, errors="coerce")
    new_origin = origin_ts + pd.to_timedelta(origin_shift_s, unit="s") if pd.notna(origin_ts) else pd.NaT

    result = {
        "event_id": event["event_id"],
        "sgc_origin_time_utc": event["origin_time_utc"],
        "hsso_origin_time_utc": new_origin.isoformat() if pd.notna(new_origin) else "",
        "origin_shift_s": origin_shift_s,
        "sgc_lat": event["event_lat"],
        "sgc_lon": event["event_lon"],
        "sgc_depth_km": event["event_depth_km"],
        "hsso_lat": hlat,
        "hsso_lon": hlon,
        "hsso_depth_km": float(gbest[2]),
        "delta_horizontal_km": hdist,
        "delta_depth_km": dd,
        "delta_3d_km": dist3,
        "sgc_catalog_rms_s": event.get("rms_s", np.nan),
        "sgc_model_p_rms_s": sgc_rms,
        "hsso_model_p_rms_s": rms,
        "rms_reduction_s": sgc_rms - rms,
        "rms_reduction_pct": 100.0 * (sgc_rms - rms) / sgc_rms if sgc_rms > 0 else np.nan,
        "huber_objective_s": obj,
        "n_picks": int(len(event_picks)),
        "n_stations": int(event_picks["station"].nunique()),
        "catalog_phases": event.get("phases", np.nan),
        "catalog_gap_deg": event.get("gap_deg", np.nan),
        "magnitude": event.get("magnitude", np.nan),
        "region": event.get("region", ""),
        "selection_source": event.get("selection_source", ""),
        "rms_source": event.get("rms_source", ""),
        "travel_time_mode": travel_time_mode,
        "depth_window_km": float(depth_window_km),
        "station_static_mean_s": float(np.mean(station_static_s)) if len(station_static_s) else 0.0,
        "station_static_rms_s": float(np.sqrt(np.mean(station_static_s**2))) if len(station_static_s) else 0.0,
        "search_rxy0_km": rxy,
        "search_rz0_km": rz,
    }

    residual_df = event_picks.copy()
    residual_df["station_static_s"] = station_static_s
    residual_df["pred_sgc_model_s"] = sgc_pred
    residual_df["residual_sgc_model_s"] = sgc_res
    residual_df["pred_hsso_model_s"] = pred
    residual_df["residual_hsso_model_s"] = residual
    residual_df["hsso_origin_shift_s"] = origin_shift_s
    residual_df["sgc_origin_shift_for_model_s"] = sgc_shift

    return result, residual_df, pd.DataFrame(history)


def plot_results(results: pd.DataFrame, selected: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not available; skipping PNG figures.", flush=True)
        return
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    order = np.arange(len(results))
    labels = results["event_id"].str.replace("SGC", "", regex=False)
    ax.bar(order - 0.18, results["sgc_model_p_rms_s"], width=0.36, label="SGC fixed hypocenter")
    ax.bar(order + 0.18, results["hsso_model_p_rms_s"], width=0.36, label="HSSO-Hypo relocated")
    ax.set_xticks(order)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("P-pick RMS on 3-D Vp model (s)")
    ax.grid(axis="y", lw=0.3, alpha=0.45)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rms_comparison_hsso_vs_sgc.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(selected["event_lon"], selected["event_lat"], c="tab:blue", s=48, label="SGC")
    ax.scatter(results["hsso_lon"], results["hsso_lat"], c="tab:red", s=48, marker="x", label="HSSO-Hypo")
    for _, row in results.iterrows():
        ax.plot([row["sgc_lon"], row["hsso_lon"]], [row["sgc_lat"], row["hsso_lat"]], color="0.35", lw=0.8)
        ax.text(row["sgc_lon"], row["sgc_lat"], str(row["event_id"]).replace("SGC", ""), fontsize=7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Relocation vectors for selected SGC events")
    ax.grid(lw=0.3, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "relocation_vectors_map.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sc = ax.scatter(
        results["delta_horizontal_km"],
        results["delta_depth_km"],
        c=results["rms_reduction_pct"],
        s=70,
        cmap="viridis",
    )
    for _, row in results.iterrows():
        ax.text(row["delta_horizontal_km"], row["delta_depth_km"], str(row["event_id"]).replace("SGC", ""), fontsize=7)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_xlabel("Horizontal shift from SGC (km)")
    ax.set_ylabel("Depth shift from SGC (km)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("RMS reduction (%)")
    ax.grid(lw=0.3, alpha=0.45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "relocation_shift_vs_rms_gain.png", dpi=220)
    plt.close(fig)


def write_depth_bin_outputs(results: pd.DataFrame, depth_bin_km: float) -> None:
    if results.empty:
        return
    table = results.copy()
    table["depth_bin_min_km"] = np.floor(table["sgc_depth_km"] / depth_bin_km) * depth_bin_km
    table["depth_bin_max_km"] = table["depth_bin_min_km"] + depth_bin_km
    summary = (
        table.groupby(["depth_bin_min_km", "depth_bin_max_km"])
        .agg(
            events=("event_id", "size"),
            mean_sgc_model_p_rms_s=("sgc_model_p_rms_s", "mean"),
            mean_hsso_model_p_rms_s=("hsso_model_p_rms_s", "mean"),
            mean_rms_reduction_pct=("rms_reduction_pct", "mean"),
            median_horizontal_shift_km=("delta_horizontal_km", "median"),
            median_abs_depth_shift_km=("delta_depth_km", lambda s: float(np.median(np.abs(s)))),
            xml_proxy_events=("selection_source", lambda s: int((s == "xml_arrival_residual_proxy").sum())),
            final_catalog_events=("selection_source", lambda s: int((s == "final_sgc_catalog").sum())),
        )
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "depth_bin_relocation_summary.csv", index=False)

    if not HAS_MATPLOTLIB:
        return
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(summary["depth_bin_min_km"], summary["depth_bin_max_km"])]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.plot(x, summary["mean_sgc_model_p_rms_s"], marker="o", label="SGC fixed hypocenter")
    ax.plot(x, summary["mean_hsso_model_p_rms_s"], marker="o", label="HSSO-Hypo relocated")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Depth interval (km)")
    ax.set_ylabel("Mean P-pick RMS on 3-D Vp model (s)")
    ax.grid(lw=0.3, alpha=0.45)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rms_by_depth_bin.png", dpi=220)
    plt.close(fig)


def write_summary(config: dict, selected: pd.DataFrame, results: pd.DataFrame) -> None:
    payload = {
        "config": config,
        "inputs": {
            "picks_csv": str(config.get("picks_csv", PICKS_CSV)),
            "catalog_csv": str(config.get("catalog_csv", CATALOG_CSV)),
            "station_statics_csv": str(config.get("station_statics_csv", "")),
            "xml_picks_csv": str(XML_PICKS_CSV),
            "xml_events_csv": str(XML_EVENTS_CSV),
            "grid_npz": str(GRID_NPZ),
            "vp_npy": str(VP_NPY),
            "travel_time_operator": (
                "3-D FSM/grid-eikonal receiver travel-time fields"
                if config.get("travel_time_mode") == "fsm"
                else "straight-ray integration through final 3-D Vp model"
            ),
        },
        "selection": {
            "events_selected": int(len(selected)),
            "rms_max_s": config["rms_max"],
            "min_stations": config["min_stations"],
            "selection_mode": config["selection_mode"],
            "depth_min_km": config["depth_min_km"],
            "depth_max_km": config["depth_max_km"],
            "depth_bin_km": config["depth_bin_km"],
            "events_per_bin": config["events_per_bin"],
            "event_ids": selected["event_id"].tolist(),
        },
        "results": {
            "mean_sgc_model_p_rms_s": float(results["sgc_model_p_rms_s"].mean()),
            "mean_hsso_model_p_rms_s": float(results["hsso_model_p_rms_s"].mean()),
            "median_horizontal_shift_km": float(results["delta_horizontal_km"].median()),
            "median_abs_depth_shift_km": float(results["delta_depth_km"].abs().median()),
            "events_improved": int((results["rms_reduction_s"] > 0).sum()),
        },
    }
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(selected: pd.DataFrame, results: pd.DataFrame, config: dict) -> None:
    report_path = RESULTS_DIR / "relocation_report.md"
    travel_time_mode = config.get("travel_time_mode", "straight")
    if travel_time_mode == "fsm":
        travel_time_text = (
            "Travel times are computed with 3-D Fast Sweeping Method (FSM) "
            "receiver-centered travel-time fields on the final Vp grid. "
        )
    else:
        travel_time_text = "Travel times are computed by straight-ray integration through the final 3-D Vp model. "
    run_label = str(config.get("run_label", "") or "").strip()
    data_prefix = f"data/{run_label}" if run_label else "data"
    results_prefix = f"results/{run_label}" if run_label else "results"
    figures_prefix = f"figures/{run_label}" if run_label else "figures"
    static_text = (
        "Station statics are applied as station-time corrections. "
        if config.get("apply_station_statics")
        else "Station statics are not applied. "
    )
    depth_window = float(config.get("depth_window_km", 0.0) or 0.0)
    depth_text = (
        f"The hypocentral search is constrained to SGC depth +/- {depth_window:.1f} km. "
        if depth_window > 0
        else "The hypocentral depth search uses the adaptive HSSO vertical radius. "
    )
    merged = results.merge(
        selected[["event_id", "time_utc", "region", "stations"]],
        on="event_id",
        how="left",
        suffixes=("", "_selected"),
    )
    lines = [
        "# HSSO-Hypo relocation report",
        "",
        f"Pilot relocation of {len(results)} SGC earthquakes with input RMS < 1 s.",
        "",
        travel_time_text
        + depth_text
        + static_text
        + "The catalog RMS is not directly comparable with the model RMS because the SGC catalog may use "
        "additional phases/stations and a different velocity model.",
        "",
        "| Event | Depth km | Time UTC | Stations | RMS input | SGC model RMS | HSSO RMS | Gain | dH km | dZ km | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            "| {event} | {depth:.1f} | {time} | {nsta:d} | {cat:.2f} | {sgc:.3f} | {hsso:.3f} | {gain:.1f}% | "
            "{dh:.2f} | {dz:.2f} | {source} |".format(
                event=row["event_id"],
                depth=float(row["sgc_depth_km"]),
                time=str(row.get("time_utc", "")),
                nsta=int(row["n_stations"]),
                cat=float(row["sgc_catalog_rms_s"]),
                sgc=float(row["sgc_model_p_rms_s"]),
                hsso=float(row["hsso_model_p_rms_s"]),
                gain=float(row["rms_reduction_pct"]),
                dh=float(row["delta_horizontal_km"]),
                dz=float(row["delta_depth_km"]),
                source=str(row.get("selection_source", "")),
            )
        )
    lines.extend(
        [
            "",
            "Main files:",
            "",
            f"- `{data_prefix}/selected_events_rms_lt1_for_hsso.csv`",
            f"- `{data_prefix}/selected_p_picks_for_hsso.csv`",
            f"- `{data_prefix}/depth_stratified_selection_bins.csv`",
            f"- `{results_prefix}/relocation_results_hsso_vs_sgc.csv`",
            f"- `{results_prefix}/pick_residuals_hsso_vs_sgc.csv`",
            f"- `{results_prefix}/hsso_convergence_history.csv`",
            f"- `{results_prefix}/depth_bin_relocation_summary.csv`",
            f"- `{results_prefix}/run_summary.json`",
            f"- `{figures_prefix}/rms_comparison_hsso_vs_sgc.png`",
            f"- `{figures_prefix}/relocation_vectors_map.png`",
            f"- `{figures_prefix}/relocation_shift_vs_rms_gain.png`",
            f"- `{figures_prefix}/rms_by_depth_bin.png`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global DATA_DIR, RESULTS_DIR, FIG_DIR
    args = parse_args()
    if not args.run_label and args.travel_time_mode != "straight":
        args.run_label = args.travel_time_mode
    if args.run_label:
        DATA_DIR = DATA_DIR / args.run_label
        RESULTS_DIR = RESULTS_DIR / args.run_label
        FIG_DIR = FIG_DIR / args.run_label
    ensure_dirs()
    rng = np.random.default_rng(args.seed)
    model = load_velocity_model()
    fsm_cache = FsmFieldCache(model, args.fsm_sweeps) if args.travel_time_mode == "fsm" else None
    picks, catalog = read_inputs(Path(args.picks_csv), Path(args.catalog_csv))
    picks, catalog = combine_sources(picks, catalog, args.include_xml_picks)
    picks = configure_station_statics(picks, args.station_statics_csv, args.apply_station_statics)
    model_picks = filter_picks_for_model(picks, model)
    selected = select_events(
        model_picks,
        catalog,
        model,
        args.n_events,
        args.rms_max,
        args.min_stations,
        selection_mode=args.selection_mode,
        depth_min_km=args.depth_min_km,
        depth_max_km=args.depth_max_km,
        depth_bin_km=args.depth_bin_km,
        events_per_bin=args.events_per_bin,
    )
    selected_picks = model_picks[model_picks["event_id"].isin(selected["event_id"])].copy()
    selected.to_csv(DATA_DIR / "selected_events_rms_lt1_for_hsso.csv", index=False)
    selected_picks.to_csv(DATA_DIR / "selected_p_picks_for_hsso.csv", index=False)

    results: list[dict] = []
    residuals: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    selected_by_id = selected.set_index("event_id", drop=False)
    for event_id in selected["event_id"]:
        event = selected_by_id.loc[event_id]
        event_picks = selected_picks[selected_picks["event_id"] == event_id].copy()
        result, residual_df, history_df = relocate_event(
            event,
            event_picks,
            model,
            population=args.population,
            iterations=args.iterations,
            neighbors=args.neighbors,
            step_km=args.ray_step_km,
            rng=rng,
            polish=not args.no_polish,
            travel_time_mode=args.travel_time_mode,
            fsm_cache=fsm_cache,
            depth_window_km=args.depth_window_km,
        )
        results.append(result)
        residuals.append(residual_df)
        histories.append(history_df)
        print(
            f"{event_id}: SGC-model RMS={result['sgc_model_p_rms_s']:.3f}s, "
            f"HSSO RMS={result['hsso_model_p_rms_s']:.3f}s, "
            f"dH={result['delta_horizontal_km']:.2f}km, dZ={result['delta_depth_km']:.2f}km",
            flush=True,
        )

    results_df = pd.DataFrame(results)
    residuals_df = pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame()
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    results_df.to_csv(RESULTS_DIR / "relocation_results_hsso_vs_sgc.csv", index=False)
    residuals_df.to_csv(RESULTS_DIR / "pick_residuals_hsso_vs_sgc.csv", index=False)
    history_df.to_csv(RESULTS_DIR / "hsso_convergence_history.csv", index=False)
    plot_results(results_df, selected)
    write_depth_bin_outputs(results_df, args.depth_bin_km)

    config = vars(args).copy()
    config["project_dir"] = str(PROJECT_DIR)
    config["picks_before_model_filter"] = int(len(picks))
    config["picks_after_model_filter"] = int(len(model_picks))
    config["effective_n_events"] = int(len(selected))
    config["n_events_ignored_for_depth_stratified"] = bool(args.selection_mode == "depth-stratified")
    write_summary(config, selected, results_df)
    write_markdown_report(selected, results_df, config)
    print("\nSaved outputs:")
    print(f"  {DATA_DIR / 'selected_events_rms_lt1_for_hsso.csv'}")
    print(f"  {RESULTS_DIR / 'relocation_results_hsso_vs_sgc.csv'}")
    print(f"  {RESULTS_DIR / 'pick_residuals_hsso_vs_sgc.csv'}")
    print(f"  {RESULTS_DIR / 'relocation_report.md'}")
    if HAS_MATPLOTLIB:
        print(f"  {FIG_DIR / 'rms_comparison_hsso_vs_sgc.png'}")


if __name__ == "__main__":
    main()
