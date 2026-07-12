#!/usr/bin/env python
"""Create manuscript tables, paired tests, QC flags, and publication figures."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "HSSO-CG-Paper" / "results"
FIG = ROOT / "HSSO-CG-Paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adjusted = np.empty(len(p)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx]); adjusted[idx] = min(1.0, running)
    return adjusted


def matched_rank_biserial(diff):
    """Matched-pairs rank-biserial effect; negative values favour HSSO-E."""
    values = np.asarray(diff, float)
    values = values[np.isfinite(values) & (values != 0.0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_tests(seismic):
    tests = []
    for dataset in seismic.dataset.unique():
        pivot = seismic[seismic.dataset == dataset].pivot(index="event_id", columns="method", values="relocated_rms_s")
        for other in ["PSO", "LPSO", "HSSO-no-helix"]:
            diff = pivot["HSSO-E"] - pivot[other]
            w, p = wilcoxon(pivot["HSSO-E"], pivot[other], alternative="two-sided", zero_method="zsplit")
            tests.append({"dataset": dataset, "comparison": f"HSSO-E vs {other}", "n": len(diff),
                          "W": w, "p_raw": p, "hsso_e_lower_rms": int((diff < 0).sum()),
                          "ties": int((diff == 0).sum()), "other_lower_rms": int((diff > 0).sum()),
                          "median_paired_difference_s": np.median(diff),
                          "rank_biserial_hsso_minus_other": matched_rank_biserial(diff)})
    tests = pd.DataFrame(tests); tests["p_holm"] = holm(tests.p_raw)
    return tests


def arguments():
    p=argparse.ArgumentParser(); p.add_argument("--seismic-results",default=str(ROOT/"HSSO-Hypo/results/shared_core_fsm/relocation_comparison.csv"))
    return p.parse_args()


def main():
    args=arguments()
    math = pd.read_csv(OUT / "mathematical_validation_runs.csv")
    seismic = pd.read_csv(args.seismic_results)
    seismic.to_csv(OUT / "seismic_validation_runs.csv", index=False)
    summary = seismic.groupby(["dataset", "method"]).agg(
        events=("event_id", "size"), mean_rms_s=("relocated_rms_s", "mean"),
        median_rms_s=("relocated_rms_s", "median"), median_gain_s=("rms_reduction_s", "median"),
        improved_vs_sgc=("rms_reduction_s", lambda x: int((x > 0).sum())),
        median_dh_km=("delta_horizontal_km", "median"),
        median_abs_dz_km=("delta_depth_km", lambda x: np.median(np.abs(x))),
        depth_boundary=("depth_boundary", "sum"), large_dh=("large_horizontal_shift", "sum"),
        median_runtime_s=("optimizer_elapsed_s", "median"), total_runtime_s=("optimizer_elapsed_s", "sum")
    ).reset_index()
    summary.to_csv(OUT / "seismic_validation_summary.csv", index=False)
    qc = seismic[(seismic.depth_boundary) | (seismic.large_horizontal_shift) |
                 (seismic.large_vertical_shift)].copy()
    qc.to_csv(OUT / "quality_control_flags.csv", index=False)
    qc_hsso = qc[qc.method == "HSSO-E"][[
        "dataset","event_id","sgc_depth_km","relocated_depth_km","sgc_model_rms_s","relocated_rms_s",
        "delta_horizontal_km","delta_depth_km","depth_boundary","large_horizontal_shift","large_vertical_shift"
    ]].sort_values(["dataset","event_id"])
    qc_hsso.to_csv(OUT / "supplementary_hsso_e_flagged_events.csv", index=False)
    event_spread = seismic.groupby(["dataset", "event_id"]).agg(
        rms_min=("relocated_rms_s", "min"), rms_max=("relocated_rms_s", "max"),
        lat_min=("relocated_lat", "min"), lat_max=("relocated_lat", "max"),
        lon_min=("relocated_lon", "min"), lon_max=("relocated_lon", "max"),
        depth_min_km=("relocated_depth_km", "min"), depth_max_km=("relocated_depth_km", "max"),
        boundary_methods=("depth_boundary", "sum"), dh25_methods=("large_horizontal_shift", "sum")
    ).reset_index()
    event_spread["rms_range_s"] = event_spread.rms_max - event_spread.rms_min
    event_spread["depth_range_km"] = event_spread.depth_max_km - event_spread.depth_min_km
    event_spread["horizontal_range_km"] = np.hypot(
        (event_spread.lat_max-event_spread.lat_min)*111.32,
        (event_spread.lon_max-event_spread.lon_min)*111.32*np.cos(np.deg2rad(7.0)))
    event_spread["within_1m"] = ((event_spread.horizontal_range_km <= .001) &
                                  (event_spread.depth_range_km <= .001))
    event_spread["within_100m"] = ((event_spread.horizontal_range_km <= .1) &
                                    (event_spread.depth_range_km <= .1))
    event_spread.to_csv(OUT / "eventwise_optimizer_equivalence_audit.csv", index=False)
    tests = paired_tests(seismic)
    tests.to_csv(OUT / "seismic_paired_tests.csv", index=False)
    history = pd.read_csv(Path(args.seismic_results).with_name("convergence_history.csv"))
    final_best = history.groupby(["dataset", "event_id", "method"]).best_objective.transform("min")
    history["gap_to_final"] = history.best_objective - final_best
    first_close = history[history.gap_to_final <= 1.0e-4].groupby(
        ["dataset", "event_id", "method"], as_index=False).iteration.min()
    convergence = first_close.groupby(["dataset", "method"]).iteration.agg(
        median_iteration="median", mean_iteration="mean").reset_index()
    convergence.to_csv(OUT / "seismic_convergence_summary.csv", index=False)

    colors = {"PSO":"#e69f00", "LPSO":"#4078b8", "HSSO-no-helix":"#888888", "HSSO-E":"#009e73"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    msum = math.groupby(["problem", "method"]).success.sum().unstack()
    x = np.arange(len(msum)); width = 0.19
    for j, method in enumerate(colors):
        axes[0].bar(x + (j-1.5)*width, msum[method], width, color=colors[method], label=method)
    axes[0].set_xticks(x, [v.replace("-", "\n", 1) for v in msum.index], fontsize=8)
    axes[0].set_ylabel("Successful runs / 30"); axes[0].set_ylim(0, 32); axes[0].grid(axis="y", alpha=.25)
    methods = list(colors)
    data = [seismic[seismic.method == m].relocated_rms_s for m in methods]
    bp = axes[1].boxplot(data, tick_labels=methods, patch_artist=True, showfliers=False)
    for box, method in zip(bp["boxes"], methods): box.set_facecolor(colors[method]); box.set_alpha(.75)
    axes[1].set_ylabel("Relocated P-pick RMS (s)"); axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False)
    axes[0].set_title("Mathematical verification"); axes[1].set_title("62 SGC earthquakes")
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(FIG / "algorithm_and_seismic_validation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "algorithm_and_seismic_validation.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    hsso = seismic[seismic.method == "HSSO-E"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].scatter(hsso.sgc_model_rms_s, hsso.relocated_rms_s,
                    c=hsso.dataset.map({"stratified32":"#4078b8", "new30":"#d55e00"}), s=42, alpha=.85)
    lim = max(hsso.sgc_model_rms_s.max(), hsso.relocated_rms_s.max()) * 1.04
    axes[0].plot([0,lim],[0,lim],"k--",lw=1); axes[0].set(xlim=(0,lim),ylim=(0,lim),
        xlabel="SGC hypocenter RMS in common model (s)", ylabel="HSSO-E relocated RMS (s)")
    axes[1].scatter(hsso.delta_horizontal_km, hsso.delta_depth_km,
                    c=np.where(hsso.depth_boundary,"#d62728","#009e73"), s=45, alpha=.85)
    axes[1].axvline(25,color="k",ls="--",lw=1); axes[1].axhline(0,color=".4",lw=.8)
    axes[1].set(xlabel="Horizontal displacement (km)", ylabel="Depth displacement (km)")
    for ax in axes: ax.grid(alpha=.22)
    axes[0].set_title("Common-data RMS comparison"); axes[1].set_title("Predefined quality-control diagnostics")
    fig.tight_layout(); fig.savefig(FIG / "hsso_e_relocation_qc.pdf", bbox_inches="tight")
    fig.savefig(FIG / "hsso_e_relocation_qc.png", dpi=240, bbox_inches="tight"); plt.close(fig)
    print(summary.to_string(index=False)); print("\nPaired tests\n", tests.to_string(index=False))


if __name__ == "__main__":
    main()
