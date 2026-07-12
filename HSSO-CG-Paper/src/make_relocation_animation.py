#!/usr/bin/env python
"""Slow 3-D PSO/HSSO-E animation using the reported FSM objective and core."""
from pathlib import Path
import sys, zlib

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HYPODIR = ROOT / "HSSO-Hypo"
sys.path.insert(0, str(ROOT / "HSSO-Hypo/src"))
sys.path.insert(0, str(ROOT / "HSSO-Core/src"))
import hsso_hypo_relocation as hypo
from run_shared_optimizer_comparison import load_dataset, evaluate_matrix
from swarm_core import SwarmConfig, optimize

EVENT_ID = "SGC2025yxrhgz"
OUT = ROOT / "HSSO-CG-Paper/animations"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    catalog, picks = load_dataset("new30")
    heldout = np.genfromtxt(HYPODIR / "data/sgc_new_30/heldout_station_statics.csv",
                            delimiter=",", names=True, dtype=None, encoding="utf-8")
    static_map = {str(row["station"]): float(row["station_static_s"]) for row in heldout}
    picks["station_static_s"] = picks.station.astype(str).map(static_map).fillna(0.0)
    event = catalog[catalog.event_id == EVENT_ID].iloc[0]
    ep = picks[picks.event_id == EVENT_ID].drop_duplicates("station")
    model = hypo.load_velocity_model(); cache = hypo.FsmFieldCache(model, 8)
    cache_dir = ROOT / "HSSO-Hypo/data/fsm_field_cache_sweeps8"
    for f in cache_dir.glob("field_k*_j*_i*.npy"):
        parts = f.stem.replace("field_k", "").replace("_j", " ").replace("_i", " ").split()
        cache.fields[tuple(map(int, parts))] = np.load(f)
    receivers = np.column_stack([ep.receiver_x_km, ep.receiver_y_km, np.zeros(len(ep))])
    obs = ep.travel_time_s.to_numpy(float); statics = ep.station_static_s.to_numpy(float)
    center = np.array([event.source_x_km, event.source_y_km, event.source_z_km])
    operator = hypo.FsmTravelTimeOperator(model, receivers, cache)
    rxy, _ = hypo.event_search_radii(event, receivers)
    bounds = np.array([[max(model.xmin, center[0]-2.5*rxy), min(model.xmax, center[0]+2.5*rxy)],
                       [max(model.ymin, center[1]-2.5*rxy), min(model.ymax, center[1]+2.5*rxy)],
                       [max(model.zmin+0.5, center[2]-20), min(model.zmax, center[2]+20)]])
    objective = lambda x: evaluate_matrix(x, operator, obs, statics)
    seed = int((20260710 + zlib.crc32(EVENT_ID.encode())) % (2**32-1))
    cfg = SwarmConfig(particles=30, iterations=300)
    runs = {m: optimize(m, objective, bounds, seed, cfg) for m in ["PSO", "HSSO-E"]}

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    fig = plt.figure(figsize=(14, 6.8), facecolor="#07111f")
    panels = []
    for j, method in enumerate(runs):
        ax = fig.add_subplot(1, 2, j+1, projection="3d", facecolor="#07111f")
        ax.scatter(receivers[:,0], receivers[:,1], receivers[:,2], marker="^", s=75,
                   c="#58d5ff", edgecolors="white", label="Stations")
        ax.scatter(*center, marker="*", s=230, c="#ff4f87", edgecolors="white", label="SGC")
        final = runs[method]["best"]
        ax.scatter(*final, marker="*", s=210, c="#55f28c", edgecolors="white", label="Relocated")
        sc = ax.scatter([], [], [], s=55, c="#ffe600", edgecolors="#5b4300", linewidths=.6)
        trail, = ax.plot([], [], [], c="#55f28c", lw=2.0)
        info = ax.text2D(.03, .96, "", transform=ax.transAxes, va="top", color="white",
                         bbox=dict(boxstyle="round", fc="#0c1d31", ec="#55f28c", alpha=.92))
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=(bounds[2,1], 0), xlabel="x (km)",
               ylabel="y (km)", zlabel="Depth (km)")
        ax.set_title(method, color="white", fontsize=15, weight="bold")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white"); ax.zaxis.label.set_color("white")
        ax.legend(loc="lower left", fontsize=8, framealpha=.85)
        panels.append((sc, trail, info))
    fig.suptitle(f"FSM hypocenter relocation — {EVENT_ID}", color="white", fontsize=17, weight="bold")
    frames = [0]*12 + [k for k in range(0,301,3) for _ in range(2)] + [300]*24
    def update(k):
        artists=[]
        for method,(sc,trail,info) in zip(runs,panels):
            h=runs[method]["history"]; pos=h[k]["positions"]
            sc._offsets3d=(pos[:,0],pos[:,1],pos[:,2])
            bh=np.array([x["best_position"] for x in h[:k+1]])
            trail.set_data(bh[:,0],bh[:,1]); trail.set_3d_properties(bh[:,2])
            info.set_text(f"Iteration {k:03d}\nHuber objective = {h[k]['best_objective']:.4f} s\nSwarm radius = {h[k]['radius']:.3f} km")
            artists += [sc,trail,info]
        return artists
    ani=animation.FuncAnimation(fig,update,frames=frames,interval=125,blit=False)
    ani.save(OUT/f"pso_vs_hsso_e_fsm_{EVENT_ID}.mp4",
             writer=animation.FFMpegWriter(fps=8,bitrate=4200,codec="libx264",extra_args=["-pix_fmt","yuv420p"]))
    update(24); fig.savefig(OUT/f"preview_{EVENT_ID}.png",dpi=180,bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__": main()
