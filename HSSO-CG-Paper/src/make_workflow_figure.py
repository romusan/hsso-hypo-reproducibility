#!/usr/bin/env python
"""Create the vector workflow figure used by the manuscript."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "HSSO-CG-Paper/figures"; FIG.mkdir(parents=True, exist_ok=True)


def main():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_xlim(0, 12.2); ax.set_ylim(0, 4.4); ax.axis("off")
    stages = [
        (0.25, "1  INPUTS", "SGC catalog and P picks\n3-D Vp grid\nstation-static table", "#dceaf7"),
        (2.65, "2  TRAVEL TIMES", "receiver-centred FSM\n14 cached Eikonal fields\ntrilinear interpolation", "#d8eee8"),
        (5.05, "3  OPTIMIZATION", "shared PSO/LPSO/HSSO core\nanalytic origin-time elimination\nHuber loss and bounded depth", "#f4e5bf"),
        (7.45, "4  QUALITY CONTROL", "common-model RMS\ndepth-boundary flag\n|dZ| >= 15 km; dH >= 25 km", "#f4d8d8"),
        (9.85, "5  OUTPUTS", "hypocenter and origin time\nresidual/convergence tables\nQC review list and figures", "#e6ddf3"),
    ]
    width, height, y = 2.05, 2.18, 1.45
    for x, title, body, color in stages:
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08",
                             linewidth=1.1, edgecolor="#374151", facecolor=color)
        ax.add_patch(box)
        ax.text(x+0.13, y+height-0.28, title, weight="bold", va="top", color="#172033")
        ax.text(x+0.13, y+height-0.72, body, va="top", linespacing=1.45, fontsize=7.8, color="#172033")
    for i in range(4):
        start = stages[i][0]+width+0.05; end = stages[i+1][0]-0.05
        ax.add_patch(FancyArrowPatch((start, y+height/2), (end, y+height/2), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.3, color="#374151"))
    audit = FancyBboxPatch((0.25, 0.35), 11.65, 0.60, boxstyle="round,pad=0.03,rounding_size=0.06",
                           linewidth=1.0, edgecolor="#4b5563", facecolor="#f3f4f6")
    ax.add_patch(audit)
    ax.text(0.48, 0.65, "REPRODUCIBILITY LAYER", weight="bold", va="center", color="#172033")
    ax.text(2.45, 0.65, "one optimizer module  |  fixed seeds and budgets  |  cached-field provenance  |  raw event rows  |  environment and QC audit",
            va="center", color="#374151")
    ax.annotate("", xy=(6.1, 1.42), xytext=(6.1, 0.96), arrowprops=dict(arrowstyle="-|>", color="#4b5563"))
    fig.tight_layout(pad=.3)
    fig.savefig(FIG / "hsso_hypo_workflow.pdf", bbox_inches="tight")
    fig.savefig(FIG / "hsso_hypo_workflow.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__": main()
