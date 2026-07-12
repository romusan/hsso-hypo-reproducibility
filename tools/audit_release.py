#!/usr/bin/env python
"""Audit the manuscript-facing reproducibility claims using only stdlib."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "HSSO-CG-Paper" / "results"


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"release audit: FAIL - {message}")


def main() -> None:
    paired = rows("seismic_paired_tests.csv")
    effect = "rank_biserial_hsso_minus_other"
    require(len(paired) == 6, "paired-test table must contain six comparisons")
    require(
        all(row.get("p_raw") and row.get("p_holm") and row.get(effect) for row in paired),
        "paired tests must include raw/adjusted p values and effect sizes",
    )

    flagged = rows("supplementary_hsso_e_flagged_events.csv")
    counts = Counter(row["dataset"] for row in flagged)
    require(len(flagged) == 35, "QC supplement must contain 35 flagged events")
    require(len({(row["dataset"], row["event_id"]) for row in flagged}) == 35,
            "QC event identifiers must be unique within data set")
    require(counts == {"stratified32": 15, "new30": 20},
            "QC counts must be 15 stratified and 20 held out")

    seed_runs = rows("seed_sensitivity_flagged12_runs.csv")
    seed_counts = Counter((row["dataset"], row["event_id"]) for row in seed_runs)
    require(len(seed_runs) == 120, "seed audit must contain 120 raw runs")
    require(len(seed_counts) == 12 and set(seed_counts.values()) == {10},
            "seed audit must contain 12 events with 10 seeds each")
    pooled = rows("seed_sensitivity_flagged12_pooled_summary.csv")
    require(len(pooled) == 1 and pooled[0]["events_with_mixed_boundary_status"] == "0",
            "pooled seed audit must report no mixed boundary status")
    config = json.loads((RESULTS / "seed_sensitivity_flagged12_config.json").read_text(encoding="utf-8"))
    require(config, "seed-audit configuration must be readable")

    pilot = ROOT / "external_pilot"
    for relative in (
        "config/nlloc.conf",
        "config/ph2dt.inp",
        "config/hypoDD.inp",
        "results/nlloc_relocations.csv",
        "results/final_relocation_catalog.csv",
        "results/cluster_convergence.csv",
        "results/relocation_summary.csv",
        "results/input_audit.txt",
        "results/dt.ct",
    ):
        require((pilot / relative).is_file(), f"missing external pilot file: {relative}")

    dt_lines = (pilot / "results/dt.ct").read_text(encoding="utf-8").splitlines()
    dt_headers = [line for line in dt_lines if line.lstrip().startswith("#")]
    dt_rows = [line for line in dt_lines if line.strip() and not line.lstrip().startswith("#")]
    require(len(dt_lines) == 602, "dt.ct must contain 602 total lines")
    require(len(dt_headers) == 82, "dt.ct must contain 82 event-pair blocks")
    require(len(dt_rows) == 520, "dt.ct must contain 520 selected P differential times")

    require((ROOT / "HSSO-CG-Paper/paper/main.tex").is_file(), "missing manuscript source")
    require((ROOT / "HSSO-CG-Paper/paper/output/main.pdf").is_file(), "missing manuscript PDF")

    subprocess.run([sys.executable, str(ROOT / "tools/verify_manifest.py")], check=True)
    print("release content audit: PASS")
    print("paired comparisons: 6; QC events: 35; seed runs: 120; ph2dt rows: 520")


if __name__ == "__main__":
    main()
