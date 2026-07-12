#!/usr/bin/env python
"""Audit the manuscript-facing reproducibility claims using only stdlib."""
from __future__ import annotations

import csv
import json
import re
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

    submission = ROOT / "HSSO-CG-Paper/submission/Computers_and_Geosciences"
    required_submission = (
        "manuscript/main.tex",
        "manuscript/main.pdf",
        "manuscript/references.bib",
        "cover_letter.tex",
        "cover_letter.pdf",
        "highlights.txt",
        "data_statement.txt",
        "figures/Figure_1.pdf",
        "figures/Figure_2.pdf",
        "figures/Figure_3.pdf",
        "supplementary/Supplementary_Table_S1.csv",
        "supplementary/Supplementary_Material_Captions.txt",
        "README_SUBMISSION.md",
        "SUBMISSION_CHECKLIST.md",
    )
    for relative in required_submission:
        require((submission / relative).is_file(), f"missing submission file: {relative}")

    highlights = [line for line in (submission / "highlights.txt").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    require(3 <= len(highlights) <= 5, "submission must contain three to five highlights")
    require(all(len(line) <= 85 for line in highlights),
            "every highlight must contain at most 85 characters")

    submission_tex = (submission / "manuscript/main.tex").read_text(encoding="utf-8")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                               submission_tex, flags=re.DOTALL)
    require(abstract_match is not None, "submission manuscript must contain an abstract")
    abstract_text = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?\{([^}]*)\}", r"\1",
                           abstract_match.group(1))
    abstract_text = re.sub(r"\$[^$]*\$", " value ", abstract_text)
    abstract_words = re.findall(r"\b[\w'-]+\b", abstract_text)
    require(len(abstract_words) <= 300, "abstract must contain at most 300 words")
    require("\\doublespacing" in submission_tex and "\\linenumbers" in submission_tex,
            "submission manuscript must be double spaced with line numbers")
    require("authoryear" in submission_tex and "\\bibliographystyle{plainnat}" in submission_tex,
            "submission manuscript must use author-year references")
    first_author = "R\\'omulo Sandoval Fl\\'orez"
    second_author = "Claudia Patricia Parra Medina"
    require(first_author in submission_tex and second_author in submission_tex,
            "submission manuscript must contain both verified author names")
    require(submission_tex.index(first_author) < submission_tex.index(second_author),
            "Rómulo Sandoval Flórez must be listed before Claudia Patricia Parra Medina")
    require("romulo@ecci.edu.co" in submission_tex,
            "submission manuscript must identify the corresponding-author email")
    submission_bib = (submission / "manuscript/references.bib").read_text(encoding="utf-8")
    require("author={Parra Medina, Claudia Patricia}" in submission_bib,
            "companion-manuscript author name must match the submission author name")
    cover_tex = (submission / "cover_letter.tex").read_text(encoding="utf-8")
    require(first_author in cover_tex and "romulo@ecci.edu.co" in cover_tex,
            "cover letter must be signed by the corresponding author")
    citation_cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    require("Sandoval Flórez" in citation_cff and "Parra Medina" in citation_cff,
            "CITATION.cff must contain both manuscript authors")
    require(citation_cff.index("Sandoval Flórez") < citation_cff.index("Parra Medina"),
            "CITATION.cff author order must match the manuscript")

    subprocess.run([sys.executable, str(ROOT / "tools/verify_manifest.py")], check=True)
    print("release content audit: PASS")
    print("paired comparisons: 6; QC events: 35; seed runs: 120; ph2dt rows: 520")
    print(f"submission audit: {len(abstract_words)} abstract words; "
          f"{len(highlights)} highlights; 3 vector figures")


if __name__ == "__main__":
    main()
