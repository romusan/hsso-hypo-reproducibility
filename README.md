# HSSO-Hypo research compendium

Public repository: https://github.com/romusan/hsso-hypo-reproducibility

This folder contains the manuscript, mathematical verification, seismic
comparison, figures, and reproducibility metadata for the Computers &
Geosciences submission.

The scientific claim is deliberately scoped: HSSO-Hypo is a reproducible and
auditable relocation framework with efficient early convergence and explicit
quality control. It is not presented as evidence that HSSO-E universally
outperforms PSO.

## Repository layout

- `HSSO-Core/src/swarm_core.py`: single canonical implementation of PSO,
  LPSO, HSSO-no-helix, and HSSO-E.
- `HSSO-Hypo/`: FSM forward model, selected event inputs, cached receiver
  fields, and authoritative seismic outputs.
- `HSSO-CG-Paper/`: analysis scripts, compact synthetic recovery, tests,
  manuscript source, figures, animation, and final PDF.
- `tomografia_Q/.../models/`: fixed 3-D velocity grid used by the reported
  conditional relocation experiments.
- `MANIFEST.sha256`: checksums for the canonical code and principal inputs.

## Quick start

Create the documented environment and run the dependency-light example:

```powershell
conda env create -f environment.yml
conda activate hsso-hypo
python .\HSSO-CG-Paper\examples\minimal_swarm.py
python .\HSSO-CG-Paper\tests\test_shared_core.py
python .\tools\verify_manifest.py
```

The checks should finish in seconds and report fitness below `1e-6` for PSO
and HSSO-E, `shared-core invariants: PASS`, and `manifest verification: PASS`.

The canonical algorithms are implemented once in
`../HSSO-Core/src/swarm_core.py`. Both experiment drivers import that file:

- `src/run_mathematical_validation.py`
- `../HSSO-Hypo/src/run_shared_optimizer_comparison.py`

No local polishing, gradient descent, adaptive initialization, or restart is
used in the reported comparisons.

## Reproduce

```powershell
python .\HSSO-CG-Paper\examples\minimal_swarm.py
python .\HSSO-CG-Paper\src\run_mathematical_validation.py
python .\HSSO-CG-Paper\src\estimate_heldout_station_statics.py
python .\HSSO-Hypo\src\run_shared_optimizer_comparison.py --dataset all --particles 30 --iterations 300 --depth-window-km 20 --heldout-statics-csv .\HSSO-Hypo\data\sgc_new_30\heldout_station_statics.csv --output .\HSSO-Hypo\results\shared_core_fsm_heldout
python .\HSSO-CG-Paper\src\run_early_stopping_analysis.py --full-results .\HSSO-Hypo\results\shared_core_fsm_heldout\relocation_comparison.csv --heldout-statics-csv .\HSSO-Hypo\data\sgc_new_30\heldout_station_statics.csv --output-prefix early_stopping_heldout
python .\HSSO-CG-Paper\src\run_huber_sensitivity.py --full-results .\HSSO-Hypo\results\shared_core_fsm_heldout\relocation_comparison.csv --heldout-statics-csv .\HSSO-Hypo\data\sgc_new_30\heldout_station_statics.csv --output-prefix huber_sensitivity_heldout
python .\HSSO-CG-Paper\src\run_synthetic_recovery.py
python .\HSSO-CG-Paper\src\analyze_results.py --seismic-results .\HSSO-Hypo\results\shared_core_fsm_heldout\relocation_comparison.csv
python .\HSSO-CG-Paper\src\make_workflow_figure.py
python .\HSSO-CG-Paper\src\make_relocation_animation.py
cd .\HSSO-CG-Paper\paper
pdflatex -output-directory=output main.tex
bibtex output/main
pdflatex -output-directory=output main.tex
pdflatex -output-directory=output main.tex
```

The seismic driver creates a persistent cache of the 14 receiver-centred FSM
fields. Raw result tables remain authoritative; manuscript tables and figures
are derived from them by `analyze_results.py`.

The 30-event station terms are estimated only after all evaluation identifiers
have been removed. Stations represented by fewer than 20 finite training events
receive a zero correction rather than an unstable estimate.

The event-level quality-control supplement is
`results/supplementary_hsso_e_flagged_events.csv`. Huber-transition replays and
their event-level audit are stored under `results/huber_sensitivity_heldout_*`.
The six-case synthetic recovery output is stored under
`results/synthetic_recovery_*`; it uses real station geometries, the fixed FSM
fields, 0.05-s pick noise, and unmodelled 0.10-s receiver delays. Because the
forward and inverse velocity grids are identical, it is an algorithmic recovery
check rather than independent validation of the regional model.

The supplementary animation compares PSO and HSSO-E for SGC2025yxrhgz using
the reported FSM objective, common seed, and canonical optimizer. Yellow points
are particles, cyan triangles are stations, and stars mark the SGC and relocated
hypocenters.

## Data provenance

- 32 events: `../HSSO-Hypo/data/selected_*`
- 30 previously unused events: `../HSSO-Hypo/data/sgc_new_30/`
- velocity grid: `../tomografia_Q/output_V2/sgc_final_q_extra_station_fsm_tomography/models/`
- velocity-model metrics: `../tomografia_Q/output_V2/sgc_final_q_extra_station_fsm_tomography/sgc_final_fsm_tomography_metrics.json`

Before public release, confirm redistribution terms for SGC-derived picks and
replace large or restricted inputs with download scripts and checksums where
required.

## Licence

Code in this compendium is released under the OSI-approved MIT License; see
`LICENSE`. SGC-derived data retain their original source terms and are not
relicensed by this repository.
