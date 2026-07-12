# Release-content audit

This audit maps manuscript-facing reproducibility claims to public artifacts.
It is executable with the Python standard library:

```powershell
python .\tools\audit_release.py
```

The check requires and verifies:

- six paired seismic comparisons with raw and Holm-adjusted p values and a
  matched-pairs rank-biserial effect size;
- 35 unique HSSO-E quality-control events (15 stratified and 20 held out);
- 120 raw repeated-seed runs covering 12 pre-flagged events with 10 seeds each,
  their configuration, event summaries, and pooled summary;
- the compact NonLinLoc/ph2dt/HypoDD pilot configurations and summary outputs;
- manuscript LaTeX source and compiled PDF; and
- every SHA-256 entry in `MANIFEST.sha256`.

The external-method pilot is archived as a sensitivity diagnostic. It is not a
matched benchmark against HSSO-Hypo because its velocity model, search window,
residual formulation, and differential-time inputs differ.
