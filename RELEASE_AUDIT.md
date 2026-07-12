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
- the compact NonLinLoc/ph2dt/HypoDD pilot configurations and summary outputs,
  including `dt.ct` with 520 selected P differential-time rows in 82 event-pair
  blocks (602 total lines);
- manuscript LaTeX source and compiled PDF; and
- the Computers & Geosciences submission package, including a double-spaced
  line-numbered manuscript, author-year references, a sub-300-word abstract,
  five compliant highlights, cover letter, three vector figures, data statement,
  and cited supplementary table; and
- every SHA-256 entry in `MANIFEST.sha256`.

The external-method pilot is archived as a sensitivity diagnostic. It is not a
matched benchmark against HSSO-Hypo because its velocity model, search window,
residual formulation, and differential-time inputs differ.
