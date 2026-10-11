# Runbook: produce figures

```
uv run python scripts/analyze.py <campaign>          # stats.json per claim id, figures
uv run python scripts/verify.py <campaign>           # recompute and compare
python scripts/check_claims.py                       # figures exist, run ids have manifests
```

Rules:

- Figures are produced only by `src/finswarm/analysis/figures.py` (Agg backend), saved to
  `paper/figures/` (PNG or PDF) with a sidecar JSON: inputs, run ids, code git sha, seed count.
- Maximum 4 main figures; others go to the appendix.
- Each figure is cited in `paper/claims.md` with its claim id; status comes from `stats.json`.
- Seeds are the unit; show intervals (bootstrap 10 000), not only means; label exploratory panels.
- Colour-blind safe palette, readable at column width, axis units from the metric registry.
- Final copies: `results/figures/final/` (tracked). Never commit raw parquet.
