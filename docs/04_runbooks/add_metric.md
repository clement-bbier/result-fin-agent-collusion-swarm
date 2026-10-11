# Runbook: add a metric

1. Create the function in `src/finswarm/metrics/<name>.py`, pure and vectorised (polars/numpy):

```python
from finswarm.metrics.registry import register_metric

@register_metric(name="markup_index", unit="ratio", level="run", higher_is_collusive=True,
                 source="Calvano et al. 2020")
def markup_index(...): ...
```

2. Add `tests/unit/test_<name>.py` with a synthetic case whose answer is known (competitive
   outcome gives 0, monopoly outcome gives 1, noise gives about 0).
3. Make it relative to the campaign benchmark, not absolute.
4. Register it in the analysis plan; a metric not in `PREREG.md` is exploratory.
5. Check:

```
uv run pytest -q tests/unit/test_<name>.py
uv run ruff check . && uv run ruff format .
```

6. If the registry signature or the table schemas must change, that is an interface change (see
   `docs/02_design/interfaces.md`).
