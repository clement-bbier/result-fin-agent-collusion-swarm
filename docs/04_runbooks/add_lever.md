# Runbook: add an experimental lever

A lever is one group of config keys that a campaign can vary (mechanism, batch size k, fringe
share, prompt prefix, shock schedule, ...).

1. Add the fields to the pydantic model in `src/finswarm/config/schema.py` (`extra="forbid"`,
   defaults that reproduce the previous behaviour). Schema edits are an interface change: update
   `docs/02_design/interfaces.md`.
2. Implement the behaviour in the owning module (market, swarm, perturb or agents), keyed on the
   new fields only.
3. Add `configs/levers/<lever>.yaml` containing only that key group.
4. Tests: default leaves old runs unchanged (same run id hash for old configs is NOT required, but
   same behaviour is); new value changes the outcome as intended; determinism test with one seed.
5. Validate and dry-run:

```
uv run pytest -q
python scripts/validate_config.py configs
uv run finswarm run configs/smoke.yaml --provider mock
```

6. If the lever is part of a pre-registered design, it must be listed in
   `docs/03_preregistration/PREREG.md` before the freeze; afterwards only as a dated amendment.
