import polars as pl
import pytest

from finswarm.metrics import registry
from finswarm.store.schemas import TABLES


def test_register_metric_records_provenance() -> None:
    name = "_test_metric_unique"

    @registry.register_metric(
        name, unit="ratio", level="run", higher_is_collusive=True, source="calvano2020"
    )
    def fn(df):  # noqa: ANN001
        """Doc."""
        return 1.0

    spec = registry.get_metric(name)
    assert spec.source == "calvano2020"
    assert spec.description == "Doc."
    with pytest.raises(ValueError):
        registry.register_metric(name, "ratio", "run", True)(fn)
    del registry.REGISTRY[name]


def test_every_table_has_run_id_first() -> None:
    assert set(TABLES) == {
        "orders",
        "trades",
        "flow_per_tick",
        "quotes",
        "book_snapshots",
        "llm_decisions",
        "events",
        "agent_period",
        "metrics_per_period",
    }
    for schema in TABLES.values():
        assert next(iter(schema)) == "run_id"
        assert schema["run_id"] == pl.Utf8
        pl.DataFrame(schema={k: v for k, v in schema.items()})  # constructible empty frame
