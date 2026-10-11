import polars as pl
import pytest

from finswarm.store.reader import SchemaMismatchError, load_campaign, load_run, table_parts
from finswarm.store.schemas import TABLES
from finswarm.store.writer import EpochBuffers, PartWriter, frame_from_rows


def event_row(i: int, run_id: str = "r") -> dict:
    return {
        "run_id": run_id,
        "event_id": i,
        "tick": i,
        "period": i,
        "event_type": "set_param",
        "payload": "{}",
        "source": "config",
    }


def test_frame_from_rows_uses_frozen_dtypes() -> None:
    df = frame_from_rows("quotes", [{"run_id": "r", "period": 1, "slot": 2, "is_llm": 1, "ask": 2}])
    assert dict(df.schema) == TABLES["quotes"]
    assert df["bid"][0] is None and df["ask"][0] == 2.0
    with pytest.raises(KeyError):
        frame_from_rows("quotes", [{"run_id": "r", "bogus": 1}])


def test_parts_written_in_order_and_restore_truncates(tmp_path) -> None:
    w, buf = PartWriter(tmp_path), EpochBuffers()
    for k in range(3):
        buf.add("events", event_row(k))
        buf.add_prompt({"decision_id": k, "user": "x"})
        w.flush(buf)
        if k == 0:
            st = w.state()
    assert [p.name for p in table_parts(tmp_path, "events")] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert load_run(tmp_path).frame("events")["event_id"].to_list() == [0, 1, 2]
    assert len((tmp_path / "prompts.jsonl").read_text().splitlines()) == 3
    w2 = PartWriter(tmp_path)
    w2.restore(st)  # back to the state after the first flush
    assert [p.name for p in table_parts(tmp_path, "events")] == ["part-00000.parquet"]
    assert len((tmp_path / "prompts.jsonl").read_text().splitlines()) == 1
    assert w2.next_index["events"] == 1


def test_no_partial_file_left_after_failed_write(tmp_path, monkeypatch) -> None:
    w, buf = PartWriter(tmp_path), EpochBuffers()
    buf.add("events", event_row(0))

    def boom(self, file, *a, **k):
        with open(file, "wb") as fh:
            fh.write(b"PAR1 half a file")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(OSError):
        w.flush(buf)
    assert list((tmp_path / "events").iterdir()) == []
    assert w.next_index["events"] == 0
    monkeypatch.undo()
    w.flush(buf)
    assert load_run(tmp_path).frame("events").height == 1


def test_reader_rejects_wrong_dtypes_and_empty_tables_have_schema(tmp_path) -> None:
    data = load_run(tmp_path)
    for t, lf in data.tables.items():
        assert dict(lf.collect_schema()) == TABLES[t]
    bad = frame_from_rows("events", [event_row(0)]).with_columns(pl.col("period").cast(pl.Int64))
    (tmp_path / "events").mkdir()
    bad.write_parquet(tmp_path / "events" / "part-00000.parquet")
    with pytest.raises(SchemaMismatchError):
        load_run(tmp_path)


def test_load_campaign_concatenates_runs(tmp_path) -> None:
    for name in ("a", "b"):
        rd = tmp_path / name
        w, buf = PartWriter(rd), EpochBuffers()
        buf.add("events", event_row(1, run_id=name))
        w.flush(buf)
        (rd / "manifest.json").write_text(f'{{"run_id": "{name}", "status": "ok", "seed": 0}}')
    c = load_campaign(tmp_path)
    assert sorted(c.tables["events"].collect()["run_id"].to_list()) == ["a", "b"]
    assert c.manifests.height == 2 and set(c.manifests["status"]) == {"ok"}
