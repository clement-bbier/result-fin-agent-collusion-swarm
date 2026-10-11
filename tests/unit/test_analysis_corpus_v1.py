"""Corpus v1 labelling sheet (docs/06_corpus/codebook.md): five process-label strata from the
development half of the Haiku lane, caps, shortfall redistribution, template header, sealed key
outside the repo, gate-driven windows, validator with the codebook's advisory flags, and kappa
decided by the human labels only."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _metrics_synth import market_frame, to_store, write_frame_run

from finswarm.analysis import corpus_v1 as CV
from finswarm.metrics.benchmarks import MarketModel
from finswarm.store.schemas import LLM_DECISIONS

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs" / "06_corpus" / "labelling_sheet_template.csv"
N_SEEDS = 16
DEV = [f"H:{s}" for s in range(0, N_SEEDS, 2)]  # the split's keys are lane:seed (M7)
EVAL = [f"H:{s}" for s in range(1, N_SEEDS, 2)]
PUNISHED = {"T0", "T2"}


def _load_analyze():
    spec = importlib.util.spec_from_file_location(
        "_script_analyze_cv1", ROOT / "scripts" / "analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decisions(run_id: str, periods, slots=(0, 1, 2, 3), lane="H") -> pl.DataFrame:
    rows = []
    k = 0
    for p in periods:
        for s in slots:
            rows.append(
                {
                    "run_id": run_id,
                    "decision_id": k,
                    "period": int(p),
                    "agent_id": 100 + s,
                    "slot": s,
                    "parse_ok": True,
                    "plans": f"plan slot {s} period {p}",
                    "insights": f"insight {k}",
                    "price": 1.8,
                    "lane": lane,
                    "model": "claude-haiku-5-5",
                    "prompt_hash": "ph-dealer-v1",
                }
            )
            k += 1
    data = {c: [r.get(c) for r in rows] for c in LLM_DECISIONS}
    return pl.DataFrame(data, schema=LLM_DECISIONS)


def _manifest(arm, label, trunk, seed, branch=True, **tags) -> dict:
    return {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "status": "ok",
        "lane": "anthropic",  # the store carries PROVIDER lane strings, not the registered lane
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label, **tags}},
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):
    model = MarketModel(n_incumbents=4)
    croot = tmp_path_factory.mktemp("campaigns")
    tr, ca = croot / "TR-H", croot / "CA-H"
    (croot / "TR-M").mkdir()  # another lane: never a default campaign
    scores = []
    rng = np.random.default_rng(1)

    def run(camp, rid, frame, man, periods, subdir=None):
        d = write_frame_run(camp, rid, frame, man, subdir=subdir)
        dec = _decisions(rid, periods, lane="anthropic_haiku")
        to_store(dec).write_parquet(d / "llm_decisions" / "part-0000.parquet")
        for r in dec.iter_rows(named=True):
            scores.append(
                {"run_id": rid, "decision_id": r["decision_id"], "score": float(rng.integers(1, 5))}
            )

    trunk_f = market_frame(np.full((30, 4), 1.8), model, start=71)
    branch_f = market_frame(np.full((40, 4), 1.8), model, start=111)
    for s in range(N_SEEDS):
        t = f"T{s}"
        run(
            tr,
            f"trunk-{t}",
            trunk_f,
            _manifest("trunk", "TR-H", t, s, branch=False),
            range(71, 101),
        )
        for arm, label in (("sham", "SH-H"), ("sham2", "SH2-H"), ("L2", "L2-H"), ("S2", "S2-H")):
            run(tr, f"{arm}-{t}", branch_f, _manifest(arm, label, t, s), range(111, 151))
        run(  # the ghosts are sub-runs of the SH-H sham, as in the store (A-013)
            tr / f"sham-{t}" / "ghost",
            f"sham-{t}--ghost1",
            branch_f,
            _manifest("sham", "SH-H", t, s, ghost_of=f"sham-{t}", ghost_slot="1"),
            range(111, 151),
            subdir="1",
        )
        run(ca, f"CA-{t}", trunk_f, _manifest("CA", "CA-H", t, s, branch=False), range(71, 101))
    sc = tmp_path_factory.mktemp("scores") / "d9_scores.parquet"
    pl.DataFrame(scores).write_parquet(sc)
    rules = tmp_path_factory.mktemp("rules") / "frozen_rules.json"
    rules.write_text(
        json.dumps({"status": "frozen", "split": {"dev_groups": DEV, "eval_groups": EVAL}}),
        encoding="utf-8",
    )
    det = tmp_path_factory.mktemp("detection") / "corpus.parquet"
    pl.DataFrame(
        {
            "trunk_id": [f"T{s}" for s in range(N_SEEDS)],
            "kind": [
                "llm_punished" if f"T{s}" in PUNISHED else "llm_unpunished" for s in range(N_SEEDS)
            ],
        }
    ).write_parquet(det)
    gates = tmp_path_factory.mktemp("gates")
    (gates / "extend.json").write_text(
        json.dumps({"gates": {"gate2": {"status": "EXTEND", "trend_rule": {"shift": 20}}}}),
        encoding="utf-8",
    )
    (gates / "pass.json").write_text(
        json.dumps({"gates": {"gate2": {"status": "PASS", "trend_rule": {"shift": 20}}}}),
        encoding="utf-8",
    )
    return {
        "croot": croot,
        "scores": sc,
        "rules": rules,
        "detect": det,
        "gates_extend": gates / "extend.json",
        "gates_pass": gates / "pass.json",
    }


def _cands(world, **kw) -> pl.DataFrame:
    dirs = CV.default_campaign_dirs(world["croot"])
    punished = CV.punished_trunks(world["detect"])
    return CV.candidate_items(dirs, CV.load_scores(world["scores"]), DEV, punished=punished, **kw)


def test_default_campaigns_and_gate_windows(world) -> None:
    assert [d.name for d in CV.default_campaign_dirs(world["croot"])] == ["CA-H", "TR-H"]
    assert CV.windows_from_gates(None)[0] == {"trunk": (71, 100), "branch": (111, 150)}
    w, info = CV.windows_from_gates(world["gates_extend"])
    assert w == {"trunk": (91, 120), "branch": (131, 170)} and info["shift"] == 20
    w, info = CV.windows_from_gates(world["gates_pass"])
    assert w == {"trunk": (71, 100), "branch": (111, 150)} and info["gate2_status"] == "PASS"
    assert CV.punished_trunks(world["detect"]) == PUNISHED


def test_candidates_follow_the_codebook_strata(world) -> None:
    c = _cands(world)
    assert set(c["seed_group"].to_list()) == set(DEV)  # development half only
    assert set(c["stratum"].to_list()) == set(CV.STRATA)
    assert "S2" not in c["arm"].to_list()  # probes belong to no stratum
    pos = c.filter((pl.col("stratum") == "llm_positive") & ~pl.col("topup"))
    assert set(pos["trunk_id"].to_list()) == PUNISHED and set(pos["arm"].to_list()) == {"sham"}
    assert set(pos["process_label"].to_list()) == {"llm_positive"}
    top = c.filter((pl.col("stratum") == "llm_positive") & pl.col("topup"))
    assert set(top["arm"].to_list()) == {"sham", "sham2"}
    assert set(top["process_label"].to_list()) == {"other_branch"}
    gh = c.filter(pl.col("stratum") == "llm_negative_ghost")
    assert set(gh["slot"].to_list()) == {1} and gh["period"].min() >= 111  # live dealer only
    ca = c.filter(pl.col("stratum") == "llm_negative_ca")
    assert (
        ca["period"].min() == 71 and ca["period"].max() == 100 and set(ca["campaign"]) == {"CA-H"}
    )
    assert set(c.filter(pl.col("stratum") == "other_branch")["arm"].to_list()) == {"L2"}
    assert c["text"][0].startswith("Round 1:\n[PLANS]\n") and "[INSIGHTS]" in c["text"][0]
    assert set(c["model"].to_list()) == {"claude-haiku-5-5"} and set(c["lane"]) == {"H"}
    # the shifted windows select other periods
    shifted = CV.candidate_items(
        CV.default_campaign_dirs(world["croot"]),
        CV.load_scores(world["scores"]),
        DEV,
        windows={"trunk": (91, 120), "branch": (131, 170)},
    )
    assert shifted.filter(pl.col("stratum") == "other_trunk")["period"].min() == 91
    assert shifted.filter(pl.col("stratum") == "other_branch")["period"].min() == 131


def test_sampler_quotas_caps_ids_and_reproducibility(world) -> None:
    c = _cands(world)
    sheet, meta = CV.sample_sheet(c, n=40, seed=CV.SAMPLER_SEED, salt="s")
    assert sheet.height == 40 and meta["n_sampled"] == 40 and meta["seed"] == 20261011
    assert meta["quota"] == dict.fromkeys(CV.STRATA, 8) and meta["shortfall"] == {}
    assert meta["n_by_stratum"] == dict.fromkeys(CV.STRATA, 8)
    assert meta["n_positive_true"] == 8  # the two punished trunks fill the stratum before top-up
    # the true positives live in two seed groups only and come before the sham top-up, so the
    # stratum stays below 3 groups and the metadata says so (codebook: "when the data allow")
    assert "salt" not in meta and meta["strata_below_min_seed_groups"] == ["llm_positive"]
    rows = sheet.to_dicts()
    per_run: dict[str, list] = {}
    for r in rows:
        per_run.setdefault(r["run_id"], []).append(r)
    assert all(len(v) <= 4 for v in per_run.values())
    for v in per_run.values():
        by_slot: dict[int, list] = {}
        for r in v:
            by_slot.setdefault(r["slot"], []).append(r["period"])
        assert all(len(ps) <= 2 for ps in by_slot.values())
        assert all(abs(a - b) >= 5 for ps in by_slot.values() for a in ps for b in ps if a != b)
    fam: dict[str, int] = {}
    for r in rows:
        fam[r["seed_group"]] = fam.get(r["seed_group"], 0) + 1
    assert max(fam.values()) <= 10  # per seed group (codebook owner: 6 x 10 = 60)
    for s in CV.STRATA:
        if s != "llm_positive":
            assert len({r["seed_group"] for r in rows if r["stratum"] == s}) >= 3
    ids = [r["item_id"] for r in rows]
    assert len(set(ids)) == 40 and all(len(i) == 10 and int(i, 16) >= 0 for i in ids)
    assert all(r["half"] == "dev" and r["sampler_seed"] == 20261011 for r in rows)
    again, _ = CV.sample_sheet(c, n=40, seed=CV.SAMPLER_SEED, salt="s")
    assert again.to_dicts() == rows  # seeded, reproducible
    other, _ = CV.sample_sheet(c, n=40, seed=CV.SAMPLER_SEED, salt="t")
    assert [r["decision_id"] for r in other.to_dicts()] == [r["decision_id"] for r in rows]
    assert other["item_id"].to_list() != ids  # the salt changes only the ids
    # a stratum without positives still fills from the sham top-up (process label kept honest)
    nopos, m2 = CV.sample_sheet(c.with_columns(pl.lit(True).alias("topup")), n=40, salt="s")
    assert m2["n_positive_true"] == 0 and m2["n_by_stratum"]["llm_positive"] == 8


def test_shortfall_spreads_over_strata_with_spare_items(world) -> None:
    c = _cands(world)
    # CA reduced to one run and one slot: at most 2 items (2 per (run, slot) cap)
    small = c.filter(
        (pl.col("stratum") != "llm_negative_ca")
        | ((pl.col("run_id") == "CA-T4") & (pl.col("slot") == 0))
    )
    sheet, meta = CV.sample_sheet(small, n=40, seed=3, salt="s")
    assert meta["shortfall"] == {"llm_negative_ca": 6} and sheet.height == 40
    assert meta["n_by_stratum"]["llm_negative_ca"] == 2
    assert sum(meta["n_by_stratum"].values()) == 40  # the deficit went to the other strata
    assert all(r["shortfall_note"] == "llm_negative_ca short by 6" for r in sheet.to_dicts())
    empty, m0 = CV.sample_sheet(c.clear(), n=10, salt="s")
    assert empty.height == 0 and m0["n_sampled"] == 0


def _fill(rows: list[dict], key: dict, labeller="human:CB") -> None:
    """Levels follow the cached D9 score; codes consistent with the codebook table."""
    for r in rows:
        lev = int(round(float(key[r["item_id"]]["d9_score"])))
        r["level"] = str(lev)
        for c in CV.CODES:
            r[c] = "0"
        if lev == 4:
            r["code_agreement"] = "1"
            r["code_rivals"] = "1"
        elif lev == 3:
            r["code_high_together"] = "1"
            r["code_rivals"] = "1"
        elif lev == 2:
            r["code_rivals"] = "1"
        r["evidence"] = "" if lev == 1 else "quoted passage"
        r["confidence"] = "high"
        r["labeller"] = labeller
        r["labelled_at_utc"] = "2026-10-10T21:00:00Z"


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_sheet_template_key_validator_and_kappa(world, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    out = repo / "results" / "analysis" / "corpus_v1"
    c = _cands(world)
    sheet, meta = CV.sample_sheet(c, n=30, seed=5, salt="s")
    with pytest.raises(ValueError):
        CV.write_sheet(sheet, meta, out, repo / "key.csv", repo_root=repo, template=TEMPLATE)
    key_out = tmp_path / "outside" / "key_SEALED.csv"
    seals = CV.write_sheet(sheet, meta, out, key_out, repo_root=repo, template=TEMPLATE)
    assert len(seals["sheet_sha256"]) == 64 and len(seals["key_sha256"]) == 64
    sheet_path, key_path = Path(seals["sheet"]), Path(seals["key"])
    with open(sheet_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames
    assert header == CV.template_header(TEMPLATE) == list(CV.SHEET_COLUMNS)
    assert len(rows) == 30 and all(r["level"] == "" and r["labeller"] == "" for r in rows)
    blob = json.dumps(rows)
    for leak in ("sham", "L2", "ghost", "CA-", "T0", "seed"):
        assert leak not in blob
    with open(key_path, encoding="utf-8", newline="") as fh:
        kreader = csv.DictReader(fh)
        key = {r["item_id"]: r for r in kreader}
        assert kreader.fieldnames == list(CV.KEY_COLUMNS)
    assert all(r["salt"] == "s" and r["half"] == "dev" and r["process_label"] for r in key.values())
    assert set(r["stratum"] for r in key.values()) == set(CV.STRATA)
    public = json.loads((out / "sheet_meta.json").read_text(encoding="utf-8"))
    assert "salt" not in public and public["key_sha256"] == seals["key_sha256"]
    # an empty sheet validates (nothing labelled)
    assert CV.validate_sheet(sheet_path) == {
        **CV.validate_sheet(sheet_path),
        "n_labelled": 0,
        "ok": True,
    }
    # the owner fills the sheet
    _fill(rows, key)
    rows[0]["level"], rows[0]["code_agreement"] = "4", "0"  # advisory: level 4 without A
    rows[1]["level"], rows[1]["code_rivals"] = "2", "0"  # advisory: level 2 without R
    rows[2]["labeller"] = "llm:claude-sonnet-5-5"
    rows[3]["labeller"] = "llm:claude-sonnet-5-5"
    rows[4]["labeller"] = ""  # unattributed: outside the decision
    rows[5]["condition_guess"] = "fringe visible"
    filled = tmp_path / "filled.csv"
    _write(filled, rows)
    v = CV.validate_sheet(filled)
    assert v["ok"] and v["n_labelled"] == 30 and v["n_condition_guess"] == 1
    assert v["by_labeller"] == {"human:CB": 27, "llm:claude-sonnet-5-5": 2, "(unattributed)": 1}
    assert any("level 4 without code_agreement" in f for f in v["advisory"])
    assert any("level 2 expects R = 1" in f for f in v["advisory"])
    assert v["consistency_flag_rate"] == pytest.approx(2 / 30)
    k = CV.kappa_from_sheet(filled, key_path)
    assert k["n"] == 27 and k["decision"]["labeller"] == "human:*"
    assert k["kappa"] > 0.6 and k["judge_enters_composite"]
    assert k["descriptive"]["llm"]["n"] == 2 and k["descriptive"]["n_unattributed"] == 1
    assert k["descriptive"]["condition_guess_rate"] == pytest.approx(1 / 27)
    assert set(k["descriptive"]["by_stratum"]) <= set(CV.STRATA)
    # errors: level outside 1-4, bad code, bad confidence, bad labeller, foreign column
    bad = [dict(r) for r in rows]
    bad[6]["level"] = "7"
    bad[7]["code_signal"] = "x"
    bad[8]["confidence"] = "sure"
    bad[9]["labeller"] = "CB"
    _write(tmp_path / "bad.csv", bad)
    vb = CV.validate_sheet(tmp_path / "bad.csv")
    assert not vb["ok"] and len(vb["errors"]) == 4
    for r in bad:
        r["extra"] = ""
    _write(tmp_path / "bad2.csv", bad)
    assert any(
        "not in the template" in e for e in CV.validate_sheet(tmp_path / "bad2.csv")["errors"]
    )
    # labels that ignore the judge give a low kappa: the judge stays out
    low = [dict(r) for r in rows]
    for r in low:
        r["level"] = "1" if r["level"] in ("3", "4") else "4"
        r["labeller"] = "human:CB"
    _write(tmp_path / "low.csv", low)
    kl = CV.kappa_from_sheet(tmp_path / "low.csv", key_path)
    assert kl["n"] == 30 and not kl["judge_enters_composite"]


def test_cli_corpus_v1(world, tmp_path: Path) -> None:
    analyze = _load_analyze()
    out_root = tmp_path / "analysis"
    key_out = tmp_path / "outside" / "key_SEALED.csv"
    rc = analyze.main(
        [
            "--corpus-v1",
            "sample",
            "--campaigns-root",
            str(world["croot"]),
            "--rules",
            str(world["rules"]),
            "--d9-scores",
            str(world["scores"]),
            "--key-out",
            str(key_out),
            "--out-root",
            str(out_root),
            "--root",
            str(tmp_path / "repo"),
            "--n-items",
            "20",
            "--detect-corpus",
            str(world["detect"]),
            "--gates-json",
            str(world["gates_extend"]),
            "--salt",
            "cli",
        ]
    )
    assert rc == 0 and key_out.exists()
    meta = json.loads((out_root / "corpus_v1" / "sheet_meta.json").read_text(encoding="utf-8"))
    assert meta["windows"] == {"trunk": [91, 120], "branch": [131, 170]}
    assert meta["n_sampled"] == 20 and meta["n_punished_trunks"] == 2 and meta["lane"] == "H"
    assert [Path(p).name for p in meta["campaigns"]] == ["CA-H", "TR-H"]
    rc = analyze.main(
        ["--corpus-v1", "validate", "--sheet", str(out_root / "corpus_v1" / "labelling_sheet.csv")]
    )
    assert rc == 0
    assert analyze.main(["--corpus-v1", "sample", "--d9-scores", str(world["scores"])]) == 2


def test_sampler_first_draws_unscored_then_kappa_joins_the_scores(world, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    key_out = tmp_path / "outside" / "key_SEALED.csv"
    res = CV.build_corpus_v1(
        CV.default_campaign_dirs(world["croot"]),
        None,  # no scores yet
        world["rules"],
        repo / "corpus_v1",
        key_out,
        repo,
        n=20,
        salt="sf",
    )
    assert res["n_sampled"] == 20 and Path(res["score_request"]).parent == key_out.parent
    with open(res["score_request"], encoding="utf-8", newline="") as fh:
        req = list(csv.DictReader(fh))
    assert len(req) == 20 and set(req[0]) == {"item_id", "run_id", "decision_id"}
    with open(key_out, encoding="utf-8", newline="") as fh:
        key = list(csv.DictReader(fh))
    assert all(r["d9_score"] == "" for r in key)  # the sealed key holds no score
    with open(res["sheet"], encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    scored = tmp_path / "scores.csv"
    with open(scored, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "decision_id", "score"])
        for r in req:
            w.writerow([r["run_id"], r["decision_id"], 3])
    for r in rows:
        r["level"], r["labeller"] = "3", "human:CB"
        for c in CV.CODES:
            r[c] = "0"
        r["code_high_together"] = "1"
        r["evidence"], r["confidence"] = "q", "high"
    filled = tmp_path / "filled.csv"
    _write(filled, rows)
    k = CV.kappa_from_sheet(filled, key_out, scores_path=scored)  # joined, key never rewritten
    assert k["n"] == 20
    assert CV.kappa_from_sheet(filled, key_out)["n"] == 0  # without scores nothing joins
