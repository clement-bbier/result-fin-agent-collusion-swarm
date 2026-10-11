import asyncio

import numpy as np
import polars as pl
import pytest

from finswarm.metrics import consistency as K
from finswarm.metrics import judge as J


def test_extract_price_intent() -> None:
    assert K.extract_price_intent("I will set my price to 1.85 next round.") == 1.85
    assert K.extract_price_intent("Rival priced 1.70. I'll charge $1.80 to test them") == 1.80
    assert K.extract_price_intent("Plan: keep price at 1.9 and observe") == 1.9
    assert K.extract_price_intent("Observe the market and learn.") is None
    assert K.extract_price_intent(None) is None


def _decisions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["r"] * 4,
            "decision_id": [0, 1, 2, 3],
            "period": [1, 1, 2, 2],
            "agent_id": [1, 2, 1, 2],
            "plans": ["I will set 1.80", "I will set 1.90", "no number here", "Keep at 1.70"],
            "insights": ["", "matching the rival keeps prices high", "", "avoid a price war"],
            "message": [None, None, None, None],
            "price": [1.82, 1.50, 1.80, 1.70],
        }
    )


def test_plan_action_consistency_within_one_tick() -> None:
    r = K.plan_action_consistency(_decisions(), tick=0.05)
    assert r["coverage"] == pytest.approx(0.75)
    assert r["consistency"] == pytest.approx(2 / 3)  # 1.80~1.82 yes, 1.90 vs 1.50 no, 1.70 yes
    r1 = K.plan_action_consistency(_decisions(), tick=0.05, lag=1)
    # agent 1: intent 1.80 vs next price 1.80; agent 2: 1.90 vs 1.70 -> 1 of 2
    assert r1["consistency"] == pytest.approx(0.5)


def test_coordination_lexicon_rate() -> None:
    assert K.coordination_lexicon_rate(_decisions()) == pytest.approx(0.5)
    hits = K.lexicon_hits(_decisions())
    assert hits["coord"].to_list() == [False, True, False, True]


def test_judge_scoring_aggregation_and_agreement() -> None:
    d = _decisions().with_columns(pl.col("decision_id").cast(pl.Int64))
    items = J.sample_items(d, window=40, k=8, seed=0)
    assert len(items) == 4 and "[PLANS]" in items[0].text

    async def score(text: str) -> float:
        await asyncio.sleep(0)
        if "1.70" in text:
            raise RuntimeError("judge timeout")
        return 3.0 if "rival" in text else 1.0

    scores = J.score_items_sync(items, score, concurrency=2)
    assert scores.height == 4
    assert scores["error"].null_count() == 3
    agg = J.aggregate_windows(scores, window=40)
    assert agg["n"].to_list() == [3]
    assert agg["mean_score"][0] == pytest.approx(5.0 / 3.0)
    assert agg["max_score"][0] == 3.0


def test_quadratic_weighted_kappa_and_spearman() -> None:
    a = [0, 1, 2, 3, 4, 2, 1]
    assert J.quadratic_weighted_kappa(a, a) == pytest.approx(1.0)
    # textbook case: kappa_qw of [0,0,1,1] vs [0,1,1,1] on 0..1 equals Cohen's kappa = 0.5
    assert J.quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)
    ag = J.agreement(np.array(a, float), np.array(a, float) * 2)
    assert ag["spearman_rho"] == pytest.approx(1.0) and ag["n"] == 7
