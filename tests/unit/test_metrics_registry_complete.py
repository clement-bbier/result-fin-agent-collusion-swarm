import finswarm.metrics  # noqa: F401  (registers every metric)
from finswarm.metrics.registry import REGISTRY

EXPECTED = {
    # collusion
    "delta",
    "delta_price",
    "markup",
    "profit_share",
    "undercut_rate",
    "price_dispersion",
    "focal_clustering",
    "coalition_advantage",
    "impulse_response",
    "sting_followers",
    "non_best_response_gap",
    # quality
    "quoted_spread",
    "effective_spread",
    "realised_spread",
    "depth",
    "kyle_lambda",
    "realised_volatility",
    "pricing_error",
    "volume",
    "gode_sunder_efficiency",
    "buyer_surplus",
    "seller_surplus",
    "total_surplus",
    "mqc",
    # dependence
    "lagged_xcorr",
    "granger_pairwise",
    "transfer_entropy",
    "net_influence",
    "net_of_ghost",
    # dynamics
    "time_to_collude",
    "time_to_break",
    "time_to_recover",
    "hysteresis",
    "mttr",
    # text, judge, detection
    "plan_action_consistency",
    "coordination_lexicon_rate",
    "judge_score",
    "detector_composite",
}


def test_every_metric_is_registered_with_provenance() -> None:
    names = {n for n in REGISTRY if not n.startswith("_")}
    assert EXPECTED <= names
    for name in EXPECTED:
        spec = REGISTRY[name]
        assert spec.level in {"run", "agent", "pair", "window"}, name
        assert spec.unit and spec.source and spec.description, name
        assert spec.higher_is_collusive in (True, False, None), name
    assert REGISTRY["delta"].source == "calvano2020"
    assert REGISTRY["net_influence"].source == "bonjour2022"
    assert REGISTRY["focal_clustering"].source == "christie_schultz1994"
    assert REGISTRY["coalition_advantage"].source == "nakamura2026"
    assert REGISTRY["gode_sunder_efficiency"].source == "gode_sunder1993"
    assert REGISTRY["non_best_response_gap"].source == "novel"
