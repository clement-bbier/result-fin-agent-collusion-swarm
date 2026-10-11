import numpy as np
import polars as pl
import pytest
from _metrics_synth import agent_period

from finswarm.metrics import quality as Q
from finswarm.metrics.benchmarks import LogitParams, benchmarks_from_config, logit_closed_form


def _quotes() -> pl.DataFrame:
    rows = [
        # tick, agent, bid, ask, bid_size, ask_size
        (0, 1, 0.99, 1.01, 1.0, 1.0),
        (0, 2, 0.98, 1.02, 2.0, 2.0),
        (5, 2, 0.99, 1.01, 2.0, 2.0),
        (10, 1, 0.995, -1.0, 1.0, 0.0),  # agent 1 withdraws its ask
    ]
    return pl.DataFrame(
        rows, schema=["tick", "agent_id", "bid", "ask", "bid_size", "ask_size"], orient="row"
    ).with_columns(pl.lit("r").alias("run_id"))


def _trades(**cols) -> pl.DataFrame:
    return pl.DataFrame(cols).with_columns(pl.lit("r").alias("run_id"))


def test_inside_quotes_spread_depth() -> None:
    ins = Q.inside_quotes(_quotes())
    assert ins["tick"].to_list() == [0, 5, 10]
    assert ins["spread"].to_list() == pytest.approx([0.02, 0.02, 0.015])
    assert ins["bid_depth"].to_list() == pytest.approx([1.0, 3.0, 1.0])
    assert ins["ask_depth"].to_list() == pytest.approx([1.0, 3.0, 2.0])
    assert Q.quoted_spread(_quotes()) == pytest.approx(0.055 / 3)
    assert Q.depth(_quotes()) == pytest.approx(11.0 / 3)


def test_effective_and_realised_spread() -> None:
    ins = Q.inside_quotes(_quotes())
    tr = _trades(tick=[6, 11], price=[1.01, 0.995], size=[1.0, 3.0], aggressor=[1, -1])
    assert Q.effective_spread(tr, ins) == pytest.approx((0.02 * 1 + 0.015 * 3) / 4)
    assert Q.realised_spread(tr, ins, horizon=5) == pytest.approx(0.015)


def test_inside_from_book() -> None:
    book = pl.DataFrame(
        {
            "tick": [1, 1, 1],
            "side": [1, -1, 1],
            "level": [0, 0, 1],
            "price": [9.9, 10.1, 9.8],
            "depth": [5.0, 4.0, 7.0],
        }
    )
    ins = Q.inside_from_book(book)
    assert ins["mid"].to_list() == pytest.approx([10.0])
    assert ins["spread"].to_list() == pytest.approx([0.2])


def test_kyle_lambda_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    sv = rng.normal(0, 2, 60)
    mid = 100 + np.cumsum(0.5 * sv)
    trades = _trades(period=np.arange(60), size=np.abs(sv), aggressor=np.sign(sv).astype(int))
    mids = pl.DataFrame({"period": np.arange(60), "mid_mean": mid})
    assert Q.kyle_lambda(trades, mids) == pytest.approx(0.5, abs=1e-9)
    lam, se, r2 = Q.kyle_lambda_arrays(0.5 * sv + rng.normal(0, 0.1, 60), sv)
    assert lam == pytest.approx(0.5, abs=0.05) and r2 > 0.9 and se > 0


def test_volatility_pricing_error_volume() -> None:
    assert Q.realised_volatility(np.array([1.0, np.exp(0.1), 1.0])) == pytest.approx(0.1)
    mpp = pl.DataFrame({"period": [1, 2], "mid_mean": [101.0, 99.0], "fundamental": [100.0] * 2})
    assert Q.pricing_error(mpp) == pytest.approx(0.01)
    assert Q.pricing_error(mpp, relative=False) == pytest.approx(1.0)
    tr = _trades(period=[1, 2], size=[1.0, 3.0])
    assert Q.volume(tr) == pytest.approx(2.0)
    assert Q.volume(tr, n_periods=4) == pytest.approx(1.0)


def test_trades_mode_surplus_and_gode_sunder_efficiency() -> None:
    comp = _trades(
        price=[5.0] * 3, size=[1.0] * 3, buyer_value=[10.0, 9.0, 8.0], seller_cost=[1.0, 2.0, 3.0]
    )
    assert Q.buyer_surplus(comp) == pytest.approx(12.0)
    assert Q.seller_surplus(comp) == pytest.approx(9.0)
    assert Q.total_surplus(comp, mode="trades") == pytest.approx(21.0)
    # the competitive allocation is fully efficient
    assert Q.gode_sunder_efficiency(comp, mode="trades") == pytest.approx(1.0)
    # an extra trade between a low-value buyer and a high-cost seller destroys surplus
    bad = pl.concat([comp, _trades(price=[9.0], size=[1.0], buyer_value=[7.0], seller_cost=[11.0])])
    assert Q.gode_sunder_efficiency(bad, mode="trades") == pytest.approx(17.0 / 21.0)
    # counting a buyer who never traded (value 12) against the full population
    eff = Q.gode_sunder_efficiency(
        comp,
        mode="trades",
        buyer_values=np.array([12.0, 10, 9, 8]),
        seller_costs=np.array([1.0, 2, 3, 4]),
    )
    assert eff == pytest.approx(21.0 / (11 + 8 + 6 + 4))
    # unknown values (NaN) and the closed-form aggregate account (buyer_id = -1) are ignored
    nan = pl.concat(
        [comp, _trades(price=[5.0], size=[1.0], buyer_value=[float("nan")], seller_cost=[1.0])]
    )
    assert Q.total_surplus(nan, mode="trades") == pytest.approx(21.0)
    agg = comp.with_columns(pl.Series("buyer_id", [3, 4, -1]))
    assert Q.total_surplus(agg, mode="trades") == pytest.approx(16.0)
    assert Q.mqc_pp(100.0, 95.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        Q.total_surplus(comp, mode="nope")


def _ts_closed(p: float, n: int = 2, a: float = 2.0, mu: float = 0.25, c: float = 1.0) -> float:
    """Independent closed form (alpha = beta = 1, a0 = 0): inclusive value + profits."""
    e = np.exp((a - p) / mu)
    return mu * np.log(1 + n * e) + n * (p - c) * e / (1 + n * e)


def test_logit_inclusive_surplus_efficiency_and_mqc() -> None:
    b = logit_closed_form()
    at_cost = agent_period(np.full((5, 2), 1.0))
    nash = agent_period(np.full((5, 2), b.p_nash))
    mono = agent_period(np.full((5, 2), b.p_mono))
    assert Q.gode_sunder_efficiency(at_cost) == pytest.approx(1.0)
    eff_n = Q.gode_sunder_efficiency(nash)
    eff_m = Q.gode_sunder_efficiency(mono)
    assert eff_n == pytest.approx(_ts_closed(b.p_nash) / _ts_closed(1.0), rel=1e-9)
    assert 0 < eff_m < eff_n < 1
    s = Q.logit_surplus(nash)
    assert s["ps"] == pytest.approx(np.full(5, 2 * b.pi_nash), rel=1e-9)
    assert (s["cs"] > 0).all()
    assert Q.total_surplus(nash) == pytest.approx(_ts_closed(b.p_nash), rel=1e-9)
    # the logit inclusive value is a separate metric, not the MQC axis
    with pytest.raises(ValueError):
        Q.market_quality_cost(nash, mono, mode="logit_inclusive")
    # a non-quoting dealer (NaN) leaves a one-dealer market; heterogeneous values average
    one = np.array([[b.p_nash, np.nan]])
    assert Q.total_surplus(one) == pytest.approx(_ts_closed(b.p_nash, n=1), rel=1e-9)
    same = Q.logit_surplus(nash, values=np.full((5, 3), 2.0))["ts"]
    assert same == pytest.approx(s["ts"])


def test_true_value_surplus_from_trades_one_unit_for_every_lever() -> None:
    # 2 dealer trades (cost 1) and 1 fringe trade (seller cost 1.5), periods 1-2, window 1-3
    tr = pl.DataFrame(
        {
            "period": [1, 1, 2],
            "price": [1.8, 1.9, 1.7],
            "size": [1.0, 2.0, 1.0],
            "buyer_value": [2.2, 2.0, 2.1],
            "seller_cost": [1.0, 1.0, 1.5],
            "buyer_id": [3, 4, 5],
        }
    )
    per, ts = Q.true_value_surplus_series(tr, start=1, end=3)
    # buyer (value - price) + seller (price - cost) = value - cost per unit
    assert per.tolist() == [1, 2, 3]
    assert ts.tolist() == pytest.approx([1.2 + 2 * 1.0, 0.6, 0.0])
    assert Q.true_value_surplus(tr, 1, 3) == pytest.approx((3.2 + 0.6) / 3)
    lever = tr.with_columns(pl.Series("size", [1.0, 1.0, 1.0]))
    mqc = Q.market_quality_cost(tr, lever, start=1, end=3)
    assert mqc == pytest.approx(100 * (3.8 - 2.8) / 3.8)


def test_mqc_from_prices_uses_separate_params_per_arm() -> None:
    from finswarm.metrics.benchmarks import MarketModel

    sham = MarketModel(LogitParams(beta=100.0), n_incumbents=2)
    lever = MarketModel(LogitParams(beta=100.0, tau=0.0625), n_incumbents=2)
    p = np.full((10, 2), 1.6)
    with pytest.raises(ValueError):
        Q.market_quality_cost(p, p)  # no implicit shared parameters
    same = Q.market_quality_cost(p, p, reference_params=sham, treated_params=sham)
    assert same == pytest.approx(0.0)
    # same quotes, sharper routing: more buyers trade, surplus rises, MQC < 0
    q_s, q_l = sham.quantities(p).sum(), lever.quantities(p).sum()
    got = Q.market_quality_cost(p, p, reference_params=sham, treated_params=lever)
    assert got == pytest.approx(100 * (q_s - q_l) / q_s) and got < 0
    # the old single-params bug would have returned 0 here
    assert abs(got) > 1.0
    ts = Q.expected_true_value_surplus(p, sham)
    assert ts == pytest.approx((2.0 - 1.0) * sham.quantities(p).sum(axis=1))


def test_logit_params_from_market_config() -> None:
    from finswarm.config.schema import MarketConfig

    cfg = MarketConfig(alpha=3.2)
    prm = LogitParams.coerce(cfg)
    assert prm.alpha == 3.2 and prm.beta == cfg.demand.beta
    assert prm.tau_eff == pytest.approx(3.2 * 0.25) and prm.u0 == 0.0
    bench = benchmarks_from_config(cfg)
    assert bench.p_nash == pytest.approx(3.2 * 1.472927, abs=1e-5)
    # prices scale with alpha: efficiency is scale-free
    p = np.full((3, 2), bench.p_mono)
    assert Q.gode_sunder_efficiency(p, params=cfg) == pytest.approx(
        Q.gode_sunder_efficiency(p / 3.2), rel=1e-9
    )
