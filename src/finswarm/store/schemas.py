"""Parquet table schemas (frozen interface, 2026-10-09). Single source of truth for logging."""

from __future__ import annotations

import polars as pl

SCHEMA_VERSION = 1

ORDERS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "tick": pl.Int64,
    "period": pl.Int32,
    "order_id": pl.Int64,
    "agent_id": pl.Int32,
    "agent_kind": pl.Int8,
    "swarm_type": pl.Int8,
    "side": pl.Int8,
    "otype": pl.Int8,
    "price": pl.Float64,
    "price_ticks": pl.Int64,
    "size": pl.Float64,
    "tif_ticks": pl.Int32,
    "status": pl.Int8,
    "reason": pl.Utf8,
}

TRADES: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "trade_id": pl.Int64,
    "tick": pl.Int64,
    "period": pl.Int32,
    "price": pl.Float64,
    "size": pl.Float64,
    "buyer_id": pl.Int32,
    "seller_id": pl.Int32,
    "buyer_kind": pl.Int8,
    "seller_kind": pl.Int8,
    "aggressor": pl.Int8,
    "mechanism": pl.Utf8,
    "batch_id": pl.Int64,
    "buyer_value": pl.Float64,
    "seller_cost": pl.Float64,
}

FLOW_PER_TICK: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "tick": pl.Int64,
    "period": pl.Int32,
    "dealer_id": pl.Int32,
    "buy_qty": pl.Float64,
    "sell_qty": pl.Float64,
    "n_arrivals": pl.Int32,
    "n_outside": pl.Int32,
}

QUOTES: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "period": pl.Int32,
    "epoch": pl.Int64,
    "tick": pl.Int64,
    "agent_id": pl.Int32,
    "slot": pl.Int8,
    "is_llm": pl.Boolean,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
    "inventory": pl.Float64,
    "cash": pl.Float64,
    "decision_id": pl.Int64,
    # additive (2026-10-10): decision tick and the period containing ``tick`` (apply tick)
    "decision_tick": pl.Int64,
    "apply_period": pl.Int32,
}

BOOK_SNAPSHOTS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "tick": pl.Int64,
    "period": pl.Int32,
    "side": pl.Int8,
    "level": pl.Int16,
    "price": pl.Float64,
    "depth": pl.Float64,
    "n_tagged": pl.Int32,
}

LLM_DECISIONS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "decision_id": pl.Int64,
    "period": pl.Int32,
    "epoch": pl.Int64,
    "obs_tick": pl.Int64,
    "apply_tick": pl.Int64,
    "agent_id": pl.Int32,
    "slot": pl.Int8,
    "provider": pl.Utf8,
    "model": pl.Utf8,
    "lane": pl.Utf8,
    "prompt_version": pl.Utf8,
    "prompt_hash": pl.Utf8,
    "view_hash": pl.Utf8,
    "view_policy_id": pl.Utf8,
    "cache_hit": pl.Boolean,
    "raw_output": pl.Utf8,
    "parsed_action": pl.Utf8,
    "action_type": pl.Utf8,
    "price": pl.Float64,
    "size": pl.Float64,
    "plans": pl.Utf8,
    "insights": pl.Utf8,
    "message": pl.Utf8,
    "parse_ok": pl.Boolean,
    "repair_used": pl.Boolean,
    "refusal": pl.Boolean,
    "latency_ms": pl.Float64,
    "input_tokens": pl.Int32,
    "output_tokens": pl.Int32,
    "cached_tokens": pl.Int32,
    "cost_usd": pl.Float64,
    "request_id": pl.Utf8,
    "error": pl.Utf8,
}

EVENTS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "event_id": pl.Int64,
    "tick": pl.Int64,
    "period": pl.Int32,
    "event_type": pl.Utf8,
    "payload": pl.Utf8,
    "source": pl.Utf8,
}

AGENT_PERIOD: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "period": pl.Int32,
    "agent_id": pl.Int32,
    "slot": pl.Int8,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "profit": pl.Float64,
    "inventory": pl.Float64,
    "cash": pl.Float64,
    "undercut": pl.Boolean,
}

METRICS_PER_PERIOD: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "period": pl.Int32,
    "tick_start": pl.Int64,
    "tick_end": pl.Int64,
    "fundamental": pl.Float64,
    "mid_mean": pl.Float64,
    "quoted_spread": pl.Float64,
    "effective_spread": pl.Float64,
    "depth_top": pl.Float64,
    "volume": pl.Float64,
    "n_trades": pl.Int32,
    "realised_vol": pl.Float64,
    "pricing_error": pl.Float64,
    "llm_profit_total": pl.Float64,
    "swarm_surplus": pl.Float64,
    "delta": pl.Float64,
    "parse_fail_count": pl.Int32,
    # additive (2026-10-10): False when the closed-form profit benchmarks do not apply
    # (fringe share > 0); delta is then NaN
    "benchmark_valid": pl.Boolean,
}

TABLES: dict[str, dict[str, pl.DataType]] = {
    "orders": ORDERS,
    "trades": TRADES,
    "flow_per_tick": FLOW_PER_TICK,
    "quotes": QUOTES,
    "book_snapshots": BOOK_SNAPSHOTS,
    "llm_decisions": LLM_DECISIONS,
    "events": EVENTS,
    "agent_period": AGENT_PERIOD,
    "metrics_per_period": METRICS_PER_PERIOD,
}
