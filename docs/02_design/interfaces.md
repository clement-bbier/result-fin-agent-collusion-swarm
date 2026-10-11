# Interfaces (frozen 2026-10-09, evening)

These contracts let the modules of `finswarm` be built and changed independently. A change to
any of them touches this file and every module that depends on it. Everything else inside a
module is that module's business.

## Contracts

- `finswarm.core.types` — enums (`Side`, `AgentKind`, `OrderType`, `MechanismKind`, `SwarmType`)
  and dataclasses (`Order`, `OrderAck`, `Trade`, `Quote`, `PeriodRecord`, `Observation`,
  `Decision`). Prices inside markets are integer ticks; floats only at the boundary.
- `finswarm.core.rng.RNGRegistry` — named streams `population, swarm, matching, batch, llm,
  interventions, view`; `capture()`/`restore()` are JSON-serialisable; `respawn("interventions",
  branch_id)` on branch.
- `finswarm.market.base.Market` — `submit`, `cancel`, `advance(n_ticks)`, `best_bid/ask/mid`,
  `snapshot(depth)`, `drain_trades`, `state`/`restore`, `migrate_from`. Mechanisms register in
  `finswarm.market.registry` under their `MechanismKind`.
- `finswarm.agents.base.Agent` + `AgentMemory` — `async decide(obs) -> Decision`,
  `record_period`, `state`/`restore`. LLM agents, scripted agents and the sting dealer all
  implement it.
- `finswarm.llm.client` — `LLMRequest` / `LLMResponse` / `Capabilities` / `LLMClient`.
  Providers live only in `finswarm.llm.providers`. The lane scheduler exposes
  `await scheduler.submit(req, lane) -> LLMResponse` and is the only path to a provider.
- `finswarm.perturb.events` — discriminated union `Event` (`set_param`, `replace_agent`,
  `inject_notice`, `info_shock`, `remove_liquidity`, `add_scripted_agent`, `sting`,
  `forced_deviation`, `revert`) with `when: {at_period | at_tick | on_condition}`.
- `finswarm.config.schema` — `RunConfig`, `CampaignConfig`, `LaneConfig`, `BranchSpec` and
  sub-models; `extra="forbid"`; one LLM family per run unless `allow_mixed_families`.
  `finswarm.config.load.run_id()` is the deterministic run id.
- `finswarm.store.schemas.TABLES` — Parquet columns and dtypes for `orders, trades,
  flow_per_tick, quotes, book_snapshots, llm_decisions, events, agent_period,
  metrics_per_period`. Writers write parts per period; readers validate dtypes.
- `finswarm.metrics.registry.register_metric(name, unit, level, higher_is_collusive, source)`
  — every metric declares its provenance (`source` = citation key or `"novel"`).

## Conventions that are part of the contract

- Simulated time is decoupled from wall time. Each LLM decision logs `obs_tick` and
  `apply_tick`.
- Determinism: same resolved config + seed ⇒ bit-identical `trades` and `quotes` tables with
  the mock provider and with `llm.mode: replay`.
- A branch is byte-identical to its parent before the branch period.
- Labels that identify colluders (`roster.colluder`) never reach an `Observation`.
- Anthropic lane: never send `temperature`; `effort: low`; `max_tokens >= 2048`.

## Integration conventions (settled 2026-10-09)

- **Tagged ids.** Swarm agents are `0..N-1` (`N = swarm.n_agents` in `agents` mode, else 0);
  tagged slot `s` is agent `N + s` (= `market.dealer_agent_id(s)`). The closed-form aggregate
  buyer is `-1`. The runner builds the market with `n_dealers = max(market.n_dealers,
  len(llm.slots)) + 2` seats (`integration.ENTRANT_RESERVE`) so entrants (`add_scripted_agent`,
  `sting` on a new slot) have a seat; unused seats never quote and their flow rows are dropped.
- **QUOTE replaces.** A `QUOTE` order replaces the sender's standing quote on that side at its
  `apply_tick`; the runner never cancels before re-quoting. `CANCEL`/`market.cancel` withdraw.
- **Market factory.** `finswarm.sim.integration.real_market_factory(cfg, rng, swarm=None)` ->
  `make_market(cfg, rng, swarm)`; in `agents` mode the swarm is drawn there from the
  `population` stream (`build_population_for_run`). Injected test doubles keep
  `factory(cfg, rngs)`.
- **Hooks** (`integration.market_hook`): `info_shock` -> `market.apply_info_shock(dF,
  diffusion)`; `remove_liquidity` -> `finswarm.swarm.shocks.remove_liquidity(market.swarm,
  share, types, rng["interventions"])` (removed ids kept for `restore_liquidity` on revert);
  `set_param` -> `market.set_param(path, value)` for paths in `market.settable_params`, then
  `integration.after_market_param(runner, path)` (runner copies re-gridded, a `derived` event
  row with the market's `last_param_report`). Anything else (`two_sided`, `price_grid.min/max`,
  mechanism change, the fringe outside `agents` mode, a missing hook, an unimplemented
  mechanism) raises `UnsupportedIntervention`: the run ends `failed` with the reason in
  `manifest.error`; config events are checked before period 0. Test doubles without
  `settable_params` are rebuilt and `migrate_from`'d.
- **Tick re-grid (lever L3).** `set_param("market.price_grid.tick", t)` is live in every
  dealer-market mode, at a period boundary. `p_min`/`p_max` stay; `n_levels = round((p_max -
  p_min)/t) + 1`. Every standing price is re-quantised by `finswarm.market.grid`: asks round
  UP and bids DOWN to the next grid point (never more aggressive, never crossed); a price with
  no image on the new grid is withdrawn. This covers dealer asks, pending (speed-bumped)
  quotes, fringe depth, and on the runner side each slot's `ask`/`bid` and the public quote
  log; the last epoch's trades shown to agents map to the nearest level; trades not yet drained
  keep their traded price. The re-grid is logged as a `derived` `regrid` event (old/new tick,
  `n_levels`, per-dealer old/new ticks and prices, pending orders, fringe lots moved/dropped).
  `market.state()` carries the grid, so a checkpoint taken after a re-grid restores onto the
  new grid whatever config builds the market. A re-grid back is lossy (rounding is one-way).
- **Public limit-order fringe (lever L2, mechanism B).** `set_param("market.fringe.share", s)`
  (`agents` mode only; 0 -> 0.3 -> 0 from any checkpoint). An arriving ZI buyer with routing
  uniform `ur < s` becomes a fringe seller (same tape consumption, so CRN with the sham). Its
  ask is its reservation price `r = c*alpha + u * max(0, v - c*alpha)`, `u = ur/s ~ U[0,1)`,
  `v` the reservation value it would have had as a buyer (`alpha*(F_belief + eps + noise +
  w_s*(H + x)) + trend_k*trend`), i.e. uniform between the dealers' marginal cost and its own
  valuation (about `U[1, 2]*alpha` in the Calvano calibration, mean ~1.5 alpha, so roughly half
  the lots undercut a Nash-priced dealer and nearly all undercut a monopoly-priced one). One
  lot, rounded UP to the dealer grid (dropped above it), lifetime `fringe.horizon_ticks`,
  aggregated per level with an expiry wheel, filled oldest-first. Buyers route by logit over
  `{dealer asks, best fringe ask, outside}` with the same `tau`. Fringe trades: `seller_id =
  -2`, `seller_cost` = traded price (the fringe earns ~0 surplus; trades-based MQC), the
  fringe is one aggregate account (conservation). `flow_per_tick` gets `dealer_id = -2` rows
  (`buy_qty` bought from, `sell_qty` posted to the fringe) once the fringe was switched on;
  fringe depth is in `snapshot()`/`book_levels` and `best_ask()`. `share = 0` with no resting
  lots runs the mechanism-A kernel unchanged. **Benchmarks with the fringe on:** Calvano's
  closed form no longer applies (dealers face the residual demand of a competitive fringe);
  `p_N`, `p_M`, `pi_N`, `pi_M` are the static Nash and symmetric joint-profit maximum of the
  dealers' one-shot pricing game in the market with the fringe, estimated by
  `metrics.benchmarks.estimate_by_sweeps(market.sweep.profit_fn(cfg), grid, n_dealers)` (CRN,
  fixed dealer asks, fringe share/horizon/tau/grid/swarm as in the evaluation window).
  `MechanismKind.FRINGE` is not registered separately (`registry.py` is frozen).
- **Drains.** Per epoch: `drain_trades_arrays()` -> one columnar `trades` chunk;
  `drain_flow()` -> one columnar `flow_per_tick` chunk (`EpochBuffers.add_columns`, no per-tick
  Python loop); per period: `drain_periods()` gives `agent_period.quantity/profit` of sellers.
  Markets without the array drains fall back to `drain_trades()`.
- **Checkpoint composition** (`run_dir/checkpoints/pNNNNNN.ckpt`): `rng.capture()`,
  `market.state()` and `market.swarm.state()` taken at the same moment (nothing draws in
  between), agent states, public-view state, liquidity-removal stack, quote/book logs, engine,
  metrics, writer counters. Restore order: `RNGRegistry(seed)` -> `SwarmState.from_state` ->
  `make_market(cfg, rng, swarm)` -> `market.restore` -> agents -> view -> `rng.restore`.
- **Decision clock** (`finswarm.core.clock`, merged 2026-10-10 00:30Z). The runner takes decision
  batches from the clock each period and runs the market up to the batch tick. `sync`: every slot
  decides every `epoch_ticks`, re-aligned at each period start (`ceil(tpp/E)` epochs; byte-identical
  to the pre-clock runner). `poisson`: per-slot next decision `t + Geometric(1/E)` (mean `E`,
  drawn from `matching`); all slots quote at tick 0, entrants at the boundary where they join.
  `staleness_window w`: decisions with ticks in `[t, t+w)` (cut at the period end) share one
  observation as of `t`. **Speed bump (L7)**: FIFO constant delay `apply_tick = decision_tick +
  speed_bump_max`, never earlier than the slot's previous apply tick, no random draw (registered
  D-9: 300 ticks = 3 periods); optional fixed per-slot extra latency `U{0..J}` from
  `tags.speed_bump_jitter` (order preserved). `cancel` actions are delayed likewise; withdrawing a
  slot drops its pending orders. Live `set_param` paths: `clock.mode`, `clock.epoch_ticks`,
  `clock.speed_bump_max`, `clock.staleness_window` (period boundary; mode / `epoch_ticks` redraw
  the schedule, D / w keep it; invalid values raise `UnsupportedIntervention` before period 0).
  The runner tracks the quote a slot last *sent* separately from the quote *in force*; rivals'
  observations, period prices and snapshot counts use the in-force quote. `quotes` rows: a slot
  that decided gets a row with its decision id, the sent quote and `tick = apply_tick`; every
  other active slot gets its in-force quote at the epoch tick; `llm_decisions.obs_tick` is the
  batch tick. Checkpoints add `clock` (config, epoch index, next decision tick per slot, sync
  grid position, jitter offsets, last apply tick), `pending_quotes` and per-slot in-force
  quote / order ids; older checkpoints still load (schedule restarts at the checkpoint tick).
- **Engine red-team follow-ups (merged 2026-10-10 05:40Z, F1-F7).**
  *Order flow per slot (F1)*: each tagged slot keeps its own trade cursor (`SlotState.trade_cursor`,
  checkpointed); its observation reports the trades since *its own* last decision, so the poisson
  clock (L5) no longer shrinks the reported flow to the last batch; sync output is byte-identical.
  *Re-grid (F2)*: `integration._sync_runner_after_regrid` also moves the in-force `live_ask`/
  `live_bid` and the prices of pending (speed-bumped) quotes; pending quotes the market withdrew
  at the re-grid are dropped. *Common random numbers (F3)*: `branch_salt` hashes only
  `(parent_run_id, branch_period)`, so a lever branch and its sham share the `interventions`
  stream; each branch event draws from its own child stream keyed by (rank among branch events
  of the same type, hash of the type), so lever-specific events do not shift the sham's draws.
  *`quotes` columns (F4)*: `tick` keeps meaning the apply tick (ghost replay relies on it);
  `decision_tick` and `apply_period` are added. *Online benchmarks (F5)*: `online_benchmarks`
  are frozen at run start (checkpointed) and the online `delta_p` always uses them; when
  `fringe.share > 0`, `metrics_per_period.delta` (profit-based) is NaN and `benchmark_valid` is
  False (the closed-form profit benchmarks do not apply; the analysis layer uses the pinned sweep
  benchmarks). *Fringe seller cost (F6, convention)*: a fringe trade records `seller_cost` = the
  traded grid level while the seller's reservation `r` lies in `(level - tick, level]`, so the
  fringe seller surplus missing from a trades-based total surplus is in `[0, tick)` per fringe
  trade (<= 0.05 alpha on the registered grid); an exact per-level accumulator is optional work.
  *Fringe flow row (F7)*: the `dealer_id = -2` row of `flow_per_tick` carries `n_arrivals =
  n_outside = 0`, so per-tick counts are summed over dealer rows only.
- **`Observation.extra`** (runner): `p_min`, `tick_size`, `price_max`, `alpha`, `two_sided`,
  `epoch`, `epoch_in_period`, `max_wtp` (`4.51 * alpha`, override `RunConfig.tags["max_wtp"]`),
  `n_rounds` (= `periods`), `nash_price`, `monopoly_price`, `default_price` (= Nash; override
  `tags["default_price"]`) from `finswarm.market.demand.benchmarks` on the routed demand
  (`demand.routed_params`: `tau` and `outside_utility` folded in) for the active sellers, and
  `view_hash` (set by the view). `cost` = `market.cost * alpha`. Keys starting with `_raw` are
  view inputs and never reach an agent.
- **View.** `finswarm.market.view.apply_view(obs, policy, rng, *, view=None) -> Observation`
  transforms rival quotes, trades, book and history, sets `obs.view_policy_id` and
  `obs.extra["view_hash"]`; the runner passes its checkpointed `PublicView` (stable pseudonyms).
- **LLM factory.** `finswarm.agents.llm.make_llm_agent(slot_cfg, *, slot, agent_id, run_cfg,
  run_id, rngs, provider, lane, mode) -> RunnerLLMAgent` (an `LLMAgent` with `last_call`,
  `last_prompt`, `set_llm_mode(mode)`, `async aclose()`); model aliases resolve through
  `configs/models.yaml` (`aliases:`), one shared `LaneScheduler` per cache mode, `base_seed =
  run seed`. The runner writes every `drain_logs()` row to `llm_decisions` with its own
  `decision_id`, `epoch` and `apply_tick`. A `mock`-family slot with a scripted `policy` is a
  scripted agent unless its `model` is a mock LLM model (`mock-v1`), in which case it is a real
  `LLMAgent` answered by `MockLLM(policy)`.
- **Manifest keys for analysis.** `seed`, `alpha`, `arm` (`tags.arm`, default `name`),
  `trunk_id` (`parent_run_id` for branches, else `run_id`), `condition` (= `name`), `label`
  (`tags.label`), plus `parent_run_id`, `branch_period`, `branch_events_hash`; tables at
  `<run_dir>/<table>/part-*.parquet`.
- **Budget.** `configs/budget.yaml`: `anthropic_total_usd`, `warn_usd`, `stop_usd` (lane
  scheduler) and `per_run_max_usd` (the runner pauses a run above it). Dry-run pricing:
  `finswarm.llm.pricing.estimate_call_cost(model, in_tokens, out_tokens)`.
