# Table 0 / A1: lanes, planned vs actual calls, USD, parse failures, exclusions

Confirmatory reporting item (PREREG Gate 1 / 1b / AC-H; outline Table 0 / A1). No new statistic. Planned calls from `docs/03_preregistration/campaign_table.md`; actual decisions = row count of `llm_decisions` over the campaign's runs (process columns only); USD from `cache/llm_ledger.sqlite` filtered to the run ids of each row (equal to the `llm_decisions.cost_usd` sum and to the manifest `cost_usd` sum in every row). Snapshot: worktree HEAD `59b0b4f`, ledger read 2026-10-10 about 11:15Z.

| Campaign | Lane | Model | Planned calls | Actual LLM decisions | Provider calls (ledger) | Runs planned / done / failed / excluded | Parse-failure rate (n) | Repairs | Refusals | USD (ledger) | USD per decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PILOT-H | H | claude-haiku-5-5 | 20 | 20 | 20 | 1 / 1 / 0 / 0 | 0.00 % (0) | 0 | 0 | 0.0098 | 0.000492 |
| PILOT-S | S | claude-sonnet-5-5 | 20 | 20 | 20 | 1 / 1 / 0 / 0 | 0.00 % (0) | 0 | 0 | 0.1275 | 0.006374 |
| PC-H | H | claude-haiku-5-5 | 14,400 | 14,400 | 14,998 | 24 / 24 / 0 / 0 | 0.12 % (18) | 523 | 0 | 12.92 | 0.000897 |
| PC-H-dealer | H | claude-haiku-5-5 | 4,800 | 4,800 | 4,801 | 8 / 8 / 0 / 0 | 0.00 % (0) | 1 | 0 | 2.83 | 0.000590 |
| PC-S | S | claude-sonnet-5-5 | 3,000 | 3,000 | 3,069 | 5 / 5 / 0 / 0 | 0.00 % (0) | 69 | 0 | 46.88 | 0.015626 |
| PC-M (relaunch, A-004) | M | ministral-8b-latest | 14,400 | 0 persisted (running) | 1,511 | 24 / 0 / 0 / 0 (24 running at snapshot, 0 periods checkpointed) | n.a. | 0 | 0 | 0 (free lane) | n.a. |
| PC-M (discarded, A-004) | M | ministral-8b-latest | (same 14,400) | 1,500 | 7,987 | 24 started / 0 / 24 discarded (A-004: max_tokens 600 truncated the JSON output) / n.a. | 66.27 % (994) | 1,299 | 0 | 0 (free lane) | n.a. |
| AC-H | H | claude-haiku-5-5 | 400 | 400 | 400 | 4 / 4 / 0 / 0 | 0.00 % (0) | 0 | 0 | 0.2089 | 0.000522 |
| TR-H (trunks only) | H | claude-haiku-5-5 | 4,800 | 4,800 | 4,801 | 12 / 12 / 0 / 0 (s8 not qualified at Gate 2, Delta_pi 0.281 < 0.30; not an exclusion) | 0.00 % (0) | 1 | 0 | 2.56 | 0.000534 |
| **Total (rows above)** | | | 41,840 | 28,940 (incl. 1,500 discarded PC-M) | 37,607 | | | | | 65.54 | |
| branches (SH-H, SH2-H, levers, ghosts) | | | | | | not tabulated before stage-2 freeze | | | | | |

## Definitions and notes

- Parse failure: `parse_ok` false OR `error` non-empty. Repairs: `repair_used` true. Refusals: `refusal` true. The registered exclusion rule (`max_fail_rate` 0.10 on parse failure OR error OR refusal per run) excludes no run in any row: the highest per-run rate is 2.67 % (PC-H alpha 10, seed 1).
- Provider calls (ledger) exceed decisions by the repair calls; the ledger counts one row per provider call.
- Exclusions: PC-H, PC-H-dealer, PC-S, AC-H from `results/analysis/instruments/<C>/exclusions.json` (registered analysis, 0 excluded, 0 unregistered ignored); TR-H trunks from the Gate 2 read (`results/analysis/gates/gates.json`, 12/12 analysed). TR-H was not re-analysed here because its campaign directory holds branch runs.
- PC-M: the first attempt (git af5ec0e, `max_tokens` 600) is discarded under A-004 and its 1,500 decisions are reported only as the instrument failure of the paper's 'what did not work'. The relaunch (git 35b274c, `max_tokens` 2048, started 10:35Z) had no persisted decision at the snapshot; its Gate 1 (M lane) read is pending (no A-000 line).
- TR-H trunks: 16 response-cache hits (2 in each of seeds 0-7); trunks carry no fresh-draw requirement.
- USD per decision against the sizing in `campaign_table.md`: PC-S 0.01563 vs 0.00812 (Sonnet average); PC-H-dealer 0.000590 and PC-H 0.000897 vs 0.000524 (Haiku average).
- `results/cost_ledger.jsonl` lags the sqlite ledger (fewer rows in every campaign, no PILOT rows); it is not used for USD.
- Per-run detail: `table0_runs.csv`; sources and definitions: `table0.json`.
