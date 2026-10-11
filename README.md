# Rules or Rivals? Paired Interventions on Entrenched Supra-Competitive Pricing by LLM Dealers

Results, paper and code of a weekend study (9–11 October 2026) on one of the questions of the Apart
Research *AI Collusion Research Sprint*. Independent work, not conducted at the sprint.

**Paper:** [`paper/release/PAPER.pdf`](paper/release/PAPER.pdf) (also [LaTeX build](paper/release/PAPER_latex.pdf),
[DOCX](paper/release/PAPER.docx), Markdown [`paper/PAPER.md`](paper/PAPER.md)).

## Question

Once LLM dealers have settled on supra-competitive prices without communicating, what moves them, and
at what cost to market quality: changes to the market's rules, or new competition?

## Setup

Four Claude Haiku 5.5 dealers post asks to a 10,000-agent simulated order flow (Bertrand-logit demand,
competitive price 1.33, joint-profit price 2.05). In 11 of 12 markets the dealers settle on a common
ask above the joint-profit price. At period 100 each market is branched: one intervention per branch,
compared with an unchanged sham branch of the same market over periods 111–150. Each intervention's
cost is measured in scripted competitive markets. Hypotheses and decision rules were pre-registered
and hash-frozen before the data they govern.

## Results (in this testbed)

Capped price-index reduction against the sham (0 = no change; the index is capped at the joint-profit
level), 11 markets, 95 % intervals; raw = uncapped contrast; cost = market-quality cost in % of
surplus (negative = gain).

| Intervention | Capped reduction | Raw reduction | Quotes above joint-profit price | Cost (%) | Holm p |
| --- | --- | --- | --- | --- | --- |
| Sham (no change) | 0 | — | 80 % | 0 | — |
| Sharper routing (L1) | 0.001 | 0.20 [0.13, 0.26] | 78 % | −1.78 | 1 |
| Public limit-order fringe (L2, primary H1) | 0.003 | 0.025 [0.010, 0.048] | 80 % | 26.53 | 1 |
| Finer tick (L3) | 0.001 | 0.00 | 80 % | −0.11 | 1 |
| Anonymised quotes (L4a) | 0.000 | 0.00 | 80 % | 0.00 | 1 |
| Staggered updates (L5) | 0.000 | 0.00 | 80 % | 0.00 | 1 |
| Speed bump (L7) | 0.000 | 0.00 | 80 % | 0.00 | 1 |
| No-regret entrant (L6) | **0.62** [0.42, 0.75] | 1.00 [0.70, 1.24] | 9 % | −0.58 | **< 0.001** |
| Static Nash-price entrant (L6c, exploratory) | 0.63 [0.53, 0.72] | 1.05 [0.88, 1.20] | 1 % | — | — |
| Entrant pegged below the best rival (L6b, exploratory) | 0.03 [0.01, 0.07] | 0.15 [0.11, 0.19] | 74 % | — | — |

- **No rule change brought asks below the joint-profit price;** the primary hypothesis (the public
  fringe) is not supported. Sharper routing and the fringe trimmed asks inside the high-price band.
- **An entrant posting a competitive price pulled asks to its own band** (average ask from about 2.35
  to about 1.63) at no measured efficiency cost. Incumbents stayed 0.34 index points above their static
  best response. A static Nash-price seller did about as well as a learning entrant; an entrant pegged
  to the incumbents' asks barely moved them.
- **Replacing an incumbent** by a fresh copy, with or without memory, left the regime in place.
- **Detection:** the frozen detection rules could not be assessed (2 positive and 26 negative held-out
  windows); a scratchpad judge agreed poorly with blind human labels (weighted kappa 0.13, n = 60).

Every number above is in [`paper/numbers.json`](paper/numbers.json) with its claim in
[`paper/claims.md`](paper/claims.md).

## Limits

One model family under one dealer prompt; 11 markets; the regime is a focal price above the
joint-profit level, and the data do not separate strategic restraint from inertia; the replication of
Fish et al. (2024) failed on Haiku 5.5 under the modified protocol and passed on Sonnet 5.5 (five runs);
entrants are scripted, with unlimited capacity and no buy-out; dealers post asks only (no bids, no
inventory). The effect of entry on LLM pricing was reported before (Garra 2026; Keppo et al. 2026);
this study adds mid-run entry into an installed regime, paired branches with sham controls, and a cost
per intervention.

## Repository

| Path | Content |
| --- | --- |
| `paper/` | the paper (PDF, DOCX, LaTeX source, Markdown), figures, tables, `numbers.json`, claims register |
| `results/manifests/` | one manifest per run: configuration hash, seed, git commit, prompt hashes, model ids |
| `results/analysis/` | per-campaign statistics, gates, family-S tests, detection, Table B0, verification report |
| `configs/` | every campaign configuration (YAML) and the analysis settings |
| `docs/03_preregistration/` | pre-registration, statistical analysis plan, detection protocol, amendments, hash list |
| `docs/06_corpus/` | labelled scratchpad corpus codebook and seal |
| `src/finswarm/` | simulator, LLM layer, interventions, metrics, analysis |
| `scripts/` | analysis, verification, figures, pre-registration check |
| `tests/` | unit, property and smoke tests (mock provider) |

## Reproduce

```bash
uv sync --frozen
uv run pytest -q                                   # engine and analysis tests (mock provider, no API key)
uv run python scripts/check_prereg.py --strict     # pre-registration hash over the frozen file set
uv run finswarm run configs/smoke.yaml --provider mock
```

Re-running the LLM campaigns needs API keys in `.env` (see `.env.example`); `configs/budget.yaml` caps
paid spend. The raw run logs (about 9 GB of Parquet) are not in this repository; the manifests, the
per-campaign statistics and `paper/numbers.json` are. `scripts/verify.py` re-derives every number of
`numbers.json` from the raw logs without the simulator's code. The paper cites the development
repository and its commits; that history and the raw logs are available on request.

## License

MIT.
