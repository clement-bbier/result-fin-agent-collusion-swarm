# Follow-up experiments (not run)

Status: none of these was run. Each needs a pre-registration entry before its data are read;
until then it is exploratory.

## Starting point (what the weekend established, in this testbed)

Four Claude Haiku 5.5 dealers (`dealer_v1`), 10,000 scripted buyers (Bertrand-logit, tau 0.25),
branching at period 100 from an entrenched supra-competitive focal-price regime (11 of 12 trunks).
Six rule changes left asks above the monopoly price; an entrant posting a competitive price pulled
asks to its band at no measured efficiency cost; the entrant is scripted, with unlimited capacity and
no buy-out. These follow-ups attack the three open conditions: *durability of entry*, *cure versus
prevention*, and *scope* (one model family, one prompt, n = 4).

## Method rules that apply to every follow-up

1. Pre-register before coding: hypothesis, decision rule, minimum effect of interest, number of trunks.
2. Two measures every time: incumbents' index (capped and raw, share above p_M) and the MQC cost on a
   healthy scripted market.
3. Paired design only: branch from a common checkpoint, sham and sham2, common random numbers.
4. Recompute p_N and p_M whenever tau, n or costs change.
5. Anything not pre-registered before its data is exploratory and labelled so.
6. Change one thing per arm; a change to the prompt is a prompt-axis arm and says so.

## Tier 1: do first (they decide whether the main conclusion survives)

| ID | Idea | Why it matters | Needs |
| --- | --- | --- | --- |
| P1 | **Early branching / rule from period 1** (branch at 10, 25, 50, 100) for L2, L5, L7, L1 and entry | Every weekend null is for an installed regime; rules may act on formation. Closes the "too late" objection. GR-H is the only from-scratch evidence so far | existing configs, new branch periods |
| P2 | **Buy-out of the entrant** | The main conclusion holds only if entry is durable; absorption is excluded by design today | an acquisition event in the engine; incumbents' decision rule |
| P3 | **Capacity-limited entrant** (10, 20, 50 % of demand) | Smallest entry that still moves asks is the regulator's number | volume cap on the scripted entrant |
| P4 | **Number of incumbents** (n = 2, 4, 8, 16) | Gap between Fish (n = 2) and this paper (n = 4); gives an HHI-style statement | recompute benchmarks per n |
| P5 | **Second model family that keeps the output format** | Without it the regime is a property of one model and one prompt | a family that passes the JSON format rule (ministral-8b and Llama failed) |
| P6 | **Prompt ablation on Haiku 5.5** (added 2026-10-11): `dealer_v1` with and without the rivals' current quotes, and with and without the market-description paragraph; optionally Fish's prompt with and without `effort: low` and the JSON output | The Fish replication failed on Haiku (price index 0.11) while `dealer_v1` gives 0.65 on the same model; the weekend never separated which of the differences matters. Visible current quotes are the leading, untested candidate | existing prompts plus a template switch; pre-register first |

## Tier 2: worth doing if Tier 1 leaves budget

- **Opening-ask test**: force an opening ask off the round grid (e.g. 1.37) and see whether the focal price follows. Attacks the focal mechanism at its root; cheap.
- **Price tremble** (exogenous +/-0.10 moves on displayed quotes with probability 0.05, 0.2, 0.5) and **random price grid**: merged as "focal-point attacks"; need a market-engine perturbation; strong candidate for a rule that works.
- **Entry threat**: announce a probabilistic entrant in the prompt. This is a prompt-axis arm; label it.
- **Permanent non-colluding incumbent**: 3 LLM + 1 scripted best responder from period 1; cheap prevention variant.
- **Third neutral dealer prompt** and **memory window**: robustness on the prompt axis.
- **Deterministic comparator at extreme routing**: partly covered by L1 (raw cut 0.20, asks still above p_M) and GR-H s4; only an extreme-tau branch is new.
- **Contingent fines**: needs a price-only detector as a trigger and an announcement in the prompt, which is a confound; exploratory.
- **Offline, no LLM cost**: classify existing trunks by focal-only versus reward-punishment signatures.

## Dropped or deferred (and why)

| Idea | Reason |
| --- | --- |
| Intermittent entry; a critical-window study on its own | information already in P1 (timing grid) and P3 |
| Demand shocks, cost asymmetry, multi-market contact, inventory, a strategic buyer | each needs a new engine mechanism; defer until Tier 1 is done |
| A competition bonus | changes the dealers' objective with a subsidy; hard to compare with the other levers |
| Random audits; published scratchpads | audits duplicate the entry threat plus contingent fines; published scratchpads are confounded by the prompt and by the weak scratchpad judge (kappa 0.13) |
| Switching costs | a negative control of low discovery value |
| Two-sided double auction | a different testbed; a next step in the paper |
| Patience; anonymity during formation | prompt axis; anonymity during formation only after P1 |
| Heterogeneous models | blocked until a second family keeps the format; prompt heterogeneity is low value |
| A larger detection set | positives are limited by punished regimes; not obtainable cheaply |

