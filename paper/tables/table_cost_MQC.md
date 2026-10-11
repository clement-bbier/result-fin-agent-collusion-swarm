# Cost table: market-quality cost per lever (SC-1 scripted controls, 20 paired seeds, window 111-150)

MQC = loss (pp) of true-value surplus per period, lever vs sham, SC1 best-responder dealers. Negative = a gain. The per-buyer and per-trade columns are EXPLORATORY (A-009), not registered; no verdict.

| Lever | MQC (pp) | BCa 95 % | SC1 mean ask: sham -> lever | Surplus per arriving buyer (sham 0.976) | Surplus per trade (sham 0.998) |
| --- | --- | --- | --- | --- | --- |
| L1 routing | -1.78 | [-1.92, -1.66] | 1.35 -> 1.10 | 0.993 | 0.993 |
| L2 fringe | 26.53 | [26.28, 26.83] | 1.35 -> 1.35 | 0.716 | 0.943 |
| L3 tick | -0.11 | [-0.14, -0.09] | 1.35 -> 1.33 | 0.977 | 0.997 |
| L4a anonymised view | 0.00 | [0.00, 0.00] | 1.35 -> 1.35 | 0.976 | 0.998 |
| L6 entrant | -0.58 | [-0.64, -0.53] | 1.35 -> 1.37 | not computed | not computed |
| L5, L7 | no MQC campaign | | | | |
| L8 | not run | | | | |

Source: `results/analysis/SC-1/mqc.json`. L2's cost is lost buyers: 30 % of arriving buyers become fringe sellers (registered design).
