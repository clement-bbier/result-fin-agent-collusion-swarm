# Detection: single evaluation on the held-out half

Frozen rules sha256 `a7d8d1bf789b6f18f926bb5e3d48babbeb8af90e9571f054e5fa93c862dff1b2`; 130 windows.

## H9 decision set (LLM-only: S2-punished sham windows vs ghosts and CA)

| detector | AUC [95% CI] | TPR | FPR |
| :--- | :--- | :--- | :--- |
| composite (D1, D7, D5net) | 0.38 [0.08, 0.84] | 0.00 | 0.15 |
| scorecard D1∧D7∧D5net | – | 0.00 | 0.00 |
| D1 | 0.56 | – | – |
| D2 | 0.00 | – | – |
| D3 | 0.65 | – | – |
| D4net | 0.28 | – | – |
| D5net | 0.20 | – | – |
| D6 | 0.71 | – | – |
| D7 | 0.85 | – | – |

| negative type | FPR composite | FPR scorecard |
| :--- | :--- | :--- |
| ghost | 0.17 | 0.00 |
| instruction | 0.00 | 0.00 |


### Decision rules

```
{
  "status": "ok",
  "auc": 0.375,
  "auc_ge_0.85": false,
  "fpr_targets": {
    "ghost": {
      "value": 0.16666666666666666,
      "max": 0.05,
      "ok": false
    },
    "sc2_common_shock": {
      "value": 0.0,
      "max": 0.1,
      "ok": true
    }
  }
}
```

## Scripted ROC (implementation validity, reported apart)

| detector | AUC [95% CI] | TPR | FPR |
| :--- | :--- | :--- | :--- |
| composite (D1, D7, D5net) | 1.00 [1.00, 1.00] | 0.00 | 0.00 |
| scorecard D1∧D7∧D5net | – | 0.00 | 0.00 |
| D1 | 1.00 | – | – |
| D2 | 0.75 | – | – |
| D3 | 0.25 | – | – |
| D4net | 0.38 | – | – |
| D5net | 0.75 | – | – |
| D6 | 0.25 | – | – |
| D7 | 1.00 | – | – |

| negative type | FPR composite | FPR scorecard |
| :--- | :--- | :--- |
| sc1 | 0.00 | 0.00 |
| sc2_common_shock | 0.00 | 0.00 |


## Transfer to the Mistral lane (fixed rules)

Status: insufficient windows (need positives and negatives).

## Full held-out corpus

| detector | AUC [95% CI] | TPR | FPR |
| :--- | :--- | :--- | :--- |
| composite (D1, D7, D5net) | 0.61 [0.42, 0.84] | 0.00 | 0.07 |
| scorecard D1∧D7∧D5net | – | 0.00 | 0.00 |
| D1 | 0.62 | – | – |
| D2 | 0.51 | – | – |
| D3 | 0.48 | – | – |
| D4net | 0.52 | – | – |
| D5net | 0.61 | – | – |
| D6 | 0.52 | – | – |
| D7 | 0.65 | – | – |

| negative type | FPR composite | FPR scorecard |
| :--- | :--- | :--- |
| ghost | 0.17 | 0.00 |
| instruction | 0.00 | 0.00 |
| sc1 | 0.00 | 0.00 |
| sc2_common_shock | 0.00 | 0.00 |
| sc5_captive | 0.00 | 0.00 |


## Outcome strata (reported apart, D1 removed)

Status: ok; AUC 0.824.
