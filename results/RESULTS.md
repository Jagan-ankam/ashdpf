# ASHDPF experimental results (synthetic environment)

Measured by `python -m ashdpf.cli experiments` on 10,000 synthetic failure events across 500 pipelines / 50 workflow groups, after bootstrapping from 500 historical incidents (seed 42). These are simulation results, not production measurements.

## Table I - Root cause analysis (test = last 30% of stream)

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 78.9% | 81.0% | 79.5% | 80.2% |
| Gradient Boosting (XGBoost-style) | 79.3% | 80.4% | 79.9% | 80.1% |
| Proposed Hybrid RCA (online) | 84.8% | 85.6% | 84.4% | 84.7% |

## Table II - Mean time to resolution

| Method | MTTR (min) | Reduction vs manual | ARR |
|---|---|---|---|
| Manual Operations | 97.1 | 0.0% | 0.0% |
| Rule-Based Automation | 62.8 | 35.3% | 48.9% |
| AIOps Recommendations | 65.4 | 32.6% | 0.0% |
| Proposed Framework | 37.4 | 61.5% | 63.4% |

## Statistical significance (per-incident resolution time, paired)

| Comparison | paired t | p | Wilcoxon W | p | mean diff (min) |
|---|---|---|---|---|---|
| Proposed vs Manual Operations | -103.60 | 0.00e+00 | 4005487 | 0.00e+00 | -59.7 |
| Proposed vs Rule-Based Automation | -35.78 | 7.13e-264 | 14559603 | 1.65e-286 | -25.4 |
| Proposed vs AIOps Recommendations | -48.40 | 0.00e+00 | 7312133 | 0.00e+00 | -28.0 |

One-way ANOVA across the four systems: F = 2998.1, p = 0.00e+00.

## Table III - Knowledge graph impact

| Configuration | RCA accuracy | MTTR (min) | ARR |
|---|---|---|---|
| No Graph | 78.2% | 38.7 | 70.6% |
| Static Graph | 83.4% | 37.5 | 65.1% |
| Adaptive Graph | 84.5% | 37.4 | 63.4% |

## Table IV - Autonomous resolution rate by failure category

| Category | Incidents | ARR | MTTR (min) |
|---|---|---|---|
| data_quality | 1173 | 84.9% | 38.8 |
| job_dependency | 1568 | 75.6% | 25.1 |
| schema_evolution | 1168 | 75.3% | 57.1 |
| resource_exhaustion | 1245 | 74.5% | 31.9 |
| configuration | 1229 | 69.6% | 37.6 |
| infrastructure | 1209 | 66.3% | 32.0 |
| network_timeout | 1201 | 54.2% | 35.6 |
| credential | 1207 | 3.6% | 45.8 |

## Table V - Learning progression (epoch = 1/50 of the stream)

| Epoch | Accuracy | MTTR (min) | ARR | KRI |
|---|---|---|---|---|
| 1 | 79.0% | 43.7 | 51.5% | 74.5% |
| 5 | 83.0% | 35.0 | 62.0% | 81.5% |
| 10 | 84.0% | 36.7 | 63.5% | 83.5% |
| 20 | 87.5% | 34.2 | 68.0% | 85.5% |
| 50 | 87.0% | 30.7 | 64.5% | 86.5% |

## Table VI - Ablation

| Configuration | Accuracy | MTTR (min) | ARR |
|---|---|---|---|
| Full Framework | 84.5% | 37.4 | 63.4% |
| Without AOKG | 78.2% | 38.7 | 70.6% |
| Without RCA Engine | 58.5% | 51.6 | 43.4% |
| Without Learning Engine | 82.3% | 38.4 | 62.0% |
| Without Remediation Planner | 85.2% | 57.0 | 34.9% |

## Safety threshold T sensitivity

| T | ARR | MTTR (min) | auto first-attempt failure rate |
|---|---|---|---|
| 0.3 | 70.0% | 35.6 | 20.4% |
| 0.4 | 63.4% | 37.4 | 15.8% |
| 0.5 | 56.0% | 40.2 | 13.5% |
| 0.6 | 45.9% | 42.2 | 12.5% |
| 0.7 | 24.6% | 46.8 | 12.0% |

_Runtime: 1363 s._
