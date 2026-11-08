# Traceability: paper / patent → code → tests

| Paper / patent element | Code | Tests |
|---|---|---|
| Monitoring layer (110), event vector E = {L, M, Q, D, C} | `layers.MonitoringLayer`, `models.EventVector` | `test_framework::test_monitoring_layer_builds_event_vector` |
| Context acquisition layer (120): runbooks, incidents, topology | `layers.ContextAcquisitionLayer`, `ASHDPF.bootstrap` | `test_framework::test_context_layer_seeds_runbook_weights` |
| AOKG G = (V, E), 8 node types, 5 edge types (FIG. 2, claim 3) | `knowledge_graph.NodeType`, `EdgeType`, `AdaptiveKnowledgeGraph` | `test_knowledge_graph::test_all_patent_node_and_edge_types_exist` |
| Graph population per incident | `AdaptiveKnowledgeGraph.ingest_incident` | `test_knowledge_graph::test_ingest_incident_populates_graph` |
| **Eq. (1)** W_e ← W_e + αR | `AdaptiveKnowledgeGraph.update_edge_weight`, `record_fix_outcome` | `test_equation_1_*`, `test_record_fix_outcome_counts_and_applies_eq1` |
| **Eq. (2)** RC = argmax P(RC_i \| F) | `RCAEngine.analyze` | `test_rca::test_engine_argmax_and_components` |
| Graph reasoning over dependency chains (310) | `rca.GraphReasoner` | `test_graph_reasoner_detects_cascading_dependency_failure` |
| Historical similarity: log + metadata + graph neighbourhood (320) | `rca.HistoricalMatcher` | `test_historical_matcher_retrieves_identical_incident` |
| ML classification: RF / gradient boosting (330, claim 4) | `rca.MLClassifier` | `test_classifier_learns_separable_classes[*]` |
| **Eq. (3)** C_RCA = β1G + β2H + β3ML | `rca.combine` | `test_equation_3_weighted_sum_is_exact`, `test_combine_renormalises_*` |
| Candidate actions A = {A1..An} (400) | `remediation.ActionLibrary`, `DEFAULT_ACTIONS` | `test_library_filters_by_category` |
| **Eq. (4)** RES = γ1S + γ2H + γ3R (claim 5) | `remediation.remediation_effectiveness`, `RemediationPlanner.plan` | `test_equation_4_exact`, `test_planner_prefers_historically_successful_action` |
| Safety layer: dependency impact, rollback, governance, threshold T (claim 6) | `safety.SafetyValidationLayer` | `test_remediation_safety::test_*` (8 tests) |
| Below T → human approval (claim 11) | `ASHDPF.handle` | `test_low_confidence_routes_to_human_approval` |
| Execution via connectors, audit history, rollback checkpoints (claim 7) | `execution.ExecutionFramework`, `Connector` | `test_execution_creates_audit_and_checkpoint`, `test_rollback_invokes_connector` |
| Validation module: success / partial / failure | `execution.ValidationModule` | `test_validation_module` |
| **Eq. (5)** R_t = 1 / MTTR (claims 2, 10) | `learning.reward` | `test_reward_is_inverse_mttr` |
| **Eq. (6)** π_{t+1} = π_t + η∇J(π) | `learning.SoftmaxPolicy.update` (REINFORCE) | `test_policy_gradient_step_matches_reinforce`, `test_policy_converges_to_fastest_action` |
| Knowledge evolution after every cycle (550, 560) | `learning.ContinuousLearningEngine.learn` | `test_cle_updates_policy_graph_and_repository` |
| Knowledge Reuse Index (claims 8, 12) | `metrics.knowledge_reuse_index`, `ASHDPF.report` | `test_mttr_arr_kri`, `test_report_contains_governance_metrics` |
| Metrics: accuracy, P, R, F1, MTTR, ARR | `metrics` | `test_classification_report_known_values` |
| Synthetic dataset (500 pipelines, 50 groups, 8 categories) | `simulation.SyntheticEnvironment` | `test_simulation_integration::test_environment_*`, `test_events_*` |
| Baselines (monitoring-only, rule-based, AIOps) | `baselines` | `test_proposed_beats_all_baselines_on_mttr` |
| Experiments 1–5, ablation, t-test / Wilcoxon / ANOVA | `experiments.run_all` | `test_experiment_protocol_quick` |

## Design decisions not fixed by the documents

* **γ3 is negative** (default −0.30) so that risk *lowers* RES; the paper writes `+γ3R` with R a risk score.
* **Gate confidence** = P(diagnosed category) × W_e(runbook→fix), i.e. "the diagnosis is right" × "this fix works for it". T defaults to 0.40, with **0.90 for credential incidents** (governance-sensitive).
* **Reward scale**: R_t = κ / MTTR with κ = 10 so gradients are not vanishingly small; partial success earns half, failure zero.
* **Edge weights are clipped to [0, 1]** after Eq. (1).
* **XGBoost / GNN**: scikit-learn's histogram gradient boosting stands in for XGBoost to keep dependencies light; no GNN is included. Both plug in via `MLClassifier`.
* **Cold start**: until the classifier has enough labels, runbook keyword signatures replace the ML term in Eq. (3).
* **Static graph** (Exp. 3) = built from the historical repository once, then frozen. "Without learning engine" = static graph + no policy updates + no retraining.
