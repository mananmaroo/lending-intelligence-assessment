# Current prototype revision

The booking model is now trained logistic regression v2. See MODEL_CARD.md for the current features, temporal splits, executed metrics and limitations. The original fixed-coefficient model described in the historical design below has been replaced.

Dashboard additions: six funnel cards; date, UTM source, campaign, variant, credit and MCP-assistance filters; a marketing view with spend, CPL, CPB, offers, attributed views/clicks and disbursed principal less ad spend; repeated sentences and trace-linked tool evidence. “Conversation” in funnel/marketing requests is interpreted as conversion; actual assistant conversations are counted separately.

Marketing delivery records are synthetic and allocated to acquired sessions, not a complete reach report. Views and clicks are not inferred from offer counts. Assisted vs unassisted comparisons are observational and do not establish the impact of assistance.

---

# Lending intelligence — technical assessment prototype

## What is implemented

An interactive local dashboard with seven views, campaign filtering, dependent lead selection, source inspection, and inspectable generated data. This is a separate expanded fixture; the original 30-user package is preserved.

The expanded fixture contains 600 authenticated users, 498 leads, 605 offers created, 540 offer-view events, 181 booked users and 378 conversation traces. All data are synthetic. Cohort entry runs October 1–23, 2026; all seven-day outcomes are mature by November 1. No currency is assumed: amounts and allocated spend use demo currency units.

| View | Question | Interaction and evidence |
| --- | --- | --- |
| Workflow | How does the system fit together? | Clickable source → Polars → unified model → three analytical branches → dashboard |
| Conversion | How many users converted and how many offers were shown? | Campaign filter; unique-user funnel; daily acquisition cohorts; campaign outcomes and allocated spend |
| Lead explorer | What happened to this customer? | Dependent lead selector and selectable lead table; event timeline; linked conversations |
| A/B experiments | Is offer copy B associated with higher booking conversion in the randomized fixture? | Campaign cut; intention-to-treat denominators; effect interval; exact test; allocation check |
| ML explainability | Why did this lead receive its score? | Lead-level additive contributions; input values; campaign-level feature importance and calibration |
| GenAI evaluation | Did the assistant use tools and answer successfully? | Campaign-level evaluation; reference-intent breakdown; trace selector with full turns, tool result and retrieved context |
| Data pipeline | How were these metrics produced and checked? | Source-to-model flow, executed quality assertions and dimensional join contracts |

Campaign filters apply to the active analytical view. Lead selection affects individual details and local model explanations, not experiment populations or campaign-wide model diagnostics. Pipeline checks always cover the full run. Unknown sessions and absent conversations must be explicit in a production system. The mock fixture has one authenticated session and one pre-assignment marketing touch per user; it does not demonstrate probabilistic cross-device attribution.

## Metric contracts

- **Converted users:** distinct assigned users with a booking within seven days of assignment, regardless of offer exposure. All generated bookings fall inside that window.
- **Conversion rate:** converted users / assigned authenticated users. Anonymous-only visitors are not included in this fixture's denominator.
- **Offers created:** distinct offer IDs in origination.
- **Offers shown:** deduplicated offer-view events carrying a valid offer ID. The fixture has one view per shown offer; in production report impressions and unique shown offers separately.
- **Shown users:** distinct users with one or more offer views.
- **Campaign:** one known pre-assignment session touch per user. For multi-touch data, retain touch history and declare a bounded attribution rule. Do not infer assignment from post-treatment campaign behavior.
- **Spend:** allocated touch-level spend, summed once. Campaign-day budgets must be stored at campaign-day grain rather than repeated per user.
- **Form-error users:** users with a pre-assignment-to-copy form error / assigned users. This is a baseline balance diagnostic, not a treatment outcome.

## Experiment design

`offer_copy_v1` randomizes authenticated users 50:50 before offer exposure. A is baseline copy and B is alternate copy. The generator uses an independent pseudorandom draw for assignment; it then simulates outcomes. Assignment is immutable and includes non-exposed and non-converting users.

Primary estimand: seven-day booking risk difference B − A, intention-to-treat. Dashboard calculations include Wilson intervals for arm rates, a Newcombe difference interval, a two-sided Fisher exact test, and a two-sided exact binomial sample-ratio test against 50:50 (diagnostic threshold 0.01). Campaign subgroup results are exploratory and unadjusted. There is no peeking-adjusted sequential test and no rollout recommendation.

Candidate extension: preregister baseline conversion, practical minimum effect, power, alpha, sample size and fixed end date. Add a post-treatment guardrail such as complaint rate or abandonment without conditioning on exposure. Verify assignment before exposure, no cross-variant contamination, complete outcome windows, and experiment-level deduplication. Missing outcomes are not automatically failures.

## ML explanation design

The prototype uses an explicitly **untrained illustrative logistic model**:

`p = sigmoid(-0.15 + credit contribution + amount contribution + prior-form-error contribution)`.

All three features are available before offers and bookings. Local terms reconstruct the score exactly in log-odds space. Global importance is mean absolute contribution. These are not SHAP values and do not establish causality. Calibration is descriptive on the generated cohort, not a held-out model performance claim.

Candidate extension: define the prediction point and seven-day label; implement event-time feature snapshots; train using temporal train/validation/test splits and grouped user separation. Compare against a baseline using ROC-AUC, PR-AUC, Brier score and calibration with uncertainty. Use a suitable explanation method (e.g. SHAP for the selected estimator), verify additivity, report background data and output scale, examine feature leakage and drift, and distinguish booking propensity from credit eligibility. Do not infer demographic fairness without lawful, appropriate group data.

## GenAI evaluation design

Logs preserve trace/session/turn identity, role, content, context and nested tool calls. Intent uses a transparent keyword baseline compared with synthetic reference intent. Tool counts and latency come from logs; p95 uses nearest rank. Correctness, groundedness and task success are **simulated rubric labels**, not human reviews or live LLM-judge results. They illustrate dashboard behavior only. The synthetic intent baseline matches templated language and is not a generalization result.

Candidate extension: version a held-out evaluation set with expected intent, expected tool and argument schema, expected answer and permitted retrieval sources. Define separate criteria:

1. Intent classification: confusion matrix and macro F1, including unknown intent.
2. Tool execution: expected tool/arguments, success/error status, latency, retries and unnecessary calls.
3. Groundedness: claims supported by retrieved evidence, with cited spans.
4. Correctness: rubric against reference answer, including permissible abstention.
5. Task success: end-to-end user goal completion, independent from sentiment.
6. Safety: prompt-injection, data leakage and inappropriate-action cases.

Store judge model/prompt/version, run ID, per-case rationale, and human adjudication. Report disagreement and inter-rater agreement on a reviewed subset; isolate judge failures from answer failures. Add token usage and actual provider pricing before claiming cost metrics.

## Pipeline and dimensional design

`raw sources → immutable landing → schema/type checks → deduplication + quarantine → event-time identity bridge → dimensional facts → semantic metrics + evaluation marts → dashboard snapshot`

Implemented source families:
- A: leads, offers and bookings.
- B: clickstream JSON metadata, parsed with Polars.
- C: conversation turns with context/tool lists.
- D: campaign touches and CRM email variants.
- Assessment extensions: randomized assignments, illustrative predictions and synthetic evaluation labels.

Dimensions: user, session and campaign. Facts: event, lead, offer, booking, conversation, assignment, prediction and marketing touch. Natural IDs are stable in this fixture; warehouse implementation should define surrogate keys, slowly changing dimensions and deletion policies where applicable. A production date dimension and run/audit dimension are also expected.

Joins use session_id → user_id, offer_id → lead_id → user_id, and trace_id → session_id. Never join all facts directly by user_id and then sum. Materialize each fact at its own grain and aggregate into one row per user only for the semantic funnel mart. Origination records lack session IDs, which remain unknown.

The current script is a deterministic local batch, not a deployed scheduler. Candidate orchestration should implement task dependencies, bounded retries, backfills, atomic output publication, idempotent reruns, incremental watermarks, late-event handling, schema drift checks and an audit run manifest. Publish only after quality gates pass. Keep failed rows and reasons in quarantine; do not silently drop them.

## Assessment tasks and rubric

Suggested scope: a take-home implementation followed by an engineering walkthrough. Adapt timing to candidate seniority; do not require production infrastructure for the prototype review.

| Area | Weight | Evidence expected |
| --- | ---: | --- |
| Ingestion and orchestration | 20 | Reproducible dependency graph, idempotence, retries/backfills, structured failures |
| Data quality and dimensional model | 20 | Grains, temporal identity, explicit join cardinalities, quarantine, reconciliation |
| Analytics and experimentation | 20 | Correct unique-user denominators, exposure distinction, mature cohorts, uncertainty, SRM and no unsupported causal claims |
| ML and explainability | 15 | Leakage-safe features, temporal validation, calibrated metrics, faithful local/global explanations |
| GenAI evaluation | 15 | Versioned cases and rubric, tool/answer separation, evidence-backed judgments, failure analysis |
| Dashboard and communication | 10 | Consistent filters, usable drill-down, source lineage, explicit empty states and limitations |

Acceptance checks should include replayed input, conflicting identities, malformed JSON, orphan offers/bookings, duplicate assignment, late bookings, zero-denominator campaigns, absent conversation traces, treatment imbalance and a failed tool call. Reconcile exports and visible metrics against the same filtered evidence. Hard failures include fabricated evaluations, outcome leakage and many-to-many joins inflating conversions.

## Run and handoff

1. Install `polars==2.1.0` in a Python environment.
2. Run `python build_data.py` to regenerate raw JSONL, Parquet facts, snapshot and validation results.
3. The dashboard's current reviewed snapshot is `dashboard/src/data.json`. To refresh it, copy the generated snapshot while preserving the existing app `id`, then rebuild using the installed Data app build helper. Set `buildStatus` to `complete` after verification.
4. Serve `dashboard/dist` over localhost HTTP; the compiled application loads its adjacent snapshot file.

The app uses the bundled Data runtime for charts, filters, source inspection and editing. It is a local prototype; no external system, production warehouse, hosted Site or live LLM is connected.

## Guardrails and client explanations

Each dashboard section has a “What this shows” information button for nontechnical readers. Summaries explain scope and each metric retains its definition tooltip.

The A/B view adds user type (New/Returning), baseline credit bracket and requested term_months filters, plus a segment comparison table. Requested term is captured before assignment; an accepted or renegotiated term must not be used as a causal subgroup. Variant sample sizes, rates, absolute uplift, 95% intervals and two-sided Fisher p-values recompute for the selected population. Subgroup p-values are exploratory and unadjusted.

| Guardrail | Definition |
| --- | --- |
| Lead-to-offer conversion | Users with at least one created offer / users with a lead |
| Average amount disbursed | Total disbursement / booked loans, in demo units |
| Acceptance rate | Users with an offer_accepted event / users shown an offer |
| Errors | Total form_error events |
| Form-error rate | Users with at least one form error / form users |
| MCP escalation rate | Escalated traces / all traces |
| Offer rejection rate | Users declining an offer / users shown an offer; not lender underwriting rejection |
| Unsubscribe rate | Email users with unsubscribe event / email users |
| Price-reduction request rate | Users requesting an option change to lower APR / users shown an offer |
| Price-reduction success rate | Users receiving a lower APR after an option change / requesting users |

Price changes include original/revised APR and revised term in raw JSON. Lower APR does not necessarily imply lower lifetime borrowing cost; a production price metric must state the pricing basis. All additional behavioral events are simulated, reproducible and separately recorded. A zero denominator displays a dash. Guardrails are descriptive; business thresholds and multiplicity-adjusted decision rules remain to be specified. Conditional exposed/accepted/borrower populations are post-treatment and must not be interpreted as unbiased treatment comparisons.
