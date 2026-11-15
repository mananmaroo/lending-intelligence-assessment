# Prototype verification

- Generated 600 unique users, 498 leads, 605 offers, 540 shown offers, 181 booked users and 378 traces.
- Eight pipeline assertions passed: session identity, booking join cardinality, exposure references, replay removal, malformed JSON quarantine, assignment uniqueness, user grain and explanation additivity.
- Independently recalculated the two-sided Fisher exact p-value using integer combinations: 0.016433653418818646, matching the dashboard's 0.0164.
- Browser checks covered every dashboard tab, workflow-to-conversion navigation, campaign Search → All (64 → 181 converted users), credit-bracket filtering, lead selection and an information-button disclosure.
- Desktop and 390px mobile renders inspected; mobile document width stayed 390px. Wide evidence tables scroll inside their panels.
- No browser page errors occurred during the interaction run.
- Final build passed authored-content verification and is marked complete.

## Trained-model release verification

- Trained and saved a logistic regression pipeline with ten requested feature groups; a reload reproduces predictions and additive per-lead log-odds contributions.
- Purged chronological holdout: 41 test leads, ROC-AUC 0.7029, precision 0.5000, recall 0.2222 at threshold 0.5. All features precede the 200-second scoring cutoff and all seven-day labels mature before the next split.
- Independent JavaScript metric calculations match scikit-learn; empty denominators, date/campaign scope and marketing reconciliation checks passed.
- Final browser checks passed date selection (latest seven days: 152 leads), all-date reset (498 leads), repeated-sentence drilldown to trace T427, campaign filters, unassisted empty state, information disclosures and mobile layout without document overflow.
- GitHub-connected Vercel production deployment reached READY. Authenticated hosted manifest returned HTTP 200 and matched the locally verified HTML and snapshot hashes.
- The most repeated user sentence is “What interest rate can I get?” across 138 conversations. The GenAI view exposes example traces and tool-call evidence.

All data and labels are synthetic. This prototype does not establish real-world model performance or causal effects. GenAI evaluation uses explicit fixture labels and rule-based evidence, not live LLM judging. Polars processing and training run offline; Vercel serves the generated interactive snapshot.
