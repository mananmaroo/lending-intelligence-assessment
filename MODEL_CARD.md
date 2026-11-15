# Booking logistic regression v2

This is an actually fitted model on synthetic data. It replaces the prior fixed-coefficient illustration. GenAI evaluation scores remain simulated.

## Target and eligibility

One row per lead, predicting a booked loan after the scoring point and within the next seven days. Scoring occurs 200 seconds after session start, before acceptance (220 seconds) and booking (300 seconds). The sample contains 498 leads, with at most one booked loan per lead. There are no real customer records.

## Features

| Input | Definition available at score time |
| --- | --- |
| Credit score | Explicit synthetic numeric score within the source credit bracket; saved in raw/credit_scores.jsonl |
| Requested amount | Origination request |
| Interest rate | Mean original quoted APR of offers created before scoring; missing if no offer |
| Term | First original quoted term; requested term when no offer exists |
| Campaign source | Original UTM source |
| Email variant | A/B for email touches; Not sent for other channels; distinct from randomized offer-copy variant |
| Session duration | Last observed pre-cutoff event timestamp minus session start; not completed session duration |
| Offer views | Deduplicated offer-view events before cutoff |
| MCP tool usage | Tool-call count from conversation turns before cutoff |
| Conversation sentiment | Supplied user-turn sentiment: negative, neutral, or no_conversation |

The model excludes acceptance, bookings, disbursed amounts, price changes, unsubscribe actions, escalation outcomes and synthetic evaluation labels. For real deployment the same event-time cutoff must be enforced at ingestion, including late-arrival policies.

## Training and validation

- Train: Oct 1–5, 102 leads (34 booked).
- Validation: Oct 13–14, 45 leads (15 booked).
- Test: Oct 22–23, 41 leads (18 booked).
- Purged from fitting/evaluation: 310 intervening leads, retained for browsing and scored with the fitted model.
- Seven-day gaps ensure every preceding cohort's full outcome window closes before the next evaluation cohort. User IDs are disjoint.
- Median imputation, missing indicators, standardization and category encoding fit on training data only.
- Logistic regression with L2 regularization; C selected from 0.1, 1 and 10 by validation ROC-AUC. Selected C = 10. No test-based selection or refitting.
- Fixed classification threshold: 0.5.

## Executed held-out results

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.702899 |
| Precision | 0.500000 |
| Recall | 0.222222 |
| Brier score | 0.221990 |
| True positives / false positives | 4 / 4 |
| True negatives / false negatives | 19 / 14 |

The threshold misses 14 of 18 bookings. This small synthetic test sample does not support a deployment recommendation. ROC-AUC uses probabilities; precision and recall use the fixed threshold. Filtered slices recompute all three from the same held-out leads; undefined denominators or one-class AUC display unavailable, not zero. Model parameters never retrain when a dashboard filter changes.

## Explanation

Per-lead contributions are coefficient × transformed feature value, grouped back to the ten original features. Missingness terms are grouped with their original field. The fitted intercept plus the sum of contributions equals the decision logit; sigmoid gives the predicted probability. Additivity is tested for every lead. Global importance is the mean absolute grouped contribution within the selected population.

These terms are not SHAP values or causal explanations. Correlated campaign/email variables and standardized inputs affect coefficient interpretation. Positive contributions increase log-odds relative to the fitted intercept, not necessarily relative to a population-average customer.

## Reproduction

Install requirements.txt; run `python build_data.py`, `python train_booking.py`, then `python test_model.py`. The trainer writes the model, encoder and preprocessing constants to model/booking_logistic.joblib, a readable summary to model/booking_model.json, per-lead rows to model/fact_prediction.parquet and the refreshed dashboard snapshot. Only load the locally generated trusted joblib artifact.

Python tests reload the saved model, reproduce predictions, verify additivity, verify AUC and check event cutoffs and disjoint users. Dashboard calculations are independently checked against sklearn metrics.

Metric references: [ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html), [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
