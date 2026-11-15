# Lending intelligence assessment

Interactive lending dashboard with a reproducible Polars pipeline, trained logistic regression booking model, campaign analytics, A/B experimentation and conversation/tool evidence.

All records are synthetic. The booking model is genuinely fitted; GenAI evaluation labels are simulated.

## Dashboard

Views: Workflow, Conversion, Marketing, Lead explorer, A/B experiments, ML explainability, GenAI evaluation and Data pipeline. Filters: date range, UTM source, campaign, variant, credit bracket and MCP assistance. Every section includes a plain-language information button.

The six lead cards show total leads, leads receiving offers, booked loans, lead-to-booking conversion, total disbursed principal and MCP-assisted conversations. Marketing shows allocated campaign spend, leads, bookings, conversion, CPL, CPB, offer views, ad views/clicks and principal minus ad spend (not profit).

## Model

Ten inputs: credit score, requested amount, interest rate, term, campaign source, email variant, pre-cutoff session duration, offer views, MCP tool calls and conversation sentiment. Features are fixed before acceptance or booking. Train/validation/test cohorts are chronological with seven-day label-window gaps.

Held-out test: 41 leads; ROC-AUC 0.703, precision 0.500, recall 0.222 at threshold 0.5. Small synthetic sample; not deployment evidence. See [MODEL_CARD.md](MODEL_CARD.md).

## Reproduce data and training

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_data.py
.venv/bin/python train_booking.py
.venv/bin/python test_model.py
```

Raw JSONL is in `raw/`; cleaned Parquet tables and the fitted model are in `model/`. The readable model summary contains preprocessing and coefficients. Load only this trusted generated joblib file.

## Frontend and hosting

`dashboard/src/content/` contains the authored React and CSS; `dashboard/src/data.json` contains reviewed data. The Data plugin build helper compiles the application using its bundled runtime. Compiled, integrity-checked assets are committed in `public/`, so Vercel needs no private build package, Python service or credentials to serve the dashboard.

```sh
npm run build
python3 -m http.server 4173 --directory public
```

Vercel deploys `public/` from `main`. The build step checks hashes and completion status; it does not retrain or silently rebuild the reviewed UI. After changing data or authored React, use the installed Data app build helper, copy the complete verified `dist/` output to `public/`, and commit the new manifest and snapshot together. Do not edit compiled HTML or snapshot bytes without regenerating their matching manifest.

See [ASSESSMENT.md](ASSESSMENT.md) for the design and rubric. No production warehouse, scheduled refresh, live inference endpoint or live LLM judge is connected.
