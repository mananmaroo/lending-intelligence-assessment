from pathlib import Path
import json,math
import numpy as np
import polars as pl
import joblib
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent
rows=pl.read_parquet(R/'model/fact_prediction.parquet').to_dicts();artifact=joblib.load(R/'model/booking_logistic.joblib')
N=np.array([[np.nan if r[k] is None else r[k] for k in artifact['numeric_features']] for r in rows],float)
C=np.array([[r[k] for k in artifact['categorical_features']] for r in rows],object)
X=np.column_stack([(np.where(np.isnan(N),artifact['median'],N)-artifact['mean'])/artifact['std'],np.isnan(N).astype(float),artifact['encoder'].transform(C)])
p=artifact['model'].predict_proba(X)[:,1]
assert np.allclose(p,[r['score'] for r in rows])
for r in rows:assert abs(1/(1+math.exp(-(r['base_logit']+sum(r['contributions'].values()))))-r['score'])<1e-10
m=json.loads((R/'model/booking_model.json').read_text());test=[r for r in rows if r['split']=='Test'];assert abs(roc_auc_score([r['converted'] for r in test],[r['score'] for r in test])-next(x['roc_auc'] for x in m['metrics'] if x['split']=='Test'))<1e-12
assert not set(r['user_id'] for r in rows if r['split']=='Train')&set(r['user_id'] for r in test)
assert all(r['latest_feature_time']<=r['prediction_time'] for r in rows)
print('Passed: saved-model reload, predictions, contribution additivity, held-out AUC, disjoint users, event cutoff.')
