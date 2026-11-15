"""Train a small reproducible logistic booking model; only synthetic fixture data."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,random,math
import polars as pl
import numpy as np
import sklearn,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score,precision_score,recall_score,roc_curve,confusion_matrix,brier_score_loss
R=Path(__file__).resolve().parent;M=R/'model'; raw=R/'raw'
read=lambda n:pl.read_ndjson(raw/f'{n}.jsonl').to_dicts()
users=pl.read_parquet(M/'fact_user_funnel.parquet').to_dicts()
leads={x['user_id']:x for x in read('leads')};touches={x['session_id']:x for x in read('marketing')}
starts={x['session_id']:datetime.fromisoformat(x['timestamp']) for x in read('marketing')}
identity={x['session_id']:x['user_id'] for x in users}
events={};seen=set()
for e in read('events'):
 if e['event_id'] in seen:continue
 seen.add(e['event_id'])
 try:json.loads(e['metadata'])
 except ValueError:continue
 u=identity.get(e['session_id']);events.setdefault(u,[]).append(e)
turns={}
for t in read('conversations'):turns.setdefault(identity.get(t['session_id']),[]).append(t)
offers={}
for o in read('offers'):offers.setdefault(o['lead_id'],[]).append(o)
bookings={b['offer_id']:b for b in read('bookings')}
rng=random.Random(8891);credits=[];delivery=[];features=[]
for u in sorted(users,key=lambda x:x['user_id']):
 uid=u['user_id'];start=starts[u['session_id']];cutoff=start+timedelta(seconds=200)
 marketing=touches[u['session_id']]
 score={'Fair':630,'Good':710,'Excellent':790}[u['credit_score_bracket']]+rng.randrange(-24,25)
 credits.append(dict(user_id=uid,credit_score=score,as_of=start.isoformat(),provenance='synthetic within-bracket score'))
 delivery.append(dict(session_id=u['session_id'],timestamp=(start-timedelta(seconds=1)).isoformat(),ad_views=rng.randrange(2,9),ad_clicks=1,provenance='synthetic allocated acquisition touch delivery'))
 pre=[e for e in events.get(uid,[]) if datetime.fromisoformat(e['timestamp'])<=cutoff]
 conv=[t for t in turns.get(uid,[]) if datetime.fromisoformat(t['timestamp'])<=cutoff]
 calls=sum(len(t['tools_called']) for t in conv)
 sentiment='negative' if any(t['sentiment']=='negative' for t in conv if t['role']=='user') else 'neutral' if conv else 'no_conversation'
 duration=(max(datetime.fromisoformat(e['timestamp']) for e in pre)-start).total_seconds()
 u.update(utm_source=marketing['utm_source'],email_variant=marketing['email_variant'] or 'Not sent',credit_score=score,mcp_assisted='Assisted' if calls else 'Unassisted',ad_views=delivery[-1]['ad_views'],ad_clicks=1,session_duration=duration,mcp_tool_usage=calls,conversation_sentiment=sentiment)
 if uid not in leads:continue
 lead=leads[uid];os=[o for o in offers.get(lead['lead_id'],[]) if datetime.fromisoformat(o['timestamp'])<=cutoff]
 booking_times=[datetime.fromisoformat(bookings[o['offer_id']]['booking_timestamp']) for o in os if o['offer_id'] in bookings]
 assert all(t>cutoff for t in booking_times)
 latest=max([datetime.fromisoformat(e['timestamp']) for e in pre]+[datetime.fromisoformat(t['timestamp']) for t in conv]+[datetime.fromisoformat(o['timestamp']) for o in os])
 assert latest<=cutoff
 # Offer rate = mean original quoted APR; term = first original quoted term, or requested term if no quote.
 rate=sum(o['interest_rate'] for o in os)/len(os) if os else None
 term=sorted(os,key=lambda x:x['timestamp'])[0]['term_months'] if os else u['term_months']
 label=int(any(cutoff<t<=cutoff+timedelta(days=7) for t in booking_times))
 assert label==u['converted']
 date=u['date'];split='Train' if date<'2026-10-06' else 'Validation' if '2026-10-13'<=date<'2026-10-15' else 'Test' if date>='2026-10-22' else 'Purged'
 features.append(dict(**{k:u[k] for k in ['user_id','lead_id','campaign_id','utm_source','variant','credit_score_bracket','mcp_assisted','date']},credit_score=score,requested_amount=lead['requested_amount'],interest_rate=rate,term_months=term,campaign_source=marketing['utm_source'],email_variant=u['email_variant'],session_duration=duration,offer_views=sum(e['event_type']=='offer_viewed' for e in pre),mcp_tool_usage=calls,conversation_sentiment=sentiment,converted=label,prediction_time=cutoff.isoformat(),latest_feature_time=latest.isoformat(),split=split))
for name,rows in [('credit_scores',credits),('marketing_delivery',delivery)]:
 (raw/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
features.sort(key=lambda r:r['prediction_time'])
num=['credit_score','requested_amount','interest_rate','term_months','session_duration','offer_views','mcp_tool_usage'];cat=['campaign_source','email_variant','conversation_sentiment']
labels=['Credit score','Requested amount','Interest rate','Term (months)','Session duration','Offer views','MCP tool usage','Campaign source','Email variant','Conversation sentiment']
N=np.array([[np.nan if r[k] is None else r[k] for k in num] for r in features],float);C=np.array([[r[k] for k in cat] for r in features],object);y=np.array([r['converted'] for r in features]);splits=np.array([r['split'] for r in features]);tr=splits=='Train';va=splits=='Validation';te=splits=='Test'
assert all(len(set(y[mask]))==2 for mask in [tr,va,te]);assert len(set(r['user_id'] for r in features))==len(features)
# Purge seven-day label windows before each subsequent evaluation period.
for left,right in [('Train','Validation'),('Validation','Test')]:assert max(datetime.fromisoformat(r['prediction_time'])+timedelta(days=7) for r in features if r['split']==left)<min(datetime.fromisoformat(r['prediction_time']) for r in features if r['split']==right)
median=np.nanmedian(N[tr],axis=0);imputed=np.where(np.isnan(N),median,N);mean=imputed[tr].mean(axis=0);std=imputed[tr].std(axis=0);std=np.where(std==0,1,std)
encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(C[tr]);encoded=encoder.transform(C)
X=np.column_stack([(imputed-mean)/std,np.isnan(N).astype(float),encoded]);groups=list(range(7))+list(range(7))+[7+i for i,cats in enumerate(encoder.categories_) for _ in cats]
models=[]
for reg in [.1,1,10]:
 model=LogisticRegression(C=reg,max_iter=2000,random_state=4271).fit(X[tr],y[tr]);auc=roc_auc_score(y[va],model.predict_proba(X[va])[:,1]);models.append((auc,reg,model))
validation_auc,reg,model=max(models,key=lambda x:(x[0],-x[1]));prob=model.predict_proba(X)[:,1];threshold=.5
contributions=X*model.coef_[0];grouped=np.zeros((len(features),10))
for j,g in enumerate(groups):grouped[:,g]+=contributions[:,j]
assert np.allclose(model.intercept_[0]+grouped.sum(axis=1),model.decision_function(X))
assert np.allclose(1/(1+np.exp(-(model.intercept_[0]+grouped.sum(axis=1)))),prob)
metrics=[]
for split in ['Train','Validation','Test']:
 mask=splits==split;yy=y[mask];pp=prob[mask];pred=pp>=threshold;tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel();metrics.append(dict(split=split,leads=int(mask.sum()),booked=int(yy.sum()),roc_auc=float(roc_auc_score(yy,pp)),precision=float(precision_score(yy,pred,zero_division=0)),recall=float(recall_score(yy,pred,zero_division=0)),brier=float(brier_score_loss(yy,pp)),threshold=threshold,tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),start=min(r['date'] for r in features if r['split']==split),end=max(r['date'] for r in features if r['split']==split)))
for i,r in enumerate(features):r.update(score=float(prob[i]),base_logit=float(model.intercept_[0]),contributions={labels[j]:float(grouped[i,j]) for j in range(10)},model_version='trained-logistic-v2',threshold=threshold)
joblib.dump(dict(model=model,encoder=encoder,numeric_features=num,categorical_features=cat,median=median,mean=mean,std=std,threshold=threshold,group_indices=groups),M/'booking_logistic.joblib')
(M/'booking_model.json').write_text(json.dumps(dict(version='trained-logistic-v2',sklearn_version=sklearn.__version__,C=reg,threshold=threshold,intercept=float(model.intercept_[0]),coefficients=model.coef_[0].tolist(),numeric_features=num,categorical_features=cat,categories=[x.tolist() for x in encoder.categories_],numeric_medians=median.tolist(),numeric_means=mean.tolist(),numeric_scales=std.tolist(),metrics=metrics,excluded_gap_leads=int((splits=='Purged').sum())),indent=2))
pl.DataFrame(features).write_parquet(M/'fact_prediction.parquet');pl.DataFrame(users).write_parquet(M/'fact_user_funnel.parquet')
s=json.loads((R/'dashboard/src/data.json').read_text());s['buildStatus']='updating';s['generatedAt']=datetime.now(timezone.utc).isoformat();s['queries']['users']['rows']=users;s['queries']['predictions']['rows']=features
for q in s['queries'].values():q['source']['caveats']=['Synthetic dataset. Logistic regression is trained locally; GenAI evaluation labels remain simulated.']
s['queries']['predictions']['source'].update(label='Trained logistic regression on synthetic lending data',files=['raw/leads.jsonl','raw/offers.jsonl','raw/bookings.jsonl','raw/events.jsonl','raw/conversations.jsonl','raw/marketing.jsonl','raw/credit_scores.jsonl','train_booking.py'],metricDefinitions=[dict(label='Prediction point',definition='200 seconds after session start; activity and original quotes available by that cutoff only. Excludes acceptance, booking, price changes, unsubscribe, escalation outcomes and evaluation labels.'),dict(label='Per-lead contributions',definition='Grouped coefficient × transformed-feature terms in log-odds. Intercept plus all terms reconstructs the probability through sigmoid. Missing-rate indicator is grouped with rate. These are not SHAP or causal effects.'),dict(label='Evaluation',definition='Chronological Train Oct1–5, Validation Oct13–14, Test Oct22–23; intervening leads purged to keep seven-day outcome windows mature before the next split. Select C on validation AUC; fixed 0.5 threshold. Test never used for fitting or model selection.')],evidenceFlow=[dict(title='Reproduce',detail='Run build_data.py then train_booking.py. Train-only preprocessing and logistic fit; validation-only C selection; score held-out test; verify contribution additivity.')])
s['queries']['model_metrics']=dict(rows=metrics,source=dict(label='Executed model evaluation',files=['model/booking_model.json','train_booking.py'],metricDefinitions=[dict(label='ROC-AUC',definition='Ranking discrimination on positive-class probability scores; 0.5 is chance.'),dict(label='Precision and recall',definition='At fixed threshold 0.5: precision TP/(TP+FP), recall TP/(TP+FN). Undefined denominators display unavailable for filtered subsets.')],caveats=['Small synthetic test cohort; not real-world performance evidence.']))
s['queries']['users']['source']['files']+=['raw/credit_scores.jsonl','raw/marketing_delivery.jsonl','train_booking.py']
s['queries']['users']['source']['metricDefinitions'] += [dict(label='Marketing scope',definition='Allocated acquisition-cohort spend, views and clicks; one click per acquired session plus synthetic attributed impressions. Not whole-campaign reach. CPL = spend/leads; CPB = spend/bookings; conversion = booked leads/leads; amount minus spend is disbursed principal less ads, not revenue, profit or ROI.'),dict(label='MCP assisted',definition='At least one MCP tool call by the prediction cutoff. Association only; not randomized assistance.')]
# Denormalize filter dimensions onto source queries so the shared inspector has matching scope.
dims={u['user_id']:{k:u[k] for k in ['date','campaign_id','utm_source','variant','credit_score_bracket','mcp_assisted']} for u in users}
for name in ['conversations','journey']:
 for r in s['queries'][name]['rows']:r.update(dims[r['user_id']])
traceusers={c['trace_id']:c['user_id'] for c in s['queries']['conversations']['rows']}
for t in s['queries']['turns']['rows']:t.update(dims[traceusers[t['trace_id']]])
s['filters']=[dict(id='date',label='Date range',field='date',type='date',mode='through',defaultValue='2026-10-01..2026-10-23',queryIds=['users','predictions','conversations','journey','turns'])]+[dict(id=k,label=label,field=field,defaultValue='all',queryIds=['users','predictions','conversations','journey','turns']) for k,label,field in [('utm','UTM source','utm_source'),('campaign','Campaign','campaign_id'),('variant','Variant','variant'),('credit','Credit bracket','credit_score_bracket'),('assistance','MCP assistance','mcp_assisted')]]
s['queries']['checks']['rows']=[r for r in s['queries']['checks']['rows'] if r['check']!='Model additivity']+[dict(check='Trained model additivity',status='Passed',detail='Grouped trained coefficients reconstruct every lead probability.'),dict(check='Feature cutoff',status='Passed',detail='All input events precede the 200-second prediction point; every booking follows it.'),dict(check='Purged temporal split',status='Passed',detail='Seven-day label windows close before the next split; no shared users; train-only preprocessing.')]
(R/'dashboard/src/data.json').write_text(json.dumps(s,indent=2));(R/'trained_snapshot.json').write_text(json.dumps(s,indent=2))
print(json.dumps(dict(metrics=metrics,C=reg,purged=int((splits=='Purged').sum()),features=num+cat),indent=2))
