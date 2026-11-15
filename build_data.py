"""Deterministic assessment fixture + Polars ingestion. No live lending or LLM calls."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import json,random,math,time
import polars as pl
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'; MODEL=ROOT/'model'; RAW.mkdir(exist_ok=True); MODEL.mkdir(exist_ok=True)
rng2=random.Random(918);rng=random.Random(4271); start=datetime(2026,10,1,tzinfo=timezone.utc)
tables={k:[] for k in ['leads','offers','bookings','events','conversations','marketing','assignments','predictions','evaluations']}
for i in range(600):
 u=f'U{i+1:04}'; s=f'S{i+1:04}'; l=f'L{i+1:04}'; c=['Search','Social','Email'][i%3]
 base=start+timedelta(minutes=i*55); ts=lambda t:(base+timedelta(seconds=t)).isoformat()
 user_segment=rng2.choice(['New','Returning']);requested_term=rng2.choice([12,24,36]);
 variant=rng.choice(['A','B']); errors=rng.choices([0,1,2],[.68,.24,.08])[0]; credit=rng.choice(['Fair','Good','Excellent']); amount=rng.randrange(40,181)*1000
 contrib={'Credit bracket':{'Fair':-.65,'Good':.1,'Excellent':.65}[credit], 'Requested amount':-(amount-100000)/150000, 'Form friction':-.35*errors}
 score=1/(1+math.exp(-(-.15+sum(contrib.values()))))
 haslead=rng.random()<.82; hasoffer=haslead and rng.random()<.86; shown=hasoffer and rng.random()<.91
 booked=shown and rng.random()<min(.9,score+(.09 if variant=='B' else 0)); assistant=rng.random()<.62
 tables['marketing'].append(dict(touch_id=f'M{i}',campaign_id=c,utm_source=c.lower(),utm_medium='email' if c=='Email' else 'paid',utm_campaign=c+' October',ad_spend=round(rng.uniform(1,6),2),user_id=None,session_id=s,timestamp=ts(0),email_variant=variant if c=='Email' else None))
 tables['assignments'].append(dict(user_id=u,session_id=s,variant=variant,experiment_id='offer_copy_v1',timestamp=ts(30),user_segment=user_segment,credit_bracket=credit,term_months=requested_term))
 def event(kind,t,metadata,auth=True):tables['events'].append(dict(event_id=f'E{i}_{len(tables["events"])}',timestamp=ts(t),session_id=s,user_id=u if auth else None,event_type=kind,metadata=json.dumps(metadata)))
 event('session_started',0,{'campaign_id':c},False);event('login',20,{});event('form_start',40,{})
 for j in range(errors):event('form_error',50+j,{'field':'income','duration_ms':900})
 if haslead:
  event('form_submit',90,{'duration_ms':50000+errors*10000})
  tables['leads'].append(dict(lead_id=l,user_id=u,timestamp=ts(100),credit_score_bracket=credit,requested_amount=amount))
  tables['predictions'].append(dict(lead_id=l,user_id=u,timestamp=ts(101),model_version='illustrative-logistic-v1',base_logit=-.15,score=score,credit=contrib['Credit bracket'],amount=contrib['Requested amount'],friction=contrib['Form friction']))
 if hasoffer:
  for j in range(1+(rng.random()<.38)):
   o=f'O{i}_{j}';tables['offers'].append(dict(offer_id=o,lead_id=l,timestamp=ts(120+j),interest_rate=.105+j*.01,term_months=requested_term+j*12,status='accepted' if booked and j==0 else 'expired'))
   if shown:event('offer_viewed',140+j,{'offer_id':o,'duration_ms':1200+j*200})
  if booked:tables['bookings'].append(dict(loan_id=f'B{i}',offer_id=f'O{i}_0',booking_timestamp=ts(300),disbursed_amount=amount))
 accepted=shown and (booked or rng2.random()<.15)
 rejected=shown and not accepted and rng2.random()<.3
 unsubscribed=c=='Email' and rng2.random()<.05
 price_requested=shown and rng2.random()<.22
 price_reduced=price_requested and rng2.random()<.55
 if accepted:event('offer_accepted',220,{'offer_id':f'O{i}_0'})
 if rejected:event('offer_rejected',220,{'offer_id':f'O{i}_0'})
 if unsubscribed:event('unsubscribed',400,{})
 if price_requested:event('price_reduction_requested',180,{'offer_id':f'O{i}_0','original_apr':.105})
 if price_reduced:event('price_reduced',210,{'offer_id':f'O{i}_0','original_apr':.105,'revised_apr':.095,'revised_term_months':requested_term+12})
 if assistant:
  intent=rng.choice(['rate_inquiry','eligibility','application_status']);fail=rng.random()<.12; grounded=rng.random()<.91; correct=grounded and rng.random()<.93
  escalated=(fail and rng2.random()<.7) or rng2.random()<.03
  if escalated:event('mcp_escalated',175,{'trace_id':f'T{i}'})
  trace=f'T{i}';latency=rng.randrange(450,2800); message={'rate_inquiry':'What interest rate can I get?','eligibility':'Am I eligible for a loan?','application_status':'What is my application status?'}[intent]
  tool={'rate_inquiry':'get_offers','eligibility':'check_eligibility','application_status':'get_application'}[intent]
  for turn,(role,content) in enumerate([('user',message),('assistant','Checking your request.'),('tool','Timeout' if fail else 'Records retrieved'),('assistant','Please retry shortly.' if fail else 'Here is the information from your application records.')]):
   tables['conversations'].append(dict(trace_id=trace,timestamp=ts(160+turn*3),session_id=s,turn_id=turn,role=role,content=content,sentiment='negative' if fail and turn==0 else 'neutral',context_retrieved=['policy_v1'] if grounded and role=='assistant' else [],tools_called=[dict(name=tool,status='error' if fail else 'success',latency_ms=latency)] if turn==1 else []))
  tables['evaluations'].append(dict(trace_id=trace,reference_intent=intent,grounded=int(grounded),correct=int(correct),task_success=int(not fail and correct),escalated=int(escalated),tool_error=int(fail),latency_ms=latency,judge_version='simulated-rubric-v1',rationale='Synthetic evaluator label; not a human or LLM judgment.'))
for row in tables['offers']:
 if any(e['event_type']=='offer_accepted' and json.loads(e['metadata']).get('offer_id')==row['offer_id'] for e in tables['events']):row['status']='accepted'
 elif any(e['event_type']=='offer_rejected' and json.loads(e['metadata']).get('offer_id')==row['offer_id'] for e in tables['events']):row['status']='rejected'
# Replay fixtures and one malformed payload exercise deduplication and quarantine.
tables['events'].extend([dict(x) for x in tables['events'][:7]])
tables['events'].append(dict(event_id='BAD',timestamp=start.isoformat(),session_id='UNKNOWN',user_id=None,event_type='form_error',metadata='{bad'))
for name,rows in tables.items():(RAW/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
read=lambda n:pl.read_ndjson(RAW/f'{n}.jsonl')
checks=[]
def check(name,condition,detail):
 assert condition,name
 checks.append(dict(check=name,status='Passed',detail=detail))
ev=read('events').unique(subset='event_id');valid=[];bad=[]
for r in ev.iter_rows(named=True):
 try:json.loads(r['metadata']);valid.append(r)
 except ValueError:bad.append(r)
(RAW/'quarantine.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in bad))
ev=pl.DataFrame(valid).with_columns(pl.col('metadata').str.json_decode(pl.Struct({'offer_id':pl.String,'duration_ms':pl.Int64,'field':pl.String,'campaign_id':pl.String})).alias('parsed')).unnest('parsed')
identity=ev.filter(pl.col('user_id').is_not_null()).group_by('session_id').agg(pl.col('user_id').n_unique().alias('users'),pl.col('user_id').first())
check('Session identity',identity.filter(pl.col('users')!=1).height==0,'600 sessions map to exactly one authenticated user.')
ev=ev.drop('user_id').join(identity.select('session_id','user_id'),on='session_id',validate='m:1')
touches=read('marketing').drop('user_id').join(identity.select('session_id','user_id'),on='session_id',validate='1:1')
leads=read('leads');offers=read('offers');bookings=read('bookings')
loans=bookings.join(offers,on='offer_id',validate='m:1').join(leads.select('lead_id','user_id'),on='lead_id',validate='m:1')
check('Booking join',loans.height==bookings.height,'No missing or multiplied booking records.')
check('Offer exposures',ev.filter(pl.col('event_type')=='offer_viewed').join(offers.select('offer_id'),on='offer_id',how='anti').height==0,'Every displayed offer references an existing offer.')
check('Event replay',len(tables['events'])-ev.height-len(bad)==7,'Seven replayed events removed by event_id.')
check('Malformed payloads',len(bad)==1,'One invalid JSON event quarantined.')
assign=read('assignments');check('Experiment assignment',assign['user_id'].n_unique()==assign.height,'One assignment per user before offer exposure; 50:50 Bernoulli allocation.')
metrics=ev.group_by('user_id').agg(*[(pl.col('event_type')==event).sum().alias(alias) for event,alias in [('form_error','form_errors'),('offer_viewed','offers_shown'),('offer_accepted','accepted'),('offer_rejected','rejected'),('unsubscribed','unsubscribed'),('price_reduction_requested','price_requested'),('price_reduced','price_reduced'),('mcp_escalated','escalated')]])
created=offers.join(leads.select('lead_id','user_id'),on='lead_id').group_by('user_id').len().rename({'len':'offers_created'})
users=touches.select('user_id','session_id','campaign_id',pl.col('timestamp').str.slice(0,10).alias('date'),'ad_spend').join(assign.select('user_id','variant','user_segment','credit_bracket','term_months'),on='user_id',validate='1:1').join(leads.select('user_id','lead_id','credit_score_bracket','requested_amount'),on='user_id',how='left',validate='1:1').join(metrics,on='user_id',how='left',validate='1:1').join(created,on='user_id',how='left',validate='1:1').join(loans.select('user_id','disbursed_amount',pl.lit(1).alias('converted')),on='user_id',how='left',validate='1:1').with_columns(pl.col('offers_created','offers_shown','converted').fill_null(0),pl.coalesce('credit_score_bracket','credit_bracket').alias('credit_score_bracket'))
# Each mock user has a single campaign touch before assignment; no ambiguous attribution.
check('User grain',users.height==600 and users['user_id'].n_unique()==600,'One row per assigned user; all seven-day outcome windows mature by November 1.')
con=read('conversations').unique(subset=['trace_id','turn_id'])
cm=con.group_by(['trace_id','session_id']).agg(pl.len().alias('turns'),pl.col('tools_called').list.len().sum().alias('tool_calls'),pl.col('content').filter(pl.col('role')=='user').first().alias('first_message'),pl.col('timestamp').min().alias('timestamp'))
cm=cm.join(identity.select('session_id','user_id'),on='session_id',validate='m:1').join(read('evaluations'),on='trace_id',validate='1:1').join(users.select('user_id','campaign_id','lead_id'),on='user_id',validate='m:1').with_columns(pl.when(pl.col('first_message').str.contains('rate')).then(pl.lit('rate_inquiry')).when(pl.col('first_message').str.contains('eligible')).then(pl.lit('eligibility')).otherwise(pl.lit('application_status')).alias('predicted_intent'))
cm=cm.with_columns((pl.col('predicted_intent')==pl.col('reference_intent')).cast(pl.Int32).alias('intent_match'))
pred=read('predictions').join(users.select('user_id','campaign_id','converted'),on='user_id',validate='1:1')
check('Model additivity',all(abs(1/(1+math.exp(-(r['base_logit']+r['credit']+r['amount']+r['friction'])))-r['score'])<1e-10 for r in pred.to_dicts()),'Exact additive log-odds contributions reconstruct every toy-model prediction.')
journey=ev.select('user_id','session_id','timestamp','event_type','offer_id').to_dicts()
for r in leads.to_dicts():journey.append(dict(user_id=r['user_id'],session_id=None,timestamp=r['timestamp'],event_type='lead_created',offer_id=None))
for r in loans.to_dicts():journey.append(dict(user_id=r['user_id'],session_id=None,timestamp=r['booking_timestamp'],event_type='loan_booked',offer_id=r['offer_id']))
for name,df in dict(dim_user=users.select('user_id'),dim_session=identity,dim_campaign=touches.select('campaign_id','utm_source','utm_medium','utm_campaign').unique(),fact_user_funnel=users,fact_lead=leads,fact_offer=offers,fact_booking=loans,fact_event=ev,fact_conversation=cm,fact_prediction=pred,fact_assignment=assign,fact_marketing_touch=touches).items():df.write_parquet(MODEL/f'{name}.parquet')
queries={}
def q(name,rows,definition,files):
 queries[name]=dict(rows=rows,source=dict(label='Synthetic assessment fixture',files=files,classification='synthetic',coverage={'start':'2026-10-01','end':'2026-11-01'},metricDefinitions=[dict(label=name.replace('_',' ').title(),definition=definition)],caveats=['All data are generated. No real customers, model training, or live LLM evaluation.'],evidenceFlow=[dict(title='Reproduce',detail='Run python build_data.py (seed 4271). Read raw JSONL, deduplicate events, quarantine invalid JSON, decode metadata using Polars, resolve session identities, join facts at declared keys, then materialize Parquet and this snapshot.')]))
q('users',users.to_dicts(),'One assigned authenticated user per row. Converted means at least one booked loan within seven days of assignment. Offers shown counts distinct offer_viewed records; shown users have at least one. Spend is additive allocated touch spend in demo currency units. One pre-assignment campaign touch per user.',['raw/marketing.jsonl','raw/assignments.jsonl','raw/leads.jsonl','raw/offers.jsonl','raw/bookings.jsonl','raw/events.jsonl'])
q('journey',sorted(journey,key=lambda x:x['timestamp']),'Timestamp-ordered user events. Origination lacks session_id; do not infer it.',['raw/events.jsonl','raw/leads.jsonl','raw/bookings.jsonl'])
q('conversations',cm.to_dicts(),'One trace per row. Evaluation fields are simulated rubric labels, not live judge scores. Tool-call counts come from nested logs; intent is a keyword baseline.',['raw/conversations.jsonl','raw/evaluations.jsonl'])
q('turns',con.to_dicts(),'One trace/turn. Context and tool results are inspectable synthetic logs.',['raw/conversations.jsonl'])
q('predictions',pred.to_dicts(),'Untrained illustrative logistic model: sigmoid(-0.15 + credit + amount + friction). Contributions are exact additive log-odds terms, not SHAP. Features precede the outcome.',['raw/predictions.jsonl'])
q('checks',checks,'Executed assertions from the local deterministic pipeline; these are not production SLA checks.',['build_data.py'])
snapshot=dict(title='Lending intelligence',surface='dashboard',status='fixture',buildStatus='creating',generatedAt=datetime.now(timezone.utc).isoformat(),filters=[dict(id='campaign',label='Campaign',field='campaign_id',defaultValue='all',queryIds=['users','conversations','predictions'])],queries=queries)
(ROOT/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
(ROOT/'validation.json').write_text(json.dumps(dict(users=users.height,leads=leads.height,offers=offers.height,offers_shown=users['offers_shown'].sum(),converted=loans.height,conversations=cm.height,checks=checks),indent=2))
print((ROOT/'validation.json').read_text())
