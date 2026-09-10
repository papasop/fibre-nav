"""Frozen protected interface; explicit vs implicit, three snapshots, reader on/off."""
import json,re
from pathlib import Path
from reader_protocol import evaluation_cases
ARMS=('on','off')
SNAPS=('zero','10','11')
TASKS=('xor','equal')
FACTS=('00','01','10','11')

def answer(task,state):
 if task not in TASKS or state not in FACTS:raise ValueError('Bad truth table input')
 return str(int((state[0]!=state[1]) if task=='xor' else (state[0]==state[1])))

def template_map():
 pp=json.loads((Path(__file__).parent/'protection_cases.json').read_text());templates={};bodies={}
 if len(pp)!=8:raise ValueError('Expected eight protection cases')
 for c in pp:
  key=(c['state'],c['task'])
  if key in templates:raise ValueError('Duplicate source prompt')
  prefix=f"register_00: {c['state'][0]}\nregister_01: {c['state'][1]}\n"
  if not c['prompt'].startswith(prefix):raise ValueError('Unexpected source facts prefix')
  if str(c['expected'])!=answer(c['task'],c['state']):raise ValueError('Wrong source label')
  templates[key]=c['prompt'];bodies.setdefault(c['task'],set()).add(c['prompt'][len(prefix):])
 if set(templates)!={(s,t) for s in FACTS for t in TASKS} or any(len(bodies[t])!=1 for t in TASKS):raise ValueError('Task body must be independent of stored values')
 return templates,{t:next(iter(bodies[t])) for t in TASKS}

def cases():
 explicit,implicit=template_map();rows=[]
 for snap in SNAPS:
  for facts in (None,)+FACTS:
   for task in TASKS:
    for arm in ARMS:
     state=(None if snap=='zero' else snap) if facts is None else facts
     rows.append(dict(id=f'{snap}-{arm}-{facts or "implicit"}-{task}',snapshot=snap,arm=arm,condition='implicit' if facts is None else 'explicit',facts=facts,state=state,task=task,expected=None if state is None else answer(task,state),prompt=implicit[task] if facts is None else explicit[(facts,task)]))
 return rows

def read_cases():
 return [c for c in evaluation_cases() if c['kind']=='copy' or(c['kind']=='joint' and c['mode']=='implicit' and c['form']==0)]

def score(c,raw,truncated=False):
 pred=None;values=None
 if not truncated:
  m=re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])\r?\nRESULT: ([01])',raw.strip())
  if m:values=m[1]+m[2];pred=m[3]
 correct=None if c['expected'] is None else pred==c['expected']
 val_ok=None if c['state'] is None else values==c['state']
 return dict(prediction=pred,reported_values=values,format_pass=pred is not None,correct=correct,reported_values_correct=val_ok,
  strict_success=None if c['state'] is None else correct and val_ok,
  self_consistent=None if values is None else pred==answer(c['task'],values))

def counts(rr):
 return dict(n=len(rr),scored=sum(r['expected'] is not None for r in rr),format_pass=sum(r['format_pass'] for r in rr),answer_correct=sum(r['correct'] is True for r in rr),reported_values_correct=sum(r['reported_values_correct'] is True for r in rr),strict_success=sum(r['strict_success'] is True for r in rr))

def summarize(rows):
 plan={c['id']:c for c in cases()};seen=set()
 for r in rows:
  if r['id'] in seen:raise ValueError('Duplicate row')
  seen.add(r['id']);c=plan[r['id']]
  if any(r.get(k)!=v for k,v in c.items()):raise ValueError('Case changed')
  if any(r.get(k)!=v for k,v in score(c,r['raw'],r['truncated']).items()):raise ValueError('Score changed')
 groups={};signals={}
 for arm in ARMS:
  groups[arm]={}
  for snap in SNAPS:
   rr=[r for r in rows if(r['arm'],r['snapshot'])==(arm,snap)]
   groups[arm][snap]=dict(implicit=counts([r for r in rr if r['condition']=='implicit']),explicit_all=counts([r for r in rr if r['condition']=='explicit']),explicit_matched=counts([r for r in rr if r['condition']=='explicit' and r['facts']==snap]))
  implicit=[r for r in rows if r['arm']==arm and r['snapshot']!='zero' and r['condition']=='implicit']
  matched=[r for r in rows if r['arm']==arm and r['facts']==r['snapshot']]
  effects=[]
  for task in TASKS:
   pair={r['snapshot']:r for r in implicit if r['task']==task}
   ok=set(pair)=={'10','11'} and all(r['strict_success'] for r in pair.values()) and pair['10']['prediction']!=pair['11']['prediction']
   effects.append(dict(task=task,known_state_prescribed_flip=bool(ok)))
  signals[arm]=dict(known_implicit_four_pass=len(implicit)==4 and all(r['strict_success'] for r in implicit),matched_explicit_four_pass=len(matched)==4 and all(r['strict_success'] for r in matched),state_effects=effects)
 comparisons=[]
 for r in rows:
  if r['condition']!='implicit' or r['snapshot']=='zero':continue
  zero=next((x for x in rows if(x['snapshot'],x['arm'],x['condition'],x['task'])==('zero',r['arm'],'implicit',r['task'])),None)
  if zero:comparisons.append(dict(snapshot=r['snapshot'],arm=r['arm'],task=r['task'],memory_prediction=r['prediction'],zero_prediction=zero['prediction'],both_valid=r['format_pass'] and zero['format_pass'],prediction_changed=r['prediction']!=zero['prediction']))
 return dict(rows=len(rows),expected_rows=60,groups=groups,logic_signals=signals,zero_comparisons=comparisons,
  formal_l6_pass=False,training_updates=0,memory_write_operations=0,first_cell_varies=False,new_memory_generalization_tested=False,
  scope='Development diagnostic selected after interface results. Fixed known10/11 snapshots, first cell always1; cannot establish two-cell joint causation or unseen-memory transfer. Values absent from implicit prompts; no external read or answer repair. Explicit conflict cases scored against supplied facts. Main reader-on arm and secondary reader-off arm reported separately, no best-arm selection.')
