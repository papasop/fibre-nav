"""Fixed 76-state development guard with immutable protected original60."""
import math
from c24_protocol import training_specs as original60_specs,test_specs as promoted_specs,CASES,GATES
from candidate_selection import check_replay

def training_specs():return original60_specs()+[{**s,'cohort':'promoted_inspected_development'} for s in promoted_specs()]
def ordered_metrics(rows):
 ids=[s['id'] for s in training_specs()]
 if set(rows)!=set(ids):raise ValueError('All original60 + promoted16 states required')
 for sid in ids:
  m=rows[sid]['metrics']
  if set(m['gates'])!=set(GATES) or not math.isfinite(m['drift']) or m['drift']<0 or not math.isfinite(m['kl']):raise ValueError('Invalid training metric: '+sid)
 return [(sid,rows[sid]['metrics']) for sid in ids]
def quality(rows):
 metrics=ordered_metrics(rows);original={t['id'] for t in original60_specs()}
 return {'passed_ids':[sid for sid,m in metrics if rows[sid]['reference_valid'] and all(m['gates'].values())],
  'maximum_drift':max(m['drift'] for _,m in metrics),'original60_maximum_drift':max(m['drift'] for sid,m in metrics if sid in original),
  'score':sum((m['drift']/.002)**2 for _,m in metrics)/76,
  'violation_score':sum((max(m['drift']-.002,0.)/.002)**2 for _,m in metrics)/76,'valid_references':sum(bool(rows[sid]['reference_valid']) for sid,_ in metrics)}
def guard(previous,candidate):
 a=quality(previous);b=quality(candidate);reasons=[]
 if a['valid_references']!=76 or b['valid_references']!=76:reasons.append('INVALID_TRAINING_REFERENCE')
 if not set(a['passed_ids']).issubset(b['passed_ids']):reasons.append('LOST_PREVIOUS_COMPLETE_PASS')
 original={t['id'] for t in original60_specs()}
 if not original.issubset(b['passed_ids']):reasons.append('ORIGINAL60_NOT_ALL_ACCEPTED')
 for sid,m in ordered_metrics(previous):
  for g in GATES:
   if g!='response' and m['gates'][g] and not candidate[sid]['metrics']['gates'][g]:reasons.append('LOST_NON_RESPONSE_GATE:'+sid+':'+g)
 if b['maximum_drift']>a['maximum_drift']:reasons.append('WORST76_RESPONSE_DRIFT_INCREASED')
 if len(b['passed_ids'])<=len(a['passed_ids']) and not b['violation_score']<a['violation_score']*(1-1e-6):reasons.append('NO_DECLARED_TRAINING_IMPROVEMENT')
 return {'accepted':not reasons,'reasons':reasons,'before':a,'after':b}
def merge_route(previous,rows,ids):
 if len(ids)!=19 or len(set(ids))!=19 or set(rows)!=set(ids) or len({sid.split('__')[0] for sid in ids})!=1 or not set(ids).issubset(previous):raise ValueError('All19 selected-route training states required')
 merged=dict(previous);merged.update(rows);quality(merged);return merged

def candidate_key(c):
 q=quality(c['rows']);return (-len(q['passed_ids']),q['maximum_drift'],q['violation_score'],q['score'],c['radius_index'],c['ridge_index'],c['backtrack_index'])
def choose_case(rows,stalled,last_index=-1):
 q=quality(rows);passed=set(q['passed_ids'])
 eligible={c for c in CASES if c not in stalled and any(s['case']==c and s['id'] not in passed for s in training_specs())}
 for offset in range(1,len(CASES)+1):
  i=(last_index+offset)%len(CASES)
  if CASES[i] in eligible:return CASES[i]
 return None
