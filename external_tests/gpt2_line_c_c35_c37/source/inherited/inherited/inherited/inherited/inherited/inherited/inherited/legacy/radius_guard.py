"""Fixed 60-state training coverage; original36 preservation is checked separately."""
import math
from c17_protocol import training_specs as original_training_specs,GATES,CASES
from candidate_selection import check_replay

TRAIN_SEED_BASE=2014031

def supplemental_training_specs():
 return [{'id':c+'__c20_train%d'%i,'case':c,'base_id':c+'__v%d'%(i%3),
  'seed':TRAIN_SEED_BASE+100*k+i,'norm':.375 if i<3 else .75,'cohort':'coverage_training'}
  for k,c in enumerate(CASES) for i in range(6)]

def training_specs():return original_training_specs()+supplemental_training_specs()

def ordered_metrics(rows):
 ids=[t['id'] for t in training_specs()]
 if set(rows)!=set(ids):raise ValueError('All original36 + supplemental24 training states required')
 for sid in ids:
  m=rows[sid]['metrics']
  if set(m['gates'])!=set(GATES) or not math.isfinite(m['drift']) or m['drift']<0:raise ValueError('Invalid training measurement: '+sid)
 return [(sid,rows[sid]['metrics']) for sid in ids]

def quality(rows):
 metrics=ordered_metrics(rows);original={t['id'] for t in original_training_specs()}
 return {'passed_ids':[sid for sid,m in metrics if rows[sid]['reference_valid'] and all(m['gates'].values())],
  'maximum_drift':max(m['drift'] for _,m in metrics),
  'score':sum((m['drift']/.002)**2 for _,m in metrics)/60,
  'original36_maximum_drift':max(m['drift'] for sid,m in metrics if sid in original),
  'valid_references':sum(bool(rows[sid]['reference_valid']) for sid,_ in metrics)}

def guard(previous,candidate):
 a=quality(previous);b=quality(candidate);reasons=[]
 if a['valid_references']!=60 or b['valid_references']!=60:reasons.append('INVALID_TRAINING_REFERENCE')
 if not set(a['passed_ids']).issubset(b['passed_ids']):reasons.append('LOST_PREVIOUS_COMPLETE_PASS')
 for sid,m in ordered_metrics(previous):
  for g in GATES:
   if g!='response' and m['gates'][g] and not candidate[sid]['metrics']['gates'][g]:reasons.append('LOST_NON_RESPONSE_GATE:'+sid+':'+g)
 if b['maximum_drift']>a['maximum_drift']:reasons.append('WORST_RESPONSE_DRIFT_INCREASED')
 if b['original36_maximum_drift']>a['original36_maximum_drift']:reasons.append('ORIGINAL36_WORST_DRIFT_INCREASED')
 if len(b['passed_ids'])<=len(a['passed_ids']) and not b['score']<a['score']*(1-1e-6):reasons.append('NO_DECLARED_TRAINING_IMPROVEMENT')
 return {'accepted':not reasons,'reasons':reasons,'before':a,'after':b}

def candidate_key(c):
 q=quality(c['rows'])
 return (-len(q['passed_ids']),q['maximum_drift'],q['score'],c['radius_index'],c['ridge_index'],c['backtrack_index'])

def select_candidate(previous,candidates):
 valid=[c for c in candidates if guard(previous,c['rows'])['accepted']]
 return min(valid,key=candidate_key) if valid else None

def merge_route(previous,group_rows,group_ids):
 if len(group_ids)!=15 or len(set(group_ids))!=15 or set(group_rows)!=set(group_ids):raise ValueError('All fifteen selected-route training states required')
 cases={sid.split('__')[0] for sid in group_ids}
 if len(cases)!=1 or not set(group_ids).issubset(previous):raise ValueError('Mixed or unknown route training IDs')
 result=dict(previous);result.update(group_rows);quality(result);return result
