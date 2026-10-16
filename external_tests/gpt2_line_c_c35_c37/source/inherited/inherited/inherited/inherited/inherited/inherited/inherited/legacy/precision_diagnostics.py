"""Interpret repeated identical-state probes without attributing a unique error source."""
import math

def controls_summary(records,state_ids):
 expected={(phase,rep,sid) for phase in ('before_scan','after_scan') for rep in range(3) for sid in state_ids}
 actual={(r['phase'],r['repeat'],r['state_id']) for r in records}
 if len(actual)!=len(records) or actual!=expected:raise ValueError('All six zero probes per state required')
 states=[]
 for sid in state_ids:
  group=[r for r in records if r['state_id']==sid];vectors=[r['probe']['response_residual'] for r in group];vectors64=[r['probe']['response_residual_float64_aggregation'] for r in group]
  if not vectors or any(len(v)!=len(vectors[0]) for v in vectors+vectors64):raise ValueError('Response dimensions changed')
  if any(not math.isfinite(x) for v in vectors+vectors64 for x in v):raise ValueError('Nonfinite repeated response')
  spread32=max(max(v[k] for v in vectors)-min(v[k] for v in vectors) for k in range(len(vectors[0])))
  spread64=max(max(v[k] for v in vectors64)-min(v[k] for v in vectors64) for k in range(len(vectors[0])))
  states.append({'id':sid,'n_repeats':len(group),'maximum_vector_spread_float32':spread32,'maximum_vector_spread_float64_aggregation':spread64,
   'distinct_full_logit_hashes':len({r['probe']['full_logits_sha256'] for r in group}),
   'maximum_aggregation_difference':max(r['probe']['maximum_aggregation_difference'] for r in group)})
 return {'n_states':len(states),'n_zero_forward_probes':len(records),'states':states,
  'maximum_vector_spread_float32':max(s['maximum_vector_spread_float32'] for s in states),
  'maximum_vector_spread_float64_aggregation':max(s['maximum_vector_spread_float64_aggregation'] for s in states),
  'pure_roundoff_cause_established':False,'full_float64_model_forward_tested':False,'same_float32_logits_for_both_aggregations':True}
