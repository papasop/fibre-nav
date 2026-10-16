"""Deterministic training-only candidate selection; no evaluation-state access."""
import math
from c17_protocol import quality,guard

def candidate_key(rows,ridge_index,backtrack_index):
 q=quality(rows)
 if not all(math.isfinite(q[k]) for k in ('maximum_drift','score')):raise ValueError('Nonfinite candidate quality')
 return (-len(q['passed_ids']),q['maximum_drift'],q['score'],ridge_index,backtrack_index)

def select_candidate(previous,candidates):
 """Candidates carry the full fixed training cohort, even if rejected."""
 valid=[c for c in candidates if guard(previous,c['rows'])['accepted']]
 if not valid:return None
 return min(valid,key=lambda c:candidate_key(c['rows'],c['ridge_index'],c['backtrack_index']))

def merge_route(previous,group_rows,group_ids):
 """Other expert outputs are unchanged; the worker audits all36 before commit."""
 if len(group_ids)!=9 or len(set(group_ids))!=9 or set(group_rows)!=set(group_ids):raise ValueError('All nine route states required')
 if not set(group_ids).issubset(previous):raise ValueError('Unknown training IDs')
 merged=dict(previous);merged.update(group_rows)
 quality(merged)
 return merged

def check_replay(expected,actual,tolerance=2e-6):
 """Full physical replay must agree before a candidate is committed."""
 if set(expected)!=set(actual):raise ValueError('Training replay denominator changed')
 for sid in expected:
  e,a=expected[sid],actual[sid]
  if e['reference_valid']!=a['reference_valid'] or e['metrics']['gates']!=a['metrics']['gates']:raise ValueError('Candidate physical gate replay mismatch: '+sid)
  for key in ('drift','kl'):
   if not math.isfinite(a['metrics'][key]) or abs(e['metrics'][key]-a['metrics'][key])>tolerance:raise ValueError('Candidate physical metric replay mismatch: '+sid)
 return True
