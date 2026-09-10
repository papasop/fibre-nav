"""Fixed 72-query read-only follow-up; explicit rescue is not an L6 pass."""
import math,re
SEED=84917
SNAPSHOTS=('zero','00','10','11','01','00_return')
STATES=('00','10','11','01')
TASKS=('copy0','copy1','xor','equal')
FORMAT='Output exactly three lines, with each <bit> replaced by one digit, 0 or 1:\nregister_00: <bit>\nregister_01: <bit>\nRESULT: <bit>\nDo not add any other text.'

def state_of(snapshot):
 if snapshot not in SNAPSHOTS:raise ValueError('Unknown snapshot')
 return None if snapshot=='zero' else '00' if snapshot=='00_return' else snapshot

def expected(task,state):
 a,b=map(int,state)
 return {'copy0':a,'copy1':b,'xor':a^b,'equal':int(a==b)}[task]

def read_prompt(cell):
 if cell not in (0,1):raise ValueError('Unknown cell')
 return f'Read register_0{cell}.\nOutput one digit only.\nResult:'

def implicit_prompt(task):
 if task in ('copy0','copy1'):return read_prompt(int(task[-1]))
 if task not in ('xor','equal'):raise ValueError('Unknown task')
 verb='differ' if task=='xor' else 'agree'
 return 'Read both register values, then apply the rule.\n'+f'Return 1 if register_00 and register_01 {verb}, otherwise return 0.\n'+FORMAT+'\nResponse:'

def cases():
 result=[]
 for snapshot in SNAPSHOTS:
  memory=state_of(snapshot)
  for task in TASKS:
   result.append(dict(id=f'{snapshot}-implicit-{task}',snapshot=snapshot,condition='implicit',memory_state=memory,
    state=memory,task=task,kind='direct' if task.startswith('copy') else 'staged',prompt=implicit_prompt(task),
    expected=expected(task,memory) if memory is not None else None,relation='no_assignment' if memory is None else 'memory_only'))
  for state in STATES:
   for task in ('xor','equal'):
    result.append(dict(id=f'{snapshot}-explicit-{state}-{task}',snapshot=snapshot,condition='explicit',memory_state=memory,
     state=state,task=task,kind='staged',prompt=f'register_00: {state[0]}\nregister_01: {state[1]}\n'+implicit_prompt(task),
     expected=expected(task,state),relation='no_assignment' if memory is None else 'matched' if state==memory else 'conflicting'))
 return result

def score(case,raw,truncated=False):
 known=case['state'] is not None
 if case['kind']=='direct':
  pred=int(raw.strip()) if not truncated and raw.strip() in ('0','1') else None
  return dict(prediction=pred,format_pass=pred is not None,correct=(pred==case['expected']) if known else None,
   strict_success=(pred==case['expected']) if known else None)
 m=None if truncated else re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])\r?\nRESULT: ([01])',raw.strip())
 trace=tuple(map(int,m.groups())) if m else None
 reads=bool(trace and ''.join(map(str,trace[:2]))==case['state']) if known else None
 correct=bool(trace and trace[2]==case['expected']) if known else None
 return dict(prediction=trace[2] if trace else None,format_pass=trace is not None,reported_values=list(trace[:2]) if trace else None,
  reported_values_correct=reads,correct=correct,strict_success=bool(reads and correct) if known else None,
  rule_consistent_with_reported_values=bool(trace and trace[2]==expected(case['task'],''.join(map(str,trace[:2])))))

def count(rows):
 return dict(n=len(rows),scored_n=sum(r['state'] is not None for r in rows),format_pass=sum(r['format_pass'] for r in rows),
  strict_success=sum(r['strict_success'] is True for r in rows),answer_correct=sum(r['correct'] is True for r in rows),
  reads_correct=sum(r.get('reported_values_correct') is True for r in rows),
  self_consistent=sum(r.get('rule_consistent_with_reported_values') is True for r in rows))

def budget_pass(a):
 return all(isinstance(a.get(k),(int,float)) and math.isfinite(a[k]) for k in ('response','kl')) and 0<=a['response']<=.02 and -1e-10<=a['kl']<=.01

def summarize(rows,status,integrity,audits):
 plan={c['id']:c for c in cases()}
 if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate rows')
 for r in rows:
  c=plan.get(r['id'])
  if c is None or any(r.get(k)!=v for k,v in c.items()):raise ValueError('Case changed')
  if any(r.get(k)!=v for k,v in score(c,r['raw'],r['truncated']).items()):raise ValueError('Score changed')
 complete=status=='EVALUATED' and len(rows)==72 and integrity
 groups={};comparisons=[];lookup={r['id']:r for r in rows}
 for snap in SNAPSHOTS:
  rr=[r for r in rows if r['snapshot']==snap]
  groups[snap]={
   'implicit_copy':count([r for r in rr if r['kind']=='direct']),
   'implicit_joint':count([r for r in rr if r['condition']=='implicit' and r['kind']=='staged']),
   'explicit_all':count([r for r in rr if r['condition']=='explicit']),
   'explicit_matched':count([r for r in rr if r['relation']=='matched']),
   'explicit_conflicting':count([r for r in rr if r['relation']=='conflicting'])}
  if snap=='zero':continue
  memory=state_of(snap)
  for task in ('xor','equal'):
   imp=lookup.get(f'{snap}-implicit-{task}');exp=lookup.get(f'{snap}-explicit-{memory}-{task}')
   if imp is not None and exp is not None:
    comparisons.append(dict(snapshot=snap,task=task,implicit_success=imp['strict_success'],explicit_matched_success=exp['strict_success'],
     rescue_observed=imp['strict_success'] is False and exp['strict_success'] is True))
 # Paired identical explicit prompts, comparing each snapshot to zero; descriptive, not independent trials.
 losses={}
 for snap in SNAPSHOTS[1:]:
  pairs=[]
  for state in STATES:
   for task in ('xor','equal'):
    z=lookup.get(f'zero-explicit-{state}-{task}');r=lookup.get(f'{snap}-explicit-{state}-{task}')
    if z is not None and r is not None:pairs.append((z,r))
  losses[snap]=dict(pairs=len(pairs),zero_success_to_failure=sum(z['strict_success'] is True and r['strict_success'] is False for z,r in pairs),
   zero_failure_to_success=sum(z['strict_success'] is False and r['strict_success'] is True for z,r in pairs))
 budget_ok=len(audits)==6 and all(budget_pass(a) for a in audits.values())
 return dict(status='READ_ONLY_DIAGNOSTIC_COMPLETE' if complete else status,complete=complete,rows=len(rows),expected_rows=72,
  integrity_verified=integrity,all_response_kl_budgets_pass=budget_ok,groups=groups,matched_comparisons=comparisons,explicit_vs_zero=losses,
  interpretation_ready=bool(complete and budget_ok),formal_l6_pass=False,prior_l6_verdict='MEMORY_COMPUTATION_NOT_SUPPORTED',
  parameter_updates=0,training_steps=0,
  scope='Public development diagnostic of existing snapshots and known prompts. Explicit facts are a control, not evidence of computation from parameter memory. Rescue suggests interface dependence, not a uniquely identified mechanism. Failure can involve formatting, reading, or computation; not proof that a rule was erased. XOR/equality are complements. Repeated/identical snapshots are not independent replications. No unseen-state, generalization or full L1-L5 claim.')
