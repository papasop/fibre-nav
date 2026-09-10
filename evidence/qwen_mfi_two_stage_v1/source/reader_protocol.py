"""Known-state paired calibration with hard candidate acceptance, NOT L6."""
import random,re,math
SEED=84918
SNAPSHOTS=('zero','10','11')
STATES=('00','10','11','01')
CONFIG=dict(seed=SEED,rank=4,scale=2.,max_attempts=40,backtracks=6,stall_limit=3,radius=.05,min_radius=.0001,response_budget=.02,kl_budget=.01,read_margin=1.,joint_bit_margin=1.,min_objective_improvement=1e-7)
PROMPTS=(
 'Read both register values. Output exactly two lines, replacing each <bit> with 0 or 1:\nregister_00: <bit>\nregister_01: <bit>\nDo not add any other text.\nResponse:',
 'What values are stored at register_00 and register_01? Give exactly two lines in this format:\nregister_00: <bit>\nregister_01: <bit>\nResponse:',
 'Report register_01 first and register_00 second. Use exactly this two-line format:\nregister_01: <bit>\nregister_00: <bit>\nResponse:')

def target(state,form=0):
 order=(1,0) if form==2 else (0,1)
 return '\n'.join(f'register_0{i}: {state[i]}' for i in order)

def prompt(form=0,facts=None):
 return (f'register_00: {facts[0]}\nregister_01: {facts[1]}\n' if facts is not None else '')+PROMPTS[form]

def evaluation_cases():
 rows=[]
 for snap in SNAPSHOTS:
  for form in range(3):
   rows.append(dict(id=f'{snap}-joint-{form}',snapshot=snap,kind='joint',form=form,mode='implicit',state=None if snap=='zero' else snap,prompt=prompt(form)))
  for state in STATES:
   rows.append(dict(id=f'{snap}-explicit-{state}',snapshot=snap,kind='joint',form=0,mode='explicit',state=state,prompt=prompt(facts=state)))
  for cell in (0,1):
   rows.append(dict(id=f'{snap}-copy-{cell}',snapshot=snap,kind='copy',cell=cell,form=0,mode='implicit',state=None if snap=='zero' else snap,prompt=f'Read register_0{cell}.\nOutput one digit only.\nResult:'))
 return rows

def score(c,raw,truncated=False):
 value=None
 if not truncated:
  if c['kind']=='copy':value=raw.strip() if raw.strip() in ('0','1') else None
  else:
   order=(1,0) if c['form']==2 else (0,1)
   match=re.fullmatch(f'register_0{order[0]}: ([01])\\r?\\nregister_0{order[1]}: ([01])',raw.strip())
   if match:value=''.join(match.groups()[order.index(i)] for i in (0,1))
 expected=None if c['state'] is None else c['state'][c['cell']] if c['kind']=='copy' else c['state']
 return dict(prediction=value,expected=expected,format_pass=value is not None,correct=None if expected is None else value==expected)


def hard_gate(audits):
 reasons=[]
 if set(audits)!=set(SNAPSHOTS):return dict(pass_all=False,reasons=['missing snapshots'])
 for snap,a in audits.items():
  response=a.get('response',math.inf);kl=a.get('kl',math.inf)
  if not math.isfinite(response) or not 0<=response<=.02:reasons.append(snap+':response')
  if not math.isfinite(kl) or not -1e-8<=kl<=.01:reasons.append(snap+':kl')
  margins=a.get('rule_margins',[]);floors=a.get('rule_floors',[])
  if len(margins)!=8 or len(floors)!=8 or any(not math.isfinite(v) or not math.isfinite(f) or v<f for v,f in zip(margins,floors)):reasons.append(snap+':rule_floors')
  if snap!='zero':
   m=a.get('margin',-math.inf)
   if not math.isfinite(m) or m<1:reasons.append(snap+':single_margin')
   copies=a.get('copies',[])
   if len(copies)!=2 or any(c.get('correct') is not True for c in copies):reasons.append(snap+':single_generation')
 return dict(pass_all=not reasons,reasons=reasons)

def pair_fit(joint):
 return set(joint)=={'10','11'} and all(joint[s].get('correct') is True and math.isfinite(joint[s].get('signed_bit_margin',-math.inf)) and joint[s]['signed_bit_margin']>=1 for s in ('10','11'))

def summarize(rows,status,attempts,accepted,audits,joint,integrity,frozen,replay,stop):
 planned={c['id']:c for c in evaluation_cases()};seen=set()
 for row in rows:
  key=(row['phase'],row['id'])
  if key in seen or row['phase'] not in ('before','after'):raise ValueError('Duplicate/invalid phase')
  seen.add(key);c=planned.get(row['id'])
  if c is None or any(row.get(k)!=v for k,v in c.items()):raise ValueError('Case mismatch')
  if any(row.get(k)!=v for k,v in score(c,row['raw'],row['truncated']).items()):raise ValueError('Score mismatch')
 complete=status=='EVALUATED' and len(rows)==54 and integrity and frozen and replay and stop in ('PAIRED_FIT_REACHED','MAX_ATTEMPTS','STALLED')
 groups={}
 for phase in ('before','after'):
  groups[phase]={}
  for snap in SNAPSHOTS:
   rr=[r for r in rows if r['phase']==phase and r['snapshot']==snap];groups[phase][snap]={}
   for label,pred in [('canonical',lambda r:r['mode']=='implicit' and r['kind']=='joint' and r['form']==0),('paraphrase',lambda r:r['mode']=='implicit' and r['kind']=='joint' and r['form']==1),('reverse_order',lambda r:r['mode']=='implicit' and r['kind']=='joint' and r['form']==2),('explicit',lambda r:r['mode']=='explicit'),('copy',lambda r:r['kind']=='copy')]:
    subset=[r for r in rr if pred(r)];groups[phase][snap][label]=dict(n=len(subset),scored=sum(r['correct'] is not None for r in subset),correct=sum(r['correct'] is True for r in subset),format_pass=sum(r['format_pass'] for r in subset))
 constraints=hard_gate(audits.get('after',{}));fit=pair_fit(joint.get('after',{}))
 canonical=all(groups['after'][s]['canonical']['correct']==1 for s in ('10','11'))
 return dict(status='PAIRED_READER_CALIBRATION_COMPLETE' if complete else status,complete=bool(complete),rows=len(rows),expected_rows=54,attempts=attempts,accepted_updates=accepted,max_attempts=40,stop_reason=stop,groups=groups,source_integrity=integrity,frozen_tensors_verified=frozen,source_replay=replay,final_constraints=constraints,paired_calibration_fit=bool(fit and canonical),paired_reader_gate=bool(complete and accepted>0 and constraints['pass_all'] and fit and canonical),formal_l6_pass=False,new_memory_generalization_tested=False,cell0_variation_tested=False,memory_write_operations=0,reader_calibration_supervised=True,rule_answers_used_in_candidate_acceptance=True,mfi_geometry_revalidated=False,scope='Bounded development calibration on known10/11: only new reader B changes. Joint training uses both known values; original logical answer guards constrain candidate acceptance. No new memory writes or implicit logical computation. Hard gates apply to checked candidate endpoints, not a certified continuous path or unseen tasks. Public paraphrase/reverse/explicit probes are excluded from update decisions. Source memory and reference remain unchanged; future MFI geometry must be rebuilt.')
