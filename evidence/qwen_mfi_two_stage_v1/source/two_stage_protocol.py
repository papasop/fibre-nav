"""Model-produced intermediate text, separate computation call, explicit corruptions."""
import re,hashlib
from bridge_protocol import template_map,answer,read_cases,SNAPS,TASKS
ARMS=('on',)
CONDITIONS=('raw','flip0','flip1','flip_both','no_intermediate')

def parse_read(raw,truncated=False):
 if truncated:return None
 m=re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])',raw.strip())
 return None if m is None else m[1]+m[2]

def cases():
 return [dict(id=f'{s}-{c}-{t}',snapshot=s,arm='on',condition=c,task=t) for s in SNAPS for c in CONDITIONS for t in TASKS]

def intermediate(raw,condition,truncated=False):
 # Main path forwards exact raw text. Parsing is NOT used to reconstruct it.
 if condition=='raw':return raw
 if condition=='no_intermediate':return None
 if condition not in ('flip0','flip1','flip_both'):raise ValueError('Bad condition')
 if parse_read(raw,truncated) is None:return None
 indices={'flip0':{0},'flip1':{1},'flip_both':{0,1}}[condition]
 # Preserve whitespace, ordering and every other character. Explicit control only.
 return re.sub(r'(register_0([01]): )([01])',lambda m:m[1]+(str(1-int(m[3])) if int(m[2]) in indices else m[3]),raw)

def compute_prompt(raw,task,condition,truncated=False):
 body=template_map()[1][task];text=intermediate(raw,condition,truncated)
 if condition=='no_intermediate':return body
 if text is None:return None
 return text+'\n'+body

def bind(c,source):
 if source['id']!=c['snapshot']+'-joint-0' or source['phase']!='before' or source['arm']!='on':raise ValueError('Wrong source read')
 raw=source['raw'];tr=source['truncated'];text=intermediate(raw,c['condition'],tr)
 supplied=parse_read(text,tr if c['condition']=='raw' else False) if text is not None else None
 memory=None if c['snapshot']=='zero' else c['snapshot']
 return dict(**c,source_read_id=source['id'],source_raw=raw,source_truncated=tr,source_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
  supplied_text=text,supplied_values=supplied,memory_state=memory,
  prompt=compute_prompt(raw,c['task'],c['condition'],tr),
  expected_from_intermediate=None if supplied is None else answer(c['task'],supplied),
  expected_from_memory=None if memory is None else answer(c['task'],memory),
  intervention=c['condition'] in ('flip0','flip1','flip_both'))

def score(c,raw,truncated=False):
 pred=None;values=None
 if not truncated:
  m=re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])\r?\nRESULT: ([01])',raw.strip())
  if m:values=m[1]+m[2];pred=m[3]
 memory_ok=None if c['memory_state'] is None else values==c['memory_state'] and pred==c['expected_from_memory']
 follow=None if c['supplied_values'] is None else values==c['supplied_values'] and pred==c['expected_from_intermediate']
 source_ok=None if c['memory_state'] is None else parse_read(c['source_raw'],c['source_truncated'])==c['memory_state']
 return dict(prediction=pred,reported_values=values,format_pass=pred is not None,follows_intermediate=follow,
  memory_answer_correct=None if c['memory_state'] is None else pred==c['expected_from_memory'],
  strict_memory_output=memory_ok,source_read_correct=source_ok,
  end_to_end_success=None if c['condition']!='raw' or c['memory_state'] is None else bool(source_ok and memory_ok and follow))

def summarize(rows,replay):
 plan={c['id']:c for c in cases()};seen=set()
 for r in rows:
  if r['id'] in seen:raise ValueError('Duplicate row')
  seen.add(r['id']);c=plan[r['id']]
  source=next(x for x in replay if x['phase']=='before' and x['arm']=='on' and x['id']==c['snapshot']+'-joint-0')
  bound=bind(c,source)
  if any(r.get(k)!=v for k,v in bound.items()):raise ValueError('Intermediate/prompt provenance mismatch')
  skip=bound['prompt'] is None
  if r['skipped']!=skip:raise ValueError('Invalid skip')
  if skip:
   if r['raw']!='' or r['output_ids'] or not r['truncated']:raise ValueError('Skipped result fabricated')
  if any(r.get(k)!=v for k,v in score(bound,r['raw'],r['truncated']).items()):raise ValueError('Score mismatch')
 groups={}
 for s in SNAPS:
  groups[s]={}
  for c in CONDITIONS:
   rr=[r for r in rows if(r['snapshot'],r['condition'])==(s,c)]
   groups[s][c]=dict(n=len(rr),generated=sum(not r['skipped'] for r in rr),format_pass=sum(r['format_pass'] for r in rr),follows_intermediate=sum(r['follows_intermediate'] is True for r in rr),strict_memory_output=sum(r['strict_memory_output'] is True for r in rr),end_to_end_success=sum(r['end_to_end_success'] is True for r in rr))
 primary=[r for r in rows if r['snapshot']!='zero' and r['condition']=='raw']
 flips=[r for r in rows if r['snapshot']!='zero' and r['condition'] in ('flip0','flip1')]
 both=[r for r in rows if r['snapshot']!='zero' and r['condition']=='flip_both']
 effects=[]
 for r in flips+both:
  original=next((x for x in primary if(x['snapshot'],x['task'])==(r['snapshot'],r['task'])),None)
  if original:
   prescribed=(r['prediction']==original['prediction']) if r['condition']=='flip_both' else (r['prediction']!=original['prediction'])
   effects.append(dict(id=r['id'],both_format_valid=r['format_pass'] and original['format_pass'],prescribed_effect=bool(r['follows_intermediate'] and original['follows_intermediate'] and prescribed)))
 return dict(rows=len(rows),expected_rows=30,generated_rows=sum(not r['skipped'] for r in rows),skipped_rows=sum(r['skipped'] for r in rows),groups=groups,
  known_raw_four_pass=len(primary)==4 and all(r['end_to_end_success'] for r in primary),
  known_single_flip_eight_pass=len(flips)==8 and all(r['follows_intermediate'] for r in flips),
  known_double_flip_four_pass=len(both)==4 and all(r['follows_intermediate'] for r in both),
  intermediate_effects=effects,formal_l6_pass=False,single_pass_l6_pass=False,training_updates=0,memory_writes=0,
  scope='Two independent generation calls using same frozen model. Model-produced text is externally carried unchanged into stage2; text-mediated computation, not single-pass parameter-memory logic. Only known10/11; first cell fixed. Corruptions alter intermediate TEXT, not parameter addresses. Zero has no assigned memory label; no training/new-state transfer or blind claim.')
