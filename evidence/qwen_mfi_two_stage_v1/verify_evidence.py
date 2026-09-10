#!/usr/bin/env python3
"""Stdlib-only review of supplied evidence; NOT an independent Qwen model run."""
import argparse,hashlib,json,math,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text())
def bit_answer(task,bits):return str(int(bits[0]!=bits[1]) if task=='xor' else int(bits[0]==bits[1]))
def parse_two(raw,truncated=False):
 m=None if truncated else re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])',raw.strip())
 return None if m is None else m[1]+m[2]
def parse_three(raw,truncated=False):
 m=None if truncated else re.fullmatch(r'register_00: ([01])\r?\nregister_01: ([01])\r?\nRESULT: ([01])',raw.strip())
 return (None,None) if m is None else (m[1]+m[2],m[3])
def flip(text,indices):
 return re.sub(r'(register_0([01]): )([01])',lambda m:m[1]+(str(1-int(m[3])) if int(m[2]) in indices else m[3]),text)
def gate(audits):
 if set(audits)!={'zero','10','11'}:return False
 for snap,a in audits.items():
  if not (math.isfinite(a['response']) and 0<=a['response']<=.02 and math.isfinite(a['kl']) and -1e-8<=a['kl']<=.01):return False
  if len(a['rule_margins'])!=8 or len(a['rule_floors'])!=8:return False
  if not all(math.isfinite(x) and math.isfinite(y) and x>=y for x,y in zip(a['rule_margins'],a['rule_floors'])):return False
  if snap!='zero' and not(math.isfinite(a['margin']) and a['margin']>=1 and len(a['copies'])==2 and all(r['correct'] is True for r in a['copies'])):return False
 return True

def verify_records(rows,replay,stage1,summary,audits,protection,protocol):
 bodies={};explicit={}
 for c in protection:
  prefix=f"register_00: {c['state'][0]}\nregister_01: {c['state'][1]}\n"
  require(c['prompt'].startswith(prefix),'protection prefix')
  body=c['prompt'][len(prefix):];bodies.setdefault(c['task'],set()).add(body);explicit[c['state'],c['task']]=c['prompt']
 require(len(explicit)==8 and set(bodies)=={'xor','equal'} and all(len(v)==1 for v in bodies.values()),'fixed protection body')
 bodies={k:next(iter(v)) for k,v in bodies.items()}
 require(len(replay)==18,'18 replay rows required')
 index={}
 for r in replay:
  key=(r['phase'],r['id']);require(key not in index,'duplicate replay');index[key]=r
  require(r['phase'] in ('before','after') and r['arm']=='on' and r['snapshot'] in ('zero','10','11'),'replay condition')
  state=None if r['snapshot']=='zero' else r['snapshot']
  require(r['state']==state,'replay state metadata')
  if r['kind']=='copy':
   require(r['cell'] in (0,1) and r['id']==r['snapshot']+'-copy-'+str(r['cell']),'copy identity')
   pred=r['raw'].strip() if not r['truncated'] and r['raw'].strip() in ('0','1') else None
   expected=None if state is None else state[r['cell']]
  else:
   require(r['kind']=='joint' and r['form']==0 and r['id']==r['snapshot']+'-joint-0','joint identity')
   pred=parse_two(r['raw'],r['truncated']);expected=state
  require(r['prediction']==pred and r['expected']==expected and r['format_pass']==(pred is not None) and r['correct']==(None if expected is None else pred==expected),'replay scoring')
 for snap in ('zero','10','11'):
  for suffix in ('joint-0','copy-0','copy-1'):
   a=index['before',snap+'-'+suffix];b=index['after',snap+'-'+suffix]
   require(all(a[k]==b[k] for k in ('raw','output_ids','truncated')),'before/after reading changed')
  require(stage1[snap]==index['before',snap+'-joint-0'],'stage1 source link')
 expected_ids={f'{s}-{c}-{t}' for s in ('zero','10','11') for c in ('raw','flip0','flip1','flip_both','no_intermediate') for t in ('xor','equal')}
 require(len(rows)==30 and {r['id'] for r in rows}==expected_ids,'exact30 plan')
 require({c['id'] for c in protocol['cases']}==expected_ids,'predeclared plan')
 for r in rows:
  snap,cond,task=r['snapshot'],r['condition'],r['task'];require(r['id']==f'{snap}-{cond}-{task}' and r['arm']=='on','case identity')
  src=stage1[snap];raw=src['raw'];bits=parse_two(raw,src['truncated']);memory=None if snap=='zero' else snap
  if cond=='raw':text=raw
  elif cond=='no_intermediate':text=None
  elif bits is None:text=None
  else:text=flip(raw,{'flip0':{0},'flip1':{1},'flip_both':{0,1}}[cond])
  prompt=bodies[task] if cond=='no_intermediate' else None if text is None else text+'\n'+bodies[task]
  supplied=None if text is None else parse_two(text,src['truncated'] if cond=='raw' else False)
  expectation=None if supplied is None else bit_answer(task,supplied)
  memory_expected=None if memory is None else bit_answer(task,memory)
  fields=dict(source_read_id=src['id'],source_raw=raw,source_truncated=src['truncated'],source_raw_sha256=digest(raw.encode()),supplied_text=text,supplied_values=supplied,memory_state=memory,prompt=prompt,expected_from_intermediate=expectation,expected_from_memory=memory_expected,intervention=cond in ('flip0','flip1','flip_both'))
  require(all(r.get(k)==v for k,v in fields.items()),'intermediate provenance or prompt changed')
  require(r['skipped']==(prompt is None),'invalid skip')
  require(not r['skipped'],'this archived run generated every row')
  cfg=r['effective_config'];require(cfg['do_sample'] is False and cfg['num_beams']==1 and cfg['max_new_tokens']==128 and cfg['repetition_penalty']==1.,'decoding changed')
  values,pred=parse_three(r['raw'],r['truncated'])
  source_ok=None if memory is None else bits==memory
  mem_ok=None if memory is None else values==memory and pred==memory_expected
  follow=None if supplied is None else values==supplied and pred==expectation
  fields=dict(prediction=pred,reported_values=values,format_pass=pred is not None,follows_intermediate=follow,memory_answer_correct=None if memory is None else pred==memory_expected,strict_memory_output=mem_ok,source_read_correct=source_ok,end_to_end_success=None if cond!='raw' or memory is None else bool(source_ok and mem_ok and follow))
  require(all(r.get(k)==v for k,v in fields.items()),'logic scoring mismatch')
 groups={}
 for snap in ('zero','10','11'):
  groups[snap]={}
  for cond in ('raw','flip0','flip1','flip_both','no_intermediate'):
   rr=[r for r in rows if(r['snapshot'],r['condition'])==(snap,cond)]
   groups[snap][cond]=dict(n=len(rr),generated=sum(not r['skipped'] for r in rr),format_pass=sum(r['format_pass'] for r in rr),follows_intermediate=sum(r['follows_intermediate'] is True for r in rr),strict_memory_output=sum(r['strict_memory_output'] is True for r in rr),end_to_end_success=sum(r['end_to_end_success'] is True for r in rr))
 require(summary['groups']==groups,'group aggregate mismatch')
 require(audits['before']==audits['after'],'audit records changed')
 require(set(audits['after'])=={'on'} and gate(audits['after']['on']),'CPU finite gates failed')
 for snap in ('10','11'):
  for phase in ('before','after'):
   require(all(c['output_ids']==index[phase,c['id']]['output_ids'] and c['correct']==index[phase,c['id']]['correct'] for c in audits[phase]['on'][snap]['copies']),'audit/replay copy mismatch')
 known=[r for r in rows if r['snapshot']!='zero'];primary=[r for r in known if r['condition']=='raw'];single=[r for r in known if r['condition'] in ('flip0','flip1')];double=[r for r in known if r['condition']=='flip_both'];direct=[r for r in known if r['condition']=='no_intermediate']
 effects=[]
 for r in single+double:
  origin=next(x for x in primary if (x['snapshot'],x['task'])==(r['snapshot'],r['task']))
  prescribed=(r['prediction']==origin['prediction']) if r['condition']=='flip_both' else r['prediction']!=origin['prediction']
  effects.append(dict(id=r['id'],both_format_valid=r['format_pass'] and origin['format_pass'],prescribed_effect=bool(r['follows_intermediate'] and origin['follows_intermediate'] and prescribed)))
 require(summary['intermediate_effects']==effects,'effect aggregate mismatch')
 rawpass=len(primary)==4 and all(r['end_to_end_success'] for r in primary);singlepass=len(single)==8 and all(r['follows_intermediate'] for r in single);doublepass=len(double)==4 and all(r['follows_intermediate'] for r in double)
 require(summary['known_raw_four_pass']==rawpass and summary['known_single_flip_eight_pass']==singlepass and summary['known_double_flip_four_pass']==doublepass,'component gates mismatch')
 require(all(summary[k] is True for k in ('complete','source_integrity','cpu_before_after_stable','frozen_verified','source_restored')),'reported completion/freeze flags')
 require(summary['known_two_stage_gate']==rawpass and summary['known_text_control_gate']==(rawpass and singlepass and doublepass),'final gate mismatch')
 require(summary['formal_l6_pass'] is False and summary['single_pass_l6_pass'] is False,'overclaim flag')
 return dict(status='OFFLINE_RECORD_AUDIT_PASS',independent_model_rerun=False,rows=30,replay_rows=18,
  known_raw_success=sum(r['end_to_end_success'] is True for r in primary),known_raw_total=4,single_text_flip_success=sum(r['follows_intermediate'] is True for r in single),single_text_flip_total=8,double_text_flip_success=sum(r['follows_intermediate'] is True for r in double),double_text_flip_total=4,no_intermediate_strict_success=sum(r['strict_memory_output'] is True for r in direct),no_intermediate_total=4,
  known_two_stage_gate=rawpass,known_text_control_gate=rawpass and singlepass and doublepass,formal_l6_pass=False,single_pass_l6_pass=False,
  limitation='Checks recorded data, provenance links and logged budgets. Does not independently observe execution or recompute model logits/KL.')

def verify(root=ROOT,check_manifest=True):
 if check_manifest:
  manifest=read(root/'SHA256SUMS.json')
  for name,sha in manifest['files'].items():
   path=root/name;require(path.resolve().is_relative_to(root.resolve()),'unsafe manifest path');require(digest(path.read_bytes())==sha,'file checksum: '+name)
 prov=read(root/'PROVENANCE.json');art=root/'artifacts'
 source=art/'qwen_self_read_compute_cpu_v1.zip';result=art/prov['input_result_filename']
 require(digest(source.read_bytes())==prov['source_zip_sha256'] and digest(result.read_bytes())==prov['input_result_sha256'],'original archives')
 with zipfile.ZipFile(source) as zs,zipfile.ZipFile(result) as zr:
  require(zs.testzip() is None and zr.testzip() is None,'archive CRC')
  for n in zs.namelist():
   require(Path(n).name==n,'unsafe source entry');require(zs.read(n)==(root/'source'/n).read_bytes()==zr.read(n),'source/run bytes: '+n)
  for n in zr.namelist():
   if n.startswith('results/'):
    require(Path(n).parent==Path('results'),'unsafe result entry');require((root/n).read_bytes()==zr.read(n),'original result bytes')
 sm=read(root/'source/source_integrity.json')
 for n,sha in sm['files'].items():require(digest((root/'source'/n).read_bytes())==sha,'upstream data hash')
 for n,sha in read(root/'source/package_manifest.json')['files'].items():require(digest((root/'source'/n).read_bytes())==sha,'source package hash')
 proto=read(root/'results/protocol.json')
 for n,sha in proto['code_hashes'].items():require(digest((root/'source'/n).read_bytes())==sha,'runtime code hash')
 get=lambda n:read(root/'results'/f'{n}.json')
 report=verify_records(get('logic_rows'),get('replay_rows'),get('stage1_rows'),get('summary'),get('audits'),read(root/'source/protection_cases.json'),proto)
 report['source_zip_sha256']=prov['source_zip_sha256'];report['result_zip_sha256']=prov['input_result_sha256']
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write-report',action='store_true');p.add_argument('--no-manifest',action='store_true',help='For initial packaging only; skips outer file inventory. Inner archives/source still checked.');args=p.parse_args()
 report=verify(check_manifest=not args.no_manifest)
 if args.write_report:(ROOT/'OFFLINE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
