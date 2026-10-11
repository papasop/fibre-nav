"""Read-only verification of recorded evidence. No GPU, model load or training."""
from pathlib import Path
import json,hashlib,zipfile,sys
root=Path(__file__).resolve().parent
p=Path(sys.argv[1]) if len(sys.argv)>1 else root/'gpt2_b5_single_backbone_a100_results.zip'
def require(condition,message):
 if not condition:raise RuntimeError(message)
with zipfile.ZipFile(p) as z:
 names=z.namelist();require(len(names)==len(set(names)),'Duplicate members')
 require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names),'Unsafe paths')
 require(z.testzip() is None,'CRC failure')
 manifest=json.loads(z.read('MANIFEST.json'))
 require(set(n for n in names if not n.endswith('/'))==set(manifest)|{'MANIFEST.json'},'Manifest coverage')
 for n,h in manifest.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'Digest: '+n)
 read=lambda n:json.loads(z.read(n))
 identity=read('identity.json');summary=read('SUMMARY.json');cfg=read('source/config_audit.json')
 require(identity['protocol']=='GPT2_B5_SINGLE_BACKBONE_A100_V3','Protocol')
 require(hashlib.sha256(z.read('parent_b4.zip')).hexdigest()==identity['parent_sha256'],'Parent identity')
 split=read('controller_split.json');data=read('controller_training_data.json')
 heldout={tuple(t) for t in split['heldout_tuples']};train={tuple(t) for t in split['train_tuples']}
 require(not heldout & train and len(train)==22 and len(data)==440,'Data split')
 require(all(tuple(r['task']) in train for r in data),'Training tuple contamination')
 progress=read('controller_progress.json');require(progress['done'] and progress['steps']==600,'Training incomplete')
 readiness=read('controller_readiness.json');require(len(readiness['episodes'])==8,'Readiness count')
 keys=set()
 for e in readiness['episodes']:
  task=tuple(e['task']);x,y=e['input_bits'];require(task in heldout,'Readiness tuple')
  keys.add((task,x,y))
  actions=[t['action'] for t in e['trace']]
  require(actions==[f'READ {task[0]} R0',f'READ {task[1]} R1','XOR R0 R1 R2',f'WRITE {task[2]} R2','HALT'],'Readiness actions')
  feedback=[t['feedback'] for t in e['trace']]
  require(feedback==[f'R0={x}',f'R1={y}',f'R2={x^y}','WRITE accepted','done'],'Readiness feedback')
  require(e['pass'],'Readiness failure')
 require(len(keys)==8 and readiness['pass'],'Readiness coverage')
 architecture=read('ARCHITECTURE.json')
 require(architecture['backbone_instances']==1 and architecture['runtime_routes_adapter'] and not architecture['shared_update_chart'],'Architecture scope')
 records={}
 for case in ('normal','stored_substitution','reset','restored'):
  r=read(case+'/COMPLETE.json');state=read(case+'/record.json');records[case]=r
  initial=state['start_bits'];expected=initial.copy();expected[3]=initial[0]^initial[1]
  require(r['final_bits']==expected,'Wrong XOR result '+case)
  require(r['final_bits'][:3]==initial[:3],'Retention '+case)
  trace=r['trace'];require([t['action'] for t in trace]==['READ 00 R0','READ 01 R1','XOR R0 R1 R2','WRITE 11 R2','HALT'],'Physical actions '+case)
  require([t['feedback'] for t in trace]==[f'R0={initial[0]}',f'R1={initial[1]}',f'R2={expected[3]}','WRITE accepted','done'],'Physical feedback '+case)
  gate=r['final_gate'];ev=gate['evaluation']
  require(gate['pass'] and ev['exact_accuracy']==1 and ev['heldout_accuracy']==1,'Read gate '+case)
  require(gate['drift']<=cfg['response_budget'] and gate['kl']<=cfg['endpoint_kl_gate'],'Budget '+case)
  require(all(w['controller_probe_score_drift']<=1e-5 for w in state['writer_records']),'Controller probe '+case)
  require(r['checks']['controller_probe_preserved'],'Controller preservation check '+case)
  require(r['pass']==(case!='reset'),'Case status '+case)
  require(summary['cases'][case]['checks']==r['checks'],'Summary checks '+case)
 reads=lambda c:[(t['action'],t['feedback']) for t in records[c]['trace'][:2]]
 controls={'reset_changes_read':reads('reset')!=reads('normal'),'restore_recovers_read':reads('restored')==reads('normal'),'stored_substitution_changes_read':reads('stored_substitution')!=reads('normal')}
 require(controls==summary['controls'] and all(controls.values()),'Controls')
 require(summary['decision']=='B5_SINGLE_BACKBONE_DEVELOPMENT_PASS' and summary['complete'],'Decision')
 require(summary['single_backbone'] and not summary['shared_update_chart'],'Single-backbone scope')
 require(summary['development_only'] and not summary['full_b5_confirmation'] and not summary['internal_parameter_update'],'Scope')
 print('RECORDED_EVIDENCE_VERIFIED:',len(manifest),'hashes; readiness 8/8; physical cases and controls consistent')
 print('No new model evaluation; no independent B5 confirmation.')
