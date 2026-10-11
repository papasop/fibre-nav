from pathlib import Path
import argparse,json,hashlib,zipfile,time,random,shutil,os,gc
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
import per_address_repair as repair
from protocol import *
from runtime_a100 import a100_device
ROOT=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,default=w.json_default));os.replace(temp,p)
def tensor_save(p,x):
 temp=p.with_suffix('.tmp');torch.save(x,temp);os.replace(temp,p)
def pack(out,drive):
 manifest={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp' and p.name!='MANIFEST.json'}
 dump(out/'MANIFEST.json',manifest)
 dst=out.parent/'gpt2_b4_minimal_causal_a100_results.zip';temp=dst.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p.suffix!='.tmp':z.write(p,str(p.relative_to(out)))
 with zipfile.ZipFile(temp) as z:
  if z.testzip():raise RuntimeError('ZIP CRC failure')
 os.replace(temp,dst)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  backup=d/f'gpt2_b4_minimal_causal_a100_results_{time.time_ns()}.zip';partial=backup.with_suffix('.partial')
  shutil.copyfile(dst,partial)
  if digest(partial)!=digest(dst):raise RuntimeError('Drive backup digest mismatch')
  os.replace(partial,backup)
  print('DRIVE_BACKUP_OK',backup,flush=True)
  for old in sorted(d.glob('gpt2_b4_minimal_causal_a100_results_*.zip'))[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dst),flush=True)

def run_seed(out,drive,seed,global_identity,device,gpu):
 sdir=out/'seeds'/str(seed);sdir.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg['seed']=seed
 cfg['model_revision']=global_identity.get('model_revision','607a30d783dfa663caf39e06633721c8d4cfcd7e')
 w.set_seed(seed)
 print('B4_MINIMAL_CAUSAL loading seed',seed,device,'float32',flush=True)
 tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
 symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
 ah=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
 ident={'seed':seed,'model_revision':getattr(model.config,'_commit_hash',None),'lora_A_sha256':ah.hexdigest(),'symbols':symbols,'chart_shapes':[list(p.shape) for p in params]}
 if ident['model_revision']!=cfg['model_revision']:raise RuntimeError('Model revision mismatch')
 ip=sdir/'identity.json'
 if ip.exists():
  if json.dumps(json.loads(ip.read_text()),sort_keys=True)!=json.dumps(ident,sort_keys=True):raise RuntimeError('Seed identity mismatch')
 else:dump(ip,ident)
 versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 origin=w.param_flat(params).detach().cpu().clone()
 if origin.abs().max().item()!=0:raise RuntimeError('Each seed must begin from actual zero B')
 rng=random.Random(seed);anchors=rng.sample(w.ANCHORS,cfg['anchor_count'])
 response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 with torch.no_grad():
  logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
 cells,exact,train,dev=audit.prompts();fresh=fresh_prompts(cells)
 if set(fresh)&set(exact+train+dev):raise RuntimeError('Evaluation prompt contamination')
 records=json.loads((sdir/'records.json').read_text()) if (sdir/'records.json').exists() else {}
 cases=json.loads((sdir/'cases.json').read_text()) if (sdir/'cases.json').exists() else {}
 def bits():
  with torch.no_grad():return audit.content_bits(w,model,tok,exact,ids,device)
 def read(i):return bits()[i]
 def load(name):return torch.load(sdir/(name+'.pt'),map_location=device,weights_only=True)
 def phase(name,start,target,steps,index):
  state=sdir/(name+'.pt')
  if name in records:
   w.assign_flat(params,load(name));return
  w.assign_flat(params,start.to(device));cp=sdir/(name+'_progress.pt');local=dict(cfg)
  if cp.exists():local['_resume']=torch.load(cp,map_location='cpu',weights_only=True)
  def save(x):
   tensor_save(cp,x)
   dump(out/'progress.json',{'seed':seed,'phase':name,'steps':x['next_step'],'completed_phases':sum(len(json.loads(p.read_text())) for p in (out/'seeds').glob('*/records.json'))})
   print('B4 MINIMAL CAUSAL TRAIN',seed,name,x['next_step'],'/',steps,flush=True);pack(out,drive)
  local['_checkpoint']=save
  result=audit.write_program(w,model,tok,params,local,anchors,response_ids,response0,logits0,exact,train,dev,[ids[b] for b in target],ids,seed+100003*index,device,steps)
  ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device)
  d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  records[name]={'target_bits':target,'observed_bits':bits(),'evaluation':ev,'response_drift':d,'anchor_kl':k,'pass':endpoint_pass(ev,d,k,cfg),'writer':result}
  if any(p._version!=versions[n] for n,p in model.named_parameters() if not p.requires_grad):raise RuntimeError('Frozen parameter changed')
  tensor_save(state,w.param_flat(params).detach().cpu());dump(sdir/'records.json',records);cp.unlink(missing_ok=True);pack(out,drive)
 for case_index,(a,b) in enumerate(CASES):
  key=f'{a}{b}'
  if key in cases:continue
  x=a^b;names={role:key+'_'+role for role in ('prepare','normal_writeback','blocked_writeback','input_substitution','substituted_writeback')}
  index=case_index*5
  phase(names['prepare'],origin,[a,b,1,1^x],cfg['l5_initial_write_steps'],index+1)
  prepared=load(names['prepare']);w.assign_flat(params,prepared)
  normal=controller(read);before=bits();target=before.copy();target[3]=normal['value']
  phase(names['normal_writeback'],prepared,target,cfg['l5_operation_steps'],index+2)
  normal_state=load(names['normal_writeback']);w.assign_flat(params,normal_state)
  actual=bits();blocked=controller(read,blocked=True,replacement=(1^actual[0],actual[1]));target=actual.copy();target[3]=blocked['value']
  phase(names['blocked_writeback'],normal_state,target,cfg['l5_operation_steps'],index+3)
  # Counterfactual branches both start from the same saved correct output state.
  w.assign_flat(params,normal_state);target=bits();target[0]^=1
  phase(names['input_substitution'],normal_state,target,cfg['l5_operation_steps'],index+4)
  substituted=load(names['input_substitution']);w.assign_flat(params,substituted)
  changed=controller(read);before=bits();target=before.copy();target[3]=changed['value']
  phase(names['substituted_writeback'],substituted,target,cfg['l5_operation_steps'],index+5)
  reloads={}
  for role,name in names.items():
   w.assign_flat(params,load(name));r=records[name]
   ev=audit.evaluate(w,model,tok,exact,dev,[ids[t] for t in r['target_bits']],ids,device)
   d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
   reloads[role]=endpoint_pass(ev,d,k,cfg) and bits()==r['observed_bits']
  w.assign_flat(params,origin.to(device));reset_bits=bits();reset_trace=controller(read)
  d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  trace={'normal':normal,'blocked':blocked,'substituted':changed,'reset':reset_trace}
  stages={role:records[name] for role,name in names.items()}
  checks=case_checks(a,b,stages,trace,reset_bits,d<=cfg['response_budget'] and k<=cfg['endpoint_kl_gate'],reloads)
  checks['reset_changes_computed_result']=reset_trace['value']!=(a^b)
  w.assign_flat(params,normal_state)
  restored_bits=bits();restored_trace=controller(read)
  rd,rk=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  checks['restore_recovers_contents_and_computation']=(restored_bits==stages['normal_writeback']['observed_bits'] and restored_trace['value']==(a^b) and rd<=cfg['response_budget'] and rk<=cfg['endpoint_kl_gate'])
  trace['restored']=restored_trace
  cases[key]={'input_bits':[a,b],'traces':trace,'reset_bits':reset_bits,'checks':checks,'pass':all(checks.values()),'reload_checks':reloads}
  dump(sdir/'cases.json',cases);pack(out,drive)
  print('CASE_COMPLETE',seed,key,'pass',cases[key]['pass'],flush=True)
 # The reused expression family is auxiliary development evidence only.
 # It is evaluated after all phases and never used in gradients or selection.
 fresh_records={}
 for name,r in records.items():
  w.assign_flat(params,load(name))
  ev=audit.evaluate(w,model,tok,exact,fresh,[ids[b] for b in r['target_bits']],ids,device)
  d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  fresh_records[name]={'evaluation':ev,'response_drift':d,'anchor_kl':k,'pass':endpoint_pass(ev,d,k,cfg)}
 dump(sdir/'evaluation.json',{'scope':'reused V2 expression family; auxiliary development diagnostic, not a new holdout or acceptance gate','prefixes':FRESH_PREFIXES,'endpoints':fresh_records})
 normal_reset_hits=sum(cases[f'{a}{b}']['traces']['reset']['value']==(a^b) for a,b in CASES)
 gates={'single_case_all_causal_and_endpoint_checks':all(c['pass'] for c in cases.values()),
        'reset_task_accuracy_below_intact':normal_reset_hits<len(CASES)}
 summary={'seed':seed,'development_seed':seed==DEVELOPMENT_SEED,'gates':gates,'pass':all(gates.values()),'reset_task_correct':normal_reset_hits,'reset_task_total':len(CASES),'auxiliary_expression_gates_pass':all(r['pass'] for r in fresh_records.values())}
 dump(sdir/'SUMMARY.json',summary);pack(out,drive)
 del model,params,logits0,response0;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return summary

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 device,gpu=a100_device(torch);repair.install(w)
 identity={'protocol':PROTOCOL,'seeds':SEEDS,'model_revision':'607a30d783dfa663caf39e06633721c8d4cfcd7e','gpu':gpu,'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'peft':peft.__version__,
           'engine_sha256':{n:digest(ROOT/n) for n in ['worker_b4.py','writer_core.py','protocol.py','runtime_a100.py','config_audit.json','run_l5_audit.py','per_address_repair.py']}}
 ip=out/'identity.json'
 if ip.exists() and json.loads(ip.read_text())!=identity:raise RuntimeError('Protocol/source/runtime identity mismatch; no fallback')
 dump(ip,identity);pack(out,args.drive_dir)
 summaries=[]
 for seed in SEEDS:
  f=out/'seeds'/str(seed)/'SUMMARY.json'
  if f.exists():summaries.append(json.loads(f.read_text()));continue
  summaries.append(run_seed(out,args.drive_dir,seed,identity,device,gpu))
 new=[s for s in summaries if not s['development_seed']]
 all_pass=all(s['pass'] for s in summaries)
 summary={'protocol':PROTOCOL,'decision':'B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS' if all_pass else 'B4_MINIMAL_CAUSAL_DEVELOPMENT_FAIL','complete':True,'all_gates_pass':all_pass,'seeds':summaries,
  'n_passed':sum(s['pass'] for s in summaries),'n_planned':len(SEEDS),'new_seed_passed':sum(s['pass'] for s in new),'new_seed_total':len(new),
  'scope':'One development seed82931 and fixed XOR input01; external deterministic XOR; actual parameter reads and writes; required output flips; executed wrong-read branch; actual input substitution; reset and restore; averaged response/KL gates. Not complete B4, four-case coverage, independent confirmation or B5.',
  'b5_claim':False,'external_independent_replication':False,'prior_b4_zip_used':False,
  'full_b4_confirmation':False,'development_only':True,'evaluation_boundary':'Reused V2 expression family is reported only as auxiliary diagnostic. It is not untouched evaluation and does not determine acceptance.'}
 dump(out/'SUMMARY.json',summary);pack(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
