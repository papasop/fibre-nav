from pathlib import Path
import argparse,json,hashlib,zipfile,time,random,shutil,os,platform
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
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=w.json_default));os.replace(q,p)
def tensor_save(p,x):
 q=p.with_suffix('.tmp');torch.save(x,q);os.replace(q,p)
def pack(out,drive):
 files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp']
 manifest={str(p.relative_to(out)):digest(p) for p in files if p.name!='MANIFEST.json'}
 dump(out/'MANIFEST.json',manifest)
 dst=out.parent/'gpt2_b4_a100_results.zip';tmp=dst.with_suffix('.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in out.rglob('*'):
   if p.is_file() and p.suffix!='.tmp':z.write(p,str(p.relative_to(out)))
 with zipfile.ZipFile(tmp) as z:
  if z.testzip():raise RuntimeError('Archive CRC failure')
 os.replace(tmp,dst)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  p=d/f'gpt2_b4_a100_results_{time.time_ns()}.zip';q=p.with_suffix('.partial')
  shutil.copyfile(dst,q)
  if digest(q)!=digest(dst):raise RuntimeError('Drive backup digest mismatch')
  os.replace(q,p);print('DRIVE_BACKUP_OK',p,flush=True)
  old=sorted(d.glob('gpt2_b4_a100_results_*.zip'))
  for p in old[:-3]:p.unlink()
 print('RESULT_ARCHIVE='+str(dst),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 cfg=json.loads((ROOT/'config_audit.json').read_text());seed=cfg['seed']
 repair.install(w);w.set_seed(seed)
 device,gpu_name=a100_device(torch)
 meta=json.loads((out/'identity.json').read_text()) if (out/'identity.json').exists() else None
 if meta:cfg['model_revision']=meta['model_revision']
 print(PROTOCOL,'Loading GPT-2',device,'float32',flush=True)
 tok,model,params=w.make_model(cfg,device)
 model.config.use_cache=False
 a_hash=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:a_hash.update(p.detach().cpu().numpy().tobytes())
 frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
 revision=getattr(model.config,'_commit_hash',None)
 if not revision:raise RuntimeError('Cannot identify model revision; refusing unpinned continuation')
 identity={'protocol':PROTOCOL,'model_revision':revision,'lora_A_sha256':a_hash.hexdigest(),
   'symbols':symbols,'source_sha256':digest(ROOT/'writer_core.py'),
   'worker_sha256':digest(ROOT/'worker_b4.py'),'protocol_sha256':digest(ROOT/'protocol.py'),'runtime_sha256':digest(ROOT/'runtime_a100.py'),
   'chart_shapes':[list(p.shape) for p in params], 'device':str(device),'gpu_name':gpu_name,'cuda':torch.version.cuda,
   'precision':'float32; autocast disabled; TF32 disabled; eager attention',
   'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}
 if meta:
  for k in ('protocol','model_revision','lora_A_sha256','symbols','source_sha256','worker_sha256','protocol_sha256','runtime_sha256','chart_shapes','torch','transformers','peft','device','gpu_name','cuda','precision'):
   if json.dumps(meta[k])!=json.dumps(identity[k]):raise RuntimeError('Resume identity mismatch: '+k)
 else:dump(out/'identity.json',identity)
 rng=random.Random(seed);anchors=rng.sample(w.ANCHORS,cfg['anchor_count'])
 response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 with torch.no_grad():
  logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
 origin=w.param_flat(params).cpu()
 if origin.abs().max().item()!=0:raise RuntimeError('LoRA-B must start at zero')
 cells,exact,train,held=audit.prompts()
 records=json.loads((out/'records.json').read_text()) if (out/'records.json').exists() else {}
 def bits():
  with torch.no_grad():return audit.content_bits(w,model,tok,exact,ids,device)
 def read(i):return bits()[i]
 def phase(name,start,target,steps,index):
  statefile=out/(name+'.pt')
  if name in records:
   w.assign_flat(params,torch.load(statefile,map_location=device,weights_only=True));return
  w.assign_flat(params,start.to(device))
  cp=out/(name+'_progress.pt');local=dict(cfg)
  if cp.exists():local['_resume']=torch.load(cp,map_location='cpu',weights_only=True)
  def save(state):
   tensor_save(cp,state)
   dump(out/'progress.json',{'phase':name,'steps':state['next_step'],'completed_phases':len(records)})
   print('B4 TRAIN',name,state['next_step'],'/',steps,flush=True);pack(out,args.drive_dir)
  local['_checkpoint']=save
  result=audit.write_program(w,model,tok,params,local,anchors,response_ids,response0,logits0,
    exact,train,held,[ids[b] for b in target],ids,seed+100003*index,device,steps)
  ev=audit.evaluate(w,model,tok,exact,held,[ids[b] for b in target],ids,device)
  drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  records[name]={'target_bits':target,'observed_bits':bits(),'evaluation':ev,
    'response_drift':drift,'anchor_kl':kl,'pass':endpoint_pass(ev,drift,kl,cfg),'writer':result}
  if any(p._version!=frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad):
   raise RuntimeError('Frozen parameter changed')
  tensor_save(statefile,w.param_flat(params).cpu());dump(out/'records.json',records)
  cp.unlink(missing_ok=True);pack(out,args.drive_dir)
 def load(name):return torch.load(out/(name+'.pt'),map_location=device,weights_only=True)
 phase('prepare',origin,INITIAL,cfg['l5_initial_write_steps'],1)
 if not records['prepare']['pass']:
  dump(out/'SUMMARY.json',{'protocol':PROTOCOL,'decision':'B3_REBUILD_NOT_READY','b4_pass':False,
    'development_only':True});pack(out,args.drive_dir);return
 prepared=load('prepare');w.assign_flat(params,prepared)
 normal=controller(read);normal_before=bits();normal_target=normal_before.copy();normal_target[3]=normal['value']
 phase('normal_writeback',prepared,normal_target,cfg['l5_operation_steps'],2)
 normal_after=bits()
 phase('substitute_input',prepared,SUBSTITUTED,cfg['l5_operation_steps'],3)
 if not records['substitute_input']['pass']:
  dump(out/'SUMMARY.json',{'protocol':PROTOCOL,'decision':'SUBSTITUTION_NOT_READY','b4_pass':False,
    'development_only':True});pack(out,args.drive_dir);return
 substituted=load('substitute_input');w.assign_flat(params,substituted)
 changed=controller(read);sub_before=bits();target=sub_before.copy();target[3]=changed['value']
 phase('substituted_writeback',substituted,target,cfg['l5_operation_steps'],4)
 sub_after=bits()
 # Independent forward reload of saved endpoints; no prompt contains oracle task values.
 reload_checks={}
 for name,target in [('normal_writeback',normal_target),('substituted_writeback',target)]:
  w.assign_flat(params,load(name))
  ev=audit.evaluate(w,model,tok,exact,held,[ids[b] for b in target],ids,device)
  d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  reload_checks[name]=endpoint_pass(ev,d,k,cfg)
 w.assign_flat(params,origin.to(device));reset_trace=controller(read);reset_bits=bits()
 reset_d,reset_k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
 w.assign_flat(params,substituted);blocked=controller(read,blocked=True)
 causal=normal['value']==0 and changed['value']==1 and blocked['value']!=changed['value']
 checks={'all_endpoint_gates':all(r['pass'] for r in records.values()),
   'normal_task':normal_after[3]==0,'substituted_task':sub_after[3]==1,
   'normal_unselected':normal_after[:3]==normal_before[:3],
   'substituted_unselected':sub_after[:3]==sub_before[:3],
   'causal_input_substitution':causal,'no_write_control_detected':normal_before[3]!=0,
   'reset_task_sensitivity':reset_trace['value']!=normal['value'] or reset_trace['value']!=changed['value'],
   'baseline_reset':reset_d<=cfg['response_budget'] and reset_k<=cfg['endpoint_kl_gate'],
   'endpoint_reload':all(reload_checks.values())}
 dump(out/'causal_traces.json',{'normal':normal,'substituted':changed,'blocked':blocked,
   'reset':reset_trace,'reset_bits':reset_bits,'reload_checks':reload_checks})
 dump(out/'SUMMARY.json',{'protocol':PROTOCOL,'decision':'B4_MINIMAL_DEVELOPMENT_PASS' if all(checks.values()) else 'B4_DEVELOPMENT_FAIL',
   'b4_pass':all(checks.values()),'checks':checks,'development_only':True,'independent_confirmation':False,
   'executor':'external XOR controller','b5_claim':False,'heldout_scope':'previously inspected B3 expressions, development checks',
   'scope':'Two XOR input states; four fixed-access shared LoRA-B cells; declared averaged response and KL budgets'})
 pack(out,args.drive_dir)
 print((out/'SUMMARY.json').read_text(),flush=True)
if __name__=='__main__':main()
