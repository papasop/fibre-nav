from pathlib import Path
import os,json,time,zipfile,shutil,random,argparse,hashlib,gc,fcntl
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
import per_address_repair as repair
from runtime_a100 import a100_device
from b5_protocol import *
from protocol import endpoint_pass
from controller_data import prompt as controller_prompt
from controller_train import train_controller
from adapter_runtime import activate
from truth_protocol import INPUTS,preparation_target
ROOT=Path(__file__).resolve().parent

def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
 q.write_text(json.dumps(x,indent=2,default=w.json_default));os.replace(q,p)
def save_tensor(p,x):
 q=p.with_suffix('.tmp');torch.save(x,q);os.replace(q,p)
def pack(out,drive):
 m={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and p.suffix!='.tmp'}
 dump(out/'MANIFEST.json',m)
 dest=out.parent/'gpt2_b5_xor_truth_table_a100_results.zip';temp=dest.with_suffix('.tmp')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for n in list(m)+['MANIFEST.json']:z.write(out/n,n)
 verified_zip(temp,PROTOCOL);os.replace(temp,dest)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True)
  dest2=d/f'gpt2_b5_xor_truth_table_a100_results_{time.time_ns()}.zip';q=dest2.with_suffix('.partial')
  shutil.copyfile(dest,q)
  if sha(q)!=sha(dest):raise RuntimeError('Drive copy mismatch')
  os.replace(q,dest2);print('DRIVE_BACKUP_OK',dest2,flush=True)
  for old in sorted(d.glob('gpt2_b5_xor_truth_table_a100_results_*.zip'))[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dest),flush=True)

@torch.no_grad()
def select_action(model,tok,prompt,device):
 # Fixed grammar, all actions considered at every state, no oracle/state masks.
 # Scores come from GPT-2 token logits, not a Python task planner.
 prefix=tok.encode(prompt,add_special_tokens=False)
 scores={}
 for action in ACTIONS:
  continuation=tok.encode(' '+action+'\n',add_special_tokens=False)
  if len(prefix)+len(continuation)>model.config.n_positions:raise RuntimeError('Policy context overflow')
  x=torch.tensor([prefix+continuation],device=device)
  logits=model(input_ids=x,use_cache=False).logits[0]
  lp=logits[len(prefix)-1:len(prefix)+len(continuation)-1].log_softmax(-1)
  y=torch.tensor(continuation,device=device)
  scores[action]=float(lp.gather(1,y[:,None]).mean().item())
 return max(ACTIONS,key=lambda a:scores[a]),scores

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 device,gpu=a100_device(torch);repair.install(w)
 verified_zip(out/'parent_v3.zip','GPT2_B5_SINGLE_BACKBONE_A100_V3')
 with zipfile.ZipFile(out/'parent_v3.zip') as z:
  if json.loads(z.read('SUMMARY.json'))['decision']!='B5_SINGLE_BACKBONE_DEVELOPMENT_PASS':raise RuntimeError('V3 parent did not pass')
 verified_zip(out/'parent_b4.zip','GPT2_B4_MINIMAL_CAUSAL_A100_V1')
 parent=out/'parent';parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(out/'parent_b4.zip') as z:
  summary=json.loads(z.read('SUMMARY.json'))
  if summary['decision']!='B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS':raise RuntimeError('Parent B4 did not pass')
  for name in ('writer_core.py','run_l5_audit.py','per_address_repair.py','config_audit.json'):
   if z.read('source/'+name)!=(ROOT/name).read_bytes():raise RuntimeError('Parent writer source mismatch: '+name)
  for n in ('seeds/82931/identity.json','seeds/82931/01_prepare.pt','seeds/82931/01_input_substitution.pt'):
   q=parent/Path(n).name
   if not q.exists():q.write_bytes(z.read(n))
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=SEED,model_revision=REVISION)
 w.set_seed(SEED);print('B5 loading pinned GPT-2; existing B4 chart; memory chart frozen except original B4 writer',flush=True)
 tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
 ah=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
 pi=json.loads((parent/'identity.json').read_text())
 if ah.hexdigest()!=pi['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:raise RuntimeError('Parent model/chart identity mismatch')
 symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
 if json.loads(json.dumps(symbols))!=pi['symbols']:raise RuntimeError('Read token identity mismatch')
 frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 origin=w.param_flat(params).detach().cpu().clone()
 if origin.abs().max().item()!=0:raise RuntimeError('Unexpected nonzero initial B')
 rng=random.Random(SEED);anchors=rng.sample(w.ANCHORS,cfg['anchor_count']);response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
 _,exact,train,dev=audit.prompts()
 ident={'protocol':PROTOCOL,'parent_sha256':sha(out/'parent_b4.zip'),'parent_v3_sha256':sha(out/'parent_v3.zip'),'gpu':gpu,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'source_hashes':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file()},'policy':'single GPT-2 backbone; two exclusive adapters routed by runtime; controller trained on other task tuples','max_actions':MAX_ACTIONS,'seed':SEED}
 ip=out/'identity.json'
 if ip.exists() and json.loads(ip.read_text())!=ident:raise RuntimeError('B5 source/runtime identity mismatch; no fallback')
 dump(ip,ident)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__'))
 pack(out,args.drive_dir)
 policy_tok,policy_model,readiness=train_controller(cfg,device,out,save_tensor,dump,pack,args.drive_dir,select_action,tok,model)
 if policy_model is not model:raise RuntimeError('Unexpected second backbone')
 dump(out/'ARCHITECTURE.json',{'backbone_instances':1,'adapters':['default(memory)','controller'],'shared_update_chart':False,'runtime_routes_adapter':True,'internal_parameter_update':False})
 if not readiness['pass']:
  dump(out/'SUMMARY.json',{'protocol':PROTOCOL,'complete':True,'decision':'CONTROLLER_NOT_READY',
       'development_only':True,'full_b5_confirmation':False,'internal_parameter_update':False,
       'physical_memory_test_executed':False,'reason':'held-out symbolic controller gate failed; no physical writes performed'})
  pack(out,args.drive_dir);print('V2 CONTROLLER_NOT_READY: physical writes skipped',flush=True);return
 print('V2 CONTROLLER_READY: begin actual B4 memory evaluation',flush=True)
 def bits():
  activate(model,'default',False)
  with torch.no_grad():return audit.content_bits(w,model,tok,exact,ids,device)
 def evaluate(target):
  activate(model,'default',False)
  ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device)
  d,k=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  return {'evaluation':ev,'drift':d,'kl':k,'pass':endpoint_pass(ev,d,k,cfg),'observed_bits':bits()}
 inputs=INPUTS
 summary_cases={}
 for a,b in inputs:
  case=f'xor_{a}{b}';target=preparation_target(a,b)
  cd=out/case;cd.mkdir(exist_ok=True);rp=cd/'record.json'
  if (cd/'COMPLETE.json').exists():summary_cases[case]=json.loads((cd/'COMPLETE.json').read_text());continue
  if (cd/'state.pt').exists():
   committed=torch.load(cd/'state.pt',map_location=device,weights_only=True)
   r=committed['record'];w.assign_flat(params,committed['theta'])
   if (cd/'pending_action.json').exists() and json.loads((cd/'pending_action.json').read_text()).get('step',-1)<len(r['trace']):
    (cd/'pending_action.json').unlink();(cd/'pending_write.pt').unlink(missing_ok=True)
  else:
   prep_file=cd/'PREPARATION.json';prep_theta=cd/'prepared.pt';prep_cp=cd/'preparation_optimizer.pt'
   if prep_file.exists():
    prep=json.loads(prep_file.read_text())
    w.assign_flat(params,torch.load(prep_theta,map_location=device,weights_only=True))
   else:
    w.assign_flat(params,origin.to(device));local=dict(cfg)
    if prep_cp.exists():local['_resume']=torch.load(prep_cp,map_location='cpu',weights_only=True)
    def prep_checkpoint(x):
     save_tensor(prep_cp,x);print('V4 PREPARE',case,x['next_step'],flush=True);pack(out,args.drive_dir)
    local['_checkpoint']=prep_checkpoint
    activate(model,'default',True)
    wr=audit.write_program(w,model,tok,params,local,anchors,response_ids,response0,logits0,exact,train,dev,[ids[v] for v in target],ids,SEED+2000+int(case[-2:],2),device,600)
    gate=evaluate(target);ok=gate['pass'] and gate['observed_bits']==target
    save_tensor(prep_theta,w.param_flat(params).detach().cpu())
    prep={'target_bits':target,'gate':gate,'pass':ok,'writer':wr,'from_zero_B':True}
    dump(prep_file,prep);pack(out,args.drive_dir)
   if not prep['pass']:
    result={'case':case,'checks':{'preparation_gate':False},'pass':False,'final_bits':bits(),'final_gate':prep['gate'],'trace':[],'reason':'PREPARATION_FAILED; remaining inputs still attempted'}
    dump(cd/'COMPLETE.json',result);summary_cases[case]=result;pack(out,args.drive_dir);continue
   initial=bits()
   if initial!=target:raise RuntimeError('Prepared checkpoint identity mismatch')
   r={'start_bits':initial,'registers':{},'trace':[],'halted':False,'writer_records':[]}
   save_tensor(cd/'state.pt',{'theta':w.param_flat(params).detach().cpu(),'record':r});dump(rp,r);pack(out,args.drive_dir)
  def write(addr,value):
   # Declared task capability guard; rejects wrong addresses, never substitutes an action.
   if addr!='11':return False
   if len(r['writer_records'])>=2:return False
   target=bits();target[int(addr,2)]=value
   before=w.param_flat(params).detach().cpu().clone();local=dict(cfg);cp=cd/'pending_write.pt'
   if cp.exists():local['_resume']=torch.load(cp,map_location='cpu',weights_only=True)
   def checkpoint(x):
    save_tensor(cp,x);print('B5 WRITE',case,addr,'step',x['next_step'],flush=True);pack(out,args.drive_dir)
   local['_checkpoint']=checkpoint
   activate(model,'controller',False)
   probe_prompt=controller_prompt(('00','01','11'),[])
   _,scores_before=select_action(model,tok,probe_prompt,device)
   activate(model,'default',True)
   result=audit.write_program(w,model,tok,params,local,anchors,response_ids,response0,logits0,exact,train,dev,[ids[b] for b in target],ids,SEED+len(r['trace'])*101,device,cfg['l5_operation_steps'])
   ev=evaluate(target);ok=ev['pass'] and ev['observed_bits']==target
   r['writer_records'].append({'address':addr,'value':value,'target_bits':target,'gate':ev,'writer':result,'accepted':ok})
   if not ok:w.assign_flat(params,before.to(device))
   activate(model,'controller',False)
   _,scores_after=select_action(model,tok,probe_prompt,device)
   delta=max(abs(scores_before[a]-scores_after[a]) for a in ACTIONS)
   r['writer_records'][-1]['controller_probe_score_drift']=delta
   if delta>1e-5:raise RuntimeError('Controller scores changed across memory write')
   activate(model,'default',False)
   return ok
  for step in range(len(r['trace']),MAX_ACTIONS):
   if r['halted']:break
   if (cd/'pending_action.json').exists():selection=json.loads((cd/'pending_action.json').read_text())
   else:
    prompt=controller_prompt(('00','01','11'),r['trace'])
    activate(model,'controller',False)
    action,scores=select_action(policy_model,policy_tok,prompt,device);selection={'action':action,'scores':scores,'prompt':prompt,'step':step}
    dump(cd/'pending_action.json',selection);pack(out,args.drive_dir)
   action=selection['action']
   feedback=dispatch(action,r['registers'],lambda addr:bits()[int(addr,2)],write)
   r['trace'].append(dict(selection,feedback=feedback,step=step+1,actual_bits_after=bits()))
   r['halted']=action=='HALT';save_tensor(cd/'state.pt',{'theta':w.param_flat(params).detach().cpu(),'record':r});dump(rp,r)
   (cd/'pending_write.pt').unlink(missing_ok=True);(cd/'pending_action.json').unlink(missing_ok=True)
   print('B5 ACTION',case,step+1,action,'=>',feedback,flush=True);pack(out,args.drive_dir)
  if any(p._version!=frozen_versions[n] for n,p in model.named_parameters() if n in frozen_versions):raise RuntimeError('Frozen base or A changed')
  final=bits();gate=evaluate(final);checks=acceptance(r['trace'],r['start_bits'],final,gate['pass'],r['halted']);checks['no_runtime_errors']=not any(t['feedback'].startswith('ERROR') for t in r['trace']);checks['controller_probe_preserved']=all(t['controller_probe_score_drift']<=1e-5 for t in r['writer_records'])
  result={'case':case,'checks':checks,'pass':all(checks.values()),'final_bits':final,'final_gate':gate,'trace':r['trace']}
  dump(cd/'COMPLETE.json',result);summary_cases[case]=result;pack(out,args.drive_dir)
 passed=len(summary_cases)==4 and all(v['pass'] for v in summary_cases.values())
 result={'protocol':PROTOCOL,'complete':True,'decision':'B5_ACTUAL_XOR_TRUTH_TABLE_DEVELOPMENT_PASS' if passed else 'B5_ACTUAL_XOR_TRUTH_TABLE_DEVELOPMENT_FAIL',
         'cases':{k:{x:v[x] for x in ('pass','checks','final_bits')} for k,v in summary_cases.items()},
         'n_planned':4,'n_completed':len(summary_cases),'n_passed':sum(v['pass'] for v in summary_cases.values()),
         'controller_new_training_steps':0,'full_b5_confirmation':False,'development_only':True,'internal_parameter_update':False,'single_backbone':True,'shared_update_chart':False,
         'scope':'Actual XOR four-input development coverage with frozen V3 controller seed82933 and reused memory chart seed82931. Each case prepared from zero B with deliberately wrong output. Original B4 writer and dev gates; external routing/XOR/gradients. No independent memory seeds or fresh evaluation; prior V3 causal controls retained in parent archive, not rerun here.'}
 dump(out/'SUMMARY.json',result);pack(out,args.drive_dir);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
