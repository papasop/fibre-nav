"""No-training, no-correction diagnosis of saved C15 checkpoints and states."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from proposer import WriterHead,one_forward_proposal
from raw_writer import StateWriter,prepared_input,route_from_features,one_raw_proposal
from c16_protocol import (PROTOCOL,RESULT,SETTINGS,ARMS,REVISION,C9_SHA,HEAD_SHA,CHECKPOINT,FINAL_HEAD_SHA,
 state_specs,source_hashes,inherited_check,validate_parent,sha,endpoint_gate,episode_gate,target_from_reads,
 snapshot,cohort_counts,paired,interpretation)
ROOT=Path(__file__).resolve().parent

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)

def save_tensor(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 torch.save(value,temp);os.replace(temp,path)

def load_obj(z,name):return torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)

def vector(value):
 if isinstance(value,dict):value=value['theta']
 if not isinstance(value,torch.Tensor) or value.shape!=(36864,) or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Original finite float32 chart required')
 return value.detach().cpu().clone()

def head_hash(head):return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in head.parameters())).hexdigest()

def replay_match(actual,recorded):
 if actual.get('bits')!=recorded.get('bits') or actual.get('episode_pass')!=recorded.get('episode_pass'):return False
 if actual.get('gates')!=recorded.get('gates'):return False
 return all(abs(actual[k]-recorded[k])<=SETTINGS['recorded_metric_tolerance'] for k in ('drift','kl'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inherited_check(ROOT);info=validate_parent(out/'parent_c15.zip');device,gpu=a100_device(torch)
 parent_sha=sha(out/'parent_c15.zip')
 ident={'protocol':PROTOCOL,'settings':SETTINGS,'parent_c15_sha256':parent_sha,'source_hashes':source_hashes(ROOT),
  'gpu':gpu,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
  'parent_runtime':{k:info['identity'].get(k) for k in ('torch','numpy','transformers','peft')},'model_revision':REVISION}
 if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('C16 runtime/source changed; refuse guessed resume')
 dump(out/'identity.json',ident)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_package.py'))
 dump(out/'SPLIT.json',{'saved_state_specs':state_specs(),'arms':list(ARMS),'same_states_all_arms':True,'no_new_state_generation':True})
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,args.drive_dir);return
 snapshot(out,args.drive_dir)
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
 w.set_seed(82931);print('C16 Loading pinned GPT-2 once; replay warm/final/C9 heads; no new training.',flush=True)
 tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
 for p in model.parameters():p.requires_grad_(False)
 base_versions={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
 pins=json.loads((ROOT/'MODEL_PINS.json').read_text());ah=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
 if ah.hexdigest()!=pins['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:raise RuntimeError('Original model/A identity mismatch')
 if [list(p.shape) for p in params]!=pins['chart_shapes'] or w.param_flat(params).abs().max()!=0:raise RuntimeError('Original chart/zero-B changed')
 symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
 if json.loads(json.dumps(symbols))!=pins['symbols']:raise RuntimeError('Original read symbols changed')
 rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count']);response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
 _,exact,_,dev=audit.prompts();exact_batch=tok(exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
 starts=pins['starts'];targets={c:target_from_reads(starts[c]) for c in starts}
 def sync():torch.cuda.synchronize(device)

 with zipfile.ZipFile(out/'parent_c15.zip') as parent:
  with zipfile.ZipFile(io.BytesIO(parent.read('parent_c9.zip'))) as c9:basecp=load_obj(c9,CHECKPOINT)
  warmcp=load_obj(parent,info['checkpoints']['warm']['member']);finalcp=load_obj(parent,info['checkpoints']['functional']['member'])
  train_ids=json.loads(parent.read('TRAINING_SET.json'))['training_ids']
  if any(cp.get('train_ids')!=train_ids for cp in (warmcp,finalcp)):raise RuntimeError('Training reference identities differ between checkpoints')
  if warmcp.get('steps')!=1000 or finalcp.get('steps')!=720 or warmcp.get('stage')!='warm' or finalcp.get('stage')!='functional':raise RuntimeError('Checkpoint stage metadata mismatch')
  extra_norm={k:v.to(device) for k,v in warmcp['extra_norm'].items()}
  if set(extra_norm)!={'mean','scale'} or extra_norm['mean'].shape!=(70,) or extra_norm['scale'].shape!=(70,) or any(not torch.isfinite(v).all() for v in extra_norm.values()) or not (extra_norm['scale']>0).all():raise RuntimeError('Invalid training-only extra normalisation')
  if any(not torch.equal(finalcp['extra_norm'][k].to(device),v) for k,v in extra_norm.items()):raise RuntimeError('Normalisation changed between stages')
  c9norm={k:v.to(device) for k,v in basecp['normalisation'].items()}
  sketchcp=load_obj(parent,'STATE_SKETCH.pt');sketch=sketchcp['matrix'].to(device)
  if sketch.shape!=(36864,64) or sketch.dtype!=torch.float32 or not torch.isfinite(sketch).all() or sketchcp['seed']!=615031:raise RuntimeError('Saved fixed sketch identity invalid')
  heads={'c9_baseline':WriterHead(3077,36864,32).to(device),'warm1000':StateWriter(3147,36864,64).to(device),'functional720':StateWriter(3147,36864,64).to(device)}
  heads['c9_baseline'].load_state_dict(basecp['head']);heads['warm1000'].load_state_dict(warmcp['head']);heads['functional720'].load_state_dict(finalcp['head'])
  for h in heads.values():
   h.eval()
   for p in h.parameters():
    if p.dtype!=torch.float32 or not torch.isfinite(p).all():raise RuntimeError('Original finite float32 head parameters required')
    p.requires_grad_(False);p.grad=None
  hashes={a:head_hash(h) for a,h in heads.items()}
  if hashes['c9_baseline']!=HEAD_SHA or hashes['functional720']!=FINAL_HEAD_SHA:raise RuntimeError('Actual frozen head identity mismatch')
  if finalcp.get('start_head_sha256')!=hashes['warm1000']:raise RuntimeError('Functional stage did not start from actual saved warm head')
  checkpoint_record={'head_sha256':hashes,'checkpoints':info['checkpoints'],'functional_starts_from_warm':True,'no_optimizer_restored':True}
  if (out/'CHECKPOINT_IDENTITIES.json').exists() and json.loads((out/'CHECKPOINT_IDENTITIES.json').read_text())!=checkpoint_record:raise RuntimeError('Checkpoint identities changed on resume')
  dump(out/'CHECKPOINT_IDENTITIES.json',checkpoint_record);print('C16 ACTUAL HEAD IDENTITIES',json.dumps(hashes),flush=True)
  del warmcp,finalcp,basecp
  head_versions={a:[p._version for p in h.parameters()] for a,h in heads.items()}
  def frozen_check():
   if any(p._version!=base_versions[n] for n,p in model.named_parameters() if n in base_versions) or any(p.requires_grad for p in model.parameters()):raise RuntimeError('Frozen base/A changed or unexpected training')
   for a,h in heads.items():
    if [p._version for p in h.parameters()]!=head_versions[a] or any(p.requires_grad or p.grad is not None for p in h.parameters()):raise RuntimeError('A diagnostic head changed')

  @torch.no_grad()
  def evaluate(theta,target,start=None):
   if not torch.isfinite(theta).all():raise RuntimeError('Nonfinite proposal; no replacement candidate allowed')
   w.assign_flat(params,theta.to(device));bits=audit.content_bits(w,model,tok,exact,ids,device)
   ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device)
   drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
   v={'bits':bits,'evaluation':ev,'drift':float(drift),'kl':float(kl)};v['pass']=endpoint_gate(v,target)
   v['gates']={'exact_read_margin':min(ev['exact_margins'])>=.05,'development_read_margin':min(ev['heldout_margins'])>=.05,
    'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005}
   if start is not None:
    v['gates'].update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3]);v['episode_pass']=episode_gate(start,v)
   per_logits=w.next_logits(model,tok,anchors,device)
   per_R=(per_logits[:,response_ids]-logits0[:,response_ids]).abs().max(1).values
   per_KL=(logits0.softmax(-1)*(logits0.log_softmax(-1)-per_logits.log_softmax(-1))).sum(-1)
   v['per_anchor_diagnostics']={'selected_logit_max_abs_changes':per_R.cpu().tolist(),'KL_values':per_KL.cpu().tolist(),
    'max_selected_logit_change':float(per_R.max()),'max_KL':float(per_KL.max()),'not_additional_gates':True}
   frozen_check();return v

  @torch.no_grad()
  def fresh_inputs(theta,desired):
   w.assign_flat(params,theta.to(device));result=model(**exact_batch,use_cache=False,output_hidden_states=True)
   last=exact_batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=device)
   hidden=result.hidden_states[-1][rows,last].reshape(-1);logits=result.logits[rows,last][:,ids]
   feat=torch.cat([hidden,logits[:,1]-logits[:,0],torch.tensor([desired],device=device,dtype=hidden.dtype)])
   # Match C15's physical response measurement path and fixed state sketch.
   response=w.response_vector(model,tok,anchors,response_ids,device)
   residual=(response-response0).cpu().double().numpy()
   extra=torch.cat([torch.from_numpy(residual).float().to(device)/.002,theta.to(device)@sketch])
   x=prepared_input(feat,extra,c9norm,extra_norm)
   return feat.cpu(),extra.cpu(),x.cpu(),route_from_features(feat).cpu()

  rows=[]
  for index,t in enumerate(state_specs()):
   sid=t['id'];cohort=t['cohort'];case=t['case'];folder=out/'evaluation'/cohort/sid;folder.mkdir(parents=True,exist_ok=True)
   complete=folder/'COMPLETE.json'
   if complete.exists():rows.append(json.loads(complete.read_text()));continue
   statebase='states/'+cohort+'/'+sid+'/';evalbase='evaluation/'+cohort+'/'+sid+'/'
   prep=vector(load_obj(parent,statebase+'prepared.pt'));teacher=vector(load_obj(parent,statebase+'teacher.pt'))
   ref=json.loads(parent.read(statebase+'REFERENCE.json'));online=load_obj(parent,evalbase+'ONLINE_INPUT.pt');train_features=load_obj(parent,statebase+'FEATURES.pt')
   if online['input'].shape!=(3147,) or online['features'].shape!=(3077,) or online['route'].numel()!=1 or any(not torch.isfinite(online[k]).all() for k in ('input','features')):raise RuntimeError('Original online input invalid')
   auditfile=folder/'STATE_AUDIT.json'
   if auditfile.exists():state_audit=json.loads(auditfile.read_text())
   else:
    sync();ts=time.perf_counter();pm=evaluate(prep,starts[case]);tm=evaluate(teacher,targets[case],starts[case]);feat,extra,x,route=fresh_inputs(prep,targets[case][3]);sync()
    fresh_input_error=float((x-online['input']).abs().max());feature_error=float((feat-online['features']).abs().max())
    old_train_x=prepared_input(train_features['features'].to(device),train_features['extras'].to(device),c9norm,extra_norm).cpu()
    state_audit={'preparation':pm,'teacher':tm,'parent_reference_valid':ref['valid'],'actual_reference_valid':bool(pm['pass'] and tm['episode_pass']),
     'fresh_feature_max_error':feature_error,'fresh_input_max_error':fresh_input_error,'training_vs_online_input_max_error':float((old_train_x-online['input']).abs().max()),
     'route_matches_fresh_features':int(route)==int(online['route']),
     'saved_route_matches_saved_features':int(route_from_features(online['features']))==int(online['route']),
     'fresh_input_used_for_primary_comparison':False,'elapsed_seconds':time.perf_counter()-ts}
    save_tensor(folder/'FRESH_INPUT.pt',{'features':feat,'extras':extra,'input':x,'route':route});dump(auditfile,state_audit)
   arm_results={}
   for arm in ARMS:
    path=folder/arm;path.mkdir(exist_ok=True);resultfile=path/'RESULT.json'
    if resultfile.exists():r=json.loads(resultfile.read_text())
    else:
     proposalfile=path/'PROPOSAL.pt'
     if proposalfile.exists():proposal=vector(torch.load(proposalfile,map_location='cpu',weights_only=True))
     else:
      if arm=='c9_baseline':delta=one_forward_proposal(heads[arm],online['features'].to(device),c9norm).cpu()
      else:delta=one_raw_proposal(heads[arm],online['input'].to(device),online['route'].to(device),c9norm['output_scale']).cpu()
      proposal=prep+delta;save_tensor(proposalfile,{'theta':proposal,'head_sha256':hashes[arm]})
     metrics=evaluate(proposal,targets[case],starts[case]);error=(proposal-teacher).double()
     teacher_delta=(teacher-prep).double();pred_delta=(proposal-prep).double();den=float(teacher_delta.norm()*pred_delta.norm())
     r={'head_sha256':hashes[arm],'metrics':metrics,'teacher_error':{'mse':float(error.square().mean()),'l2':float(error.norm()),'maximum_abs':float(error.abs().max()),
       'delta_cosine':float((pred_delta@teacher_delta)/den) if den>0 else None},'no_correction_applied':True}
     if arm in ('c9_baseline','functional720'):
      oldname='frozen_c9_raw' if arm=='c9_baseline' else 'c15_raw';oldtheta=vector(load_obj(parent,evalbase+oldname+'.pt'))
      oldmetrics=json.loads(parent.read(evalbase+oldname+'.json'));err=float((proposal-oldtheta).abs().max())
      r['replay']={'parameter_max_error':err,'metrics_match':replay_match(metrics,oldmetrics),
       'replay_pass':bool(err<=SETTINGS['parameter_replay_tolerance'] and replay_match(metrics,oldmetrics)),
       'drift_difference':metrics['drift']-oldmetrics['drift'],'KL_difference':metrics['kl']-oldmetrics['kl']}
     dump(resultfile,r);frozen_check()
    arm_results[arm]=r;print('C16 ACTUAL',cohort,sid,arm,'pass',r['metrics'].get('episode_pass',False),'drift',r['metrics']['drift'],'teacher_error_l2',r['teacher_error']['l2'],flush=True)
   row={'id':sid,'cohort':cohort,'case':case,'reference_valid':state_audit['actual_reference_valid'],'state_audit':state_audit,'arms':arm_results}
   dump(complete,row);rows.append(row)
   if (index+1)%SETTINGS['archive_every_states']==0:snapshot(out,args.drive_dir)
  if len(rows)!=52:raise RuntimeError('All 52 saved states required')
  training=[r for r in rows if r['cohort']=='training'];new=[r for r in rows if r['cohort']=='new']
  replay_ok=all(r['arms'][a]['replay']['replay_pass'] for r in rows for a in ('c9_baseline','functional720'))
  stage_premises_ok=replay_ok and all(r['reference_valid'] and r['state_audit']['saved_route_matches_saved_features'] for r in training)
  summary={'protocol':PROTOCOL,'complete':True,'decision':'C16_PAIRED_CHECKPOINT_DIAGNOSIS_COMPLETE',
   'stage_interpretation':interpretation(training,stage_premises_ok),'stage_attribution_premises_met':stage_premises_ok,
   'training':{a:cohort_counts(training,a) for a in ARMS},
   'inspected_new_states':{a:cohort_counts(new,a) for a in ARMS},'paired_training':paired(training),'paired_inspected_new_states':paired(new),
   'archive_replays':{'n':104,'passed':sum(r['arms'][a]['replay']['replay_pass'] for r in rows for a in ('c9_baseline','functional720')),'consistent':replay_ok},
   'checkpoint_head_sha256':hashes,'saved_input_audits':{'n':52,'fresh_route_matches':sum(r['state_audit']['route_matches_fresh_features'] for r in rows),
     'maximum_fresh_feature_error':max(r['state_audit']['fresh_feature_max_error'] for r in rows),
     'maximum_fresh_input_error':max(r['state_audit']['fresh_input_max_error'] for r in rows),
     'maximum_training_vs_online_input_error':max(r['state_audit']['training_vs_online_input_max_error'] for r in rows)},
   'new_training_steps':0,'new_states':0,'Jacobian_builds':0,'SVD_calls':0,'correction_steps':0,
   'fresh_evaluation':False,'production_head_selected':False,'parent_C15_decision_unchanged':info['summary']['decision'],
   'internal_parameter_update':False,'controller_dispatch_tested':False,'scope':SETTINGS['scope']}
  frozen_check();w.assign_flat(params,torch.zeros(36864,device=device));frozen_check()
  dump(out/'SUMMARY.json',summary);snapshot(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
