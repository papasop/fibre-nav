"""C23 fixed-origin development diagnosis. No training or persistent head update."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,time,zipfile
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from raw_writer import StateWriter,route_from_features,prepared_input,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from radius_calibration import reconstruct_direction,scale_direction
from joint_calibration import response_prediction
from precision_diagnostics import controls_summary
from c23_protocol import (PROTOCOL,RESULT,SETTINGS,REVISION,CASES,GATES,START_MEMBER,START_CP_SHA,START_HEAD_SHA,
 training_specs,original_training_specs,candidate_specs,source_hashes,inherited_check,sha,validate_capsule,
 endpoint_gate,episode_gate,target_from_reads,snapshot,quality,guard,merge_route,check_replay,candidate_key,decide)
ROOT=Path(__file__).resolve().parent

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temp,path)
def save_tensor(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)
def load_obj(z,n):return torch.load(io.BytesIO(z.read(n)),map_location='cpu',weights_only=True)
def vector(v):
 if isinstance(v,dict):v=v['theta']
 if not isinstance(v,torch.Tensor) or v.shape!=(36864,) or v.dtype!=torch.float32 or not torch.isfinite(v).all():raise ValueError('Finite original float32 memory chart required')
 return v.detach().cpu().clone()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def head_hash(h):return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in h.parameters())).hexdigest()
def cpu_tree(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
 if isinstance(v,list):return [cpu_tree(x) for x in v]
 if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
 return v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 inherited_check(ROOT);parent_info=validate_capsule(out/'parent_capsule.zip');device,gpu=a100_device(torch)
 ident={'protocol':PROTOCOL,'settings':SETTINGS,'parent_capsule_sha256':sha(out/'parent_capsule.zip'),'parent_c22_sha256':parent_info['capsule_identity']['parent_c22_sha256'],
  'origin_head_sha256':START_HEAD_SHA,'source_hashes':source_hashes(ROOT),'gpu':gpu,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'model_revision':REVISION}
 if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('Runtime/source changed; refuse incompatible diagnostic resume')
 dump(out/'identity.json',ident)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_*'))
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,args.drive_dir);return
 snapshot(out,args.drive_dir)
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION);w.set_seed(82931)
 print('C23 Loading pinned GPT-2 once; frozen C22 origin, cached directions, no training or new Jacobians.',flush=True)
 tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
 for p in model.parameters():p.requires_grad_(False)
 base_versions={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
 pins=json.loads((ROOT/'MODEL_PINS.json').read_text());ah=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
 if ah.hexdigest()!=pins['lora_A_sha256'] or getattr(model.config,'_commit_hash',None)!=REVISION:raise RuntimeError('Original model/A identity mismatch')
 if [list(p.shape) for p in params]!=pins['chart_shapes'] or w.param_flat(params).abs().max()!=0:raise RuntimeError('Original chart/zero-B changed')
 symbols=w.single_token_pair(tok);ids=[s[1] for s in symbols]
 if json.loads(json.dumps(symbols))!=pins['symbols']:raise RuntimeError('Read token identity changed')
 rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count']);response_ids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,response_ids].mean(0)
 _,exact,_,dev=audit.prompts();exact_batch=tok(exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
 anchor_batch=tok(anchors,return_tensors='pt',padding=True,truncation=True,max_length=48).to(device)
 starts=pins['starts'];targets={c:target_from_reads(starts[c]) for c in CASES}
 def sync():torch.cuda.synchronize(device)
 def frozen_model_check():
  if any(p._version!=base_versions[n] for n,p in model.named_parameters() if n in base_versions):raise RuntimeError('Frozen base/A changed')
  if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('Unexpected model gradients outside offline preparation')
 @torch.no_grad()
 def response_probe(theta):
  if not torch.isfinite(theta).all():raise RuntimeError('Nonfinite diagnostic state')
  sync();ts=time.perf_counter();w.assign_flat(params,theta.to(device))
  if not torch.equal(w.param_flat(params).cpu(),theta.cpu()):raise RuntimeError('Physical parameter assignment differs from requested state')
  al=w.next_logits(model,tok,anchors,device)
  if not torch.isfinite(al).all():raise RuntimeError('Nonfinite diagnostic forward logits')
  selected=al[:,response_ids]
  residual=selected.mean(0)-response0
  residual64=selected.double().mean(0)-logits0[:,response_ids].double().mean(0)
  pk=(logits0.softmax(-1)*(logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
  kl=max(0.,float(pk.mean()));drift=float(residual.abs().max());drift64=float(residual64.abs().max())
  ulp=(torch.nextafter(selected,torch.full_like(selected,float('inf')))-selected).abs()
  sync();result={'theta_sha256':tensor_hash(theta),'drift':drift,'kl':kl,'response_residual':residual.cpu().double().tolist(),
   'response_residual_float64_aggregation':residual64.cpu().tolist(),'float64_aggregation_drift':drift64,
   'primary_response_gate':drift<=.002,'float64_aggregation_gate_diagnostic':drift64<=.002,
   'maximum_aggregation_difference':float((residual.double()-residual64).abs().max()),
   'full_logits_sha256':tensor_hash(al),'selected_logits_sha256':tensor_hash(selected),'selected_logits':selected.cpu().tolist(),
   'selected_logits_minimum_ulp':float(ulp.min()),'selected_logits_maximum_ulp':float(ulp.max()),
   'same_float32_logits_for_both_aggregations':True,'float64_changes_primary_gate':False,'elapsed_seconds':time.perf_counter()-ts}
  frozen_model_check();return result
 @torch.no_grad()
 def evaluate(theta,target,start):
  w.assign_flat(params,theta.to(device));bits=audit.content_bits(w,model,tok,exact,ids,device)
  ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device);probe=response_probe(theta)
  v={'bits':bits,'evaluation':ev,'drift':probe['drift'],'kl':probe['kl'],'pass':False,**probe}
  v['pass']=endpoint_gate(v,target);v['gates']={'bits_match':bits==target,'response':v['drift']<=.002,'KL':v['kl']<=.005,
   'exact_read_margin':min(ev['exact_margins'])>=.05,'development_read_margin':min(ev['heldout_margins'])>=.05,
   'output_flip':bits[3]!=start[3],'unselected_preserved':bits[:3]==start[:3]}
  v['episode_pass']=episode_gate(start,v);return v
 with ExitStack() as stack:
  parent=stack.enter_context(zipfile.ZipFile(out/'parent_capsule.zip'))
  inherited_parent=stack.enter_context(zipfile.ZipFile(io.BytesIO(parent.read('training_capsule.zip'))))
  capsule=stack.enter_context(zipfile.ZipFile(io.BytesIO(inherited_parent.read('training_capsule.zip'))))
  basecp=load_obj(capsule,'c9_checkpoint.pt');startcp=load_obj(parent,'start_checkpoint.pt')
  if hashlib.sha256(parent.read('start_checkpoint.pt')).hexdigest()!=START_CP_SHA or startcp['rounds']!=15 or startcp['accepted_updates']!=13 or not startcp['done']:raise RuntimeError('True C22 final checkpoint required')
  train_ids=[t['id'] for t in training_specs()]
  if startcp['train_ids']!=train_ids:raise RuntimeError('C22 actual60 training IDs changed')
  c9norm={k:v.to(device) for k,v in basecp['normalisation'].items()};extra_norm={k:v.to(device) for k,v in startcp['extra_norm'].items()}
  if set(extra_norm)!={'mean','scale'} or extra_norm['mean'].shape!=(70,) or any(not torch.isfinite(v).all() for v in extra_norm.values()) or not (extra_norm['scale']>0).all():raise RuntimeError('Fixed writer normalisation invalid')
  sketchcp=load_obj(capsule,'STATE_SKETCH.pt');sketch=sketchcp['matrix'].to(device)
  if sketch.shape!=(36864,64) or not torch.isfinite(sketch).all() or sketchcp['seed']!=615031:raise RuntimeError('Fixed state sketch changed')
  baseline=StateWriter().to(device);baseline.load_state_dict(startcp['head']);baseline.eval()
  head=StateWriter().to(device);head.load_state_dict(startcp['head']);head.eval()
  for writer in (baseline,head):
   for p in writer.parameters():p.requires_grad_(False);p.grad=None
  if head_hash(head)!=START_HEAD_SHA or head_hash(baseline)!=START_HEAD_SHA:raise RuntimeError('Actual frozen C22 writer differs')
  case=SETTINGS['scan_case'];route=CASES.index(case);old_layer=layer_matrix(head,route)
  protected_versions={n:p._version for n,p in head.named_parameters() if not n.startswith('experts.%d.2.'%route)}
  baseline_versions=[p._version for p in baseline.parameters()]
  def invariant_check(origin=False):
   if any(p._version!=protected_versions[n] for n,p in head.named_parameters() if n in protected_versions):raise RuntimeError('Unselected writer tensors changed')
   if [p._version for p in baseline.parameters()]!=baseline_versions or head_hash(baseline)!=START_HEAD_SHA:raise RuntimeError('Frozen baseline changed')
   if any(p.requires_grad or p.grad is not None for writer in (baseline,head) for p in writer.parameters()):raise RuntimeError('Unexpected writer gradients')
   if origin and head_hash(head)!=START_HEAD_SHA:raise RuntimeError('Frozen origin was not restored')
   frozen_model_check()
  def restore_origin():set_layer_matrix(head,route,old_layer);invariant_check(True)
  @torch.no_grad()
  def fresh_input(theta,desired):
   w.assign_flat(params,theta.to(device));r=model(**exact_batch,use_cache=False,output_hidden_states=True)
   last=exact_batch['attention_mask'].sum(1)-1;rr=torch.arange(4,device=device)
   f=torch.cat([r.hidden_states[-1][rr,last].reshape(-1),r.logits[rr,last][:,ids][:,1]-r.logits[rr,last][:,ids][:,0],torch.tensor([desired],device=device,dtype=torch.float32)])
   residual=(w.response_vector(model,tok,anchors,response_ids,device)-response0).cpu().double()
   extra=torch.cat([residual.float().to(device)/.002,theta.to(device)@sketch]);return prepared_input(f,extra,c9norm,extra_norm).cpu()
  data=[]
  for t in training_specs():
   b='training/'+t['id']+'/';prep=vector(load_obj(capsule,b+'prepared.pt'));f=load_obj(capsule,b+'INPUT.pt');ref=json.loads(capsule.read(b+'REFERENCE.json'))
   if not ref['valid'] or f['input'].shape!=(3147,) or not torch.isfinite(f['input']).all() or int(f['route'])!=CASES.index(t['case']) or int(route_from_features(f['features']))!=int(f['route']):raise RuntimeError('Inherited reference/input invalid')
   data.append({'id':t['id'],'case':t['case'],'prep':prep,'input':f['input'].to(device),'route':f['route'].to(device),'reference_valid':True})
  @torch.no_grad()
  def measure_training(writer,items=None):
   result={};items=data if items is None else items
   for t in items:
    delta=one_raw_proposal(writer,t['input'],t['route'],c9norm['output_scale']).cpu();m=evaluate(t['prep']+delta,targets[t['case']],starts[t['case']]);result[t['id']]={'reference_valid':True,'metrics':m}
   invariant_check();return result
  if (out/'BASELINE_REPLAY.json').exists():current=json.loads((out/'BASELINE_REPLAY.json').read_text())
  else:
   current=measure_training(baseline);check_replay(startcp['current_training'],current);dump(out/'BASELINE_REPLAY.json',current)
  original_ids={t['id'] for t in original_training_specs()};q=quality(current)
  if len(q['passed_ids'])!=54 or len(set(q['passed_ids'])&original_ids)!=32:raise RuntimeError('Actual C22 54/60 origin did not replay')
  print('C23 FROZEN ORIGIN',json.dumps(q),flush=True)
  group=[t for t in data if t['case']==case];group_ids=[t['id'] for t in group];states=[]
  for t in group:
   r=load_obj(parent,'geometry/'+t['id']+'/JACOBIAN.pt');delta=one_raw_proposal(baseline,t['input'],t['route'],c9norm['output_scale']).cpu()
   if r['id']!=t['id'] or r['head_sha256']!=START_HEAD_SHA or r['input_sha256']!=tensor_hash(t['input']) or not torch.equal(vector(r['theta']),t['prep']+delta):raise RuntimeError('Cached Jacobian at different head/input/parameter state')
   if r['J'].shape!=(6,36864) or r['H'].shape!=(65,) or r['residual'].shape!=(6,) or any(not torch.isfinite(r[k]).all() for k in ('J','H','residual','theta')):raise RuntimeError('Cached geometry shape/finite check failed')
   if not torch.equal(r['H'],hidden_augmented(baseline,t['input'],route).cpu()):raise RuntimeError('Cached hidden features changed')
   if (fresh_input(t['prep'],targets[case][3])-t['input'].cpu()).abs().max()>1e-4:raise RuntimeError('Actual inherited access features changed')
   if np.max(np.abs(r['residual'].numpy()-np.asarray(current[t['id']]['metrics']['response_residual'])))>1e-7:raise RuntimeError('Cached response residual does not replay')
   states.append(r)
  J=np.stack([r['J'].double().numpy() for r in states]);H=np.stack([r['H'].double().numpy() for r in states]);E=np.stack([r['residual'].numpy() for r in states]);scale=float(c9norm['output_scale'])
  j=J.reshape(90,36864);h=np.repeat(H,6,axis=0);den=np.maximum(scale*np.linalg.norm(j,axis=1)*np.linalg.norm(h,axis=1),1e-12)
  info=json.loads(parent.read('direction_info/DIRECTION_INFO_00_00.json'))['info'];system={'J':J,'H':H,'E':E,'scale':scale,'j':j,'h':h,'den':den,'info':info};bases=[];solver_records=[]
  for ri,ridge in enumerate(SETTINGS['ridge_grid']):
   c=load_obj(parent,'coefficients/COEFFICIENTS_%02d.pt'%ri)
   if c['start_head_sha256']!=START_HEAD_SHA or c['case']!=case or c['ridge']!=ridge or c['training_ids']!=group_ids or c['coefficients'].shape!=(90,) or c['coefficients'].dtype!=torch.float64 or not torch.isfinite(c['coefficients']).all():raise RuntimeError('Cached interval coefficients identity changed')
   D=reconstruct_direction(system,c['coefficients'].numpy());expected=json.loads(parent.read('direction_info/DIRECTION_INFO_00_%02d.json'%ri))
   if not np.isclose(np.linalg.norm(D),expected['info']['unclipped_layer_norm'],rtol=1e-10,atol=1e-12):raise RuntimeError('Cached direction reconstruction mismatch')
   bases.append(D);solver_records.append(c['solver_info'])
  dump(out/'CACHED_GEOMETRY_VERIFIED.json',{'head_sha256':START_HEAD_SHA,'cached_states':15,'source_round':14,'new_Jacobians':0,'new_solver_calls':0,'solver_records':solver_records})
  def zero_phase(phase):
   restore_origin()
   for repeat in range(SETTINGS['zero_repeats_per_phase']):
    for t,r in zip(group,states):
     f=out/'zero_controls'/phase/('repeat_%02d'%repeat)/(t['id']+'.json')
     if f.exists():record=json.loads(f.read_text())
     else:record={'phase':phase,'repeat':repeat,'state_id':t['id'],'origin_head_sha256':START_HEAD_SHA,'probe':response_probe(vector(r['theta']))};dump(f,record)
     if record['origin_head_sha256']!=START_HEAD_SHA or record['probe']['theta_sha256']!=tensor_hash(r['theta']):raise RuntimeError('Cached zero probe identity differs')
    snapshot(out,args.drive_dir);print('C23 ZERO CONTROL',phase,repeat+1,'/3',flush=True)
  zero_phase('before_scan');records=[];eligible=[];seen={}
  for index,spec in enumerate(candidate_specs()):
   folder=out/'candidates'/spec['id'];folder.mkdir(parents=True,exist_ok=True);complete=folder/'COMPLETE.json'
   D,di=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_frobenius_radius'],spec['increment_radius'])
   trial_layer=(old_layer.double()+spec['alpha']*torch.from_numpy(D)).float()
   try:
    set_layer_matrix(head,route,trial_layer);invariant_check();trial_hash=head_hash(head)
    if complete.exists():
     record=json.loads(complete.read_text())
     if record['spec']!=spec or record['origin_head_sha256']!=START_HEAD_SHA or record['candidate_head_sha256']!=trial_hash:raise RuntimeError('Cached candidate identity changed')
    else:
     displacement=[]
     for t,r in zip(group,states):
      delta=one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale']).cpu();displacement.append(((t['prep']+delta).double()-r['theta'].double()).numpy())
     displacement=np.stack(displacement);pred=response_prediction(J,E,displacement)
     alias=seen.get(trial_hash)
     if alias is not None:rows=alias['route_rows'];reuse=alias['spec']['id']
     else:
      rows={};reuse=None
      for t in group:
       rf=folder/'states'/(t['id']+'.json');delta=one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale']).cpu();theta=t['prep']+delta
       if rf.exists():row=json.loads(rf.read_text())
       else:row={'reference_valid':True,'metrics':evaluate(theta,targets[case],starts[case])};dump(rf,row)
       if row['metrics']['theta_sha256']!=tensor_hash(theta):raise RuntimeError('Cached candidate state differs')
       rows[t['id']]=row
     merged=merge_route(current,rows,group_ids);g=guard(current,merged);actual=np.array([rows[t['id']]['metrics']['response_residual'] for t in group])
     record={'spec':spec,'origin_head_sha256':START_HEAD_SHA,'candidate_head_sha256':trial_hash,'route_rows':rows,'route_guard':g,
      'route_guard_eligible':g['accepted'],'full60_verified':False,'byte_identical_alias_of':reuse,
      'prediction':{'maximum_drift':float(np.max(np.abs(pred))),'maximum_prediction_error':float(np.max(np.abs(actual-pred))),
       'max_memory_step_norm':float(np.max(np.linalg.norm(displacement,axis=1))),'layer_step_norm':float((trial_layer.double()-old_layer.double()).norm()),
       'max_float32_displacement_difference':float(np.max(np.abs(displacement-scale*((trial_layer.double()-old_layer.double()).numpy()@H.T).T)))},
      'float64_aggregation_response_passes_diagnostic':sum(r['metrics']['float64_aggregation_gate_diagnostic'] for r in rows.values()),
      'aggregation_gate_disagreements':sum(r['metrics']['primary_response_gate']!=r['metrics']['float64_aggregation_gate_diagnostic'] for r in rows.values()),
      'maximum_aggregation_difference':max(r['metrics']['maximum_aggregation_difference'] for r in rows.values()),'direction_info':di}
     dump(complete,record)
    if set(record['route_rows'])!=set(group_ids):raise RuntimeError('Cached candidate route denominator changed')
    for t in group:
     delta=one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale']).cpu()
     if record['route_rows'][t['id']]['metrics']['theta_sha256']!=tensor_hash(t['prep']+delta):raise RuntimeError('Cached completed state differs from actual candidate')
    replay_guard=guard(current,merge_route(current,record['route_rows'],group_ids))
    if replay_guard['accepted']!=record['route_guard_eligible'] or replay_guard['reasons']!=record['route_guard']['reasons']:raise RuntimeError('Cached candidate guard changed')
    seen.setdefault(trial_hash,record);records.append(record)
    if record['route_guard_eligible']:
     merged=merge_route(current,record['route_rows'],group_ids);eligible.append({**spec,'rows':merged,'record':record})
    print('C23 SCAN',index+1,'/90',spec['id'],'alpha',spec['alpha'],'eligible',record['route_guard_eligible'],'R',record['route_guard']['after']['maximum_drift'],flush=True)
   finally:restore_origin()
   if (index+1)%SETTINGS['archive_every_candidates']==0:snapshot(out,args.drive_dir)
  best=None;audits=[];audited_heads=set()
  for candidate in sorted(eligible,key=candidate_key):
   if candidate['record']['candidate_head_sha256'] in audited_heads:continue
   audited_heads.add(candidate['record']['candidate_head_sha256'])
   spec=candidate['record']['spec'];folder=out/'candidate_audits'/spec['id'];folder.mkdir(parents=True,exist_ok=True);f=folder/'AUDIT.json'
   D,_=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_frobenius_radius'],spec['increment_radius']);trial_layer=(old_layer.double()+spec['alpha']*torch.from_numpy(D)).float()
   try:
    set_layer_matrix(head,route,trial_layer);invariant_check()
    if head_hash(head)!=candidate['record']['candidate_head_sha256']:raise RuntimeError('Ranked candidate identity changed')
    if f.exists():a=json.loads(f.read_text())
    else:
     measured=measure_training(head);reason=None
     try:check_replay(candidate['rows'],measured)
     except ValueError as err:reason=str(err)
     g=guard(current,measured);a={'spec':spec,'head_sha256':head_hash(head),'full60_rows':measured,'guard':g,'replay_error':reason,'full60_verified':bool(reason is None and g['accepted'])};dump(f,a)
    if a['spec']!=spec or a['head_sha256']!=head_hash(head):raise RuntimeError('Cached full60 audit identity changed')
    audits.append(a)
    if a['full60_verified']:
     best={'spec':spec,'candidate_head_sha256':a['head_sha256'],'quality':quality(a['full60_rows']),'full60_verified':True,'head_committed':False}
     save_tensor(out/'best_candidate'/'CANDIDATE_LAYER.pt',{'layer':trial_layer,'spec':spec,'origin_head_sha256':START_HEAD_SHA,'candidate_head_sha256':a['head_sha256'],'diagnostic_only_not_committed':True})
     dump(out/'BEST_CANDIDATE.json',best);break
   finally:restore_origin()
  zero_phase('after_scan');zeros=[]
  for f in sorted((out/'zero_controls').rglob('*.json')):zeros.append(json.loads(f.read_text()))
  zero_info=controls_summary(zeros,group_ids);dump(out/'ZERO_CONTROL_SUMMARY.json',zero_info)
  restore_origin();invariant_check(True);w.assign_flat(params,torch.zeros(36864,device=device))
  summary={'protocol':PROTOCOL,'complete':True,'decision':decide(records,best,True),'origin_head_sha256':START_HEAD_SHA,'origin_head_unchanged':True,
   'origin_training_quality':quality(current),'n_requested_candidate_specs':90,'n_completed_candidate_specs':len(records),'n_unique_candidate_heads':len(seen),
   'n_route_guard_eligible_specs':len(eligible),'n_full60_audits':len(audits),'best_candidate':best,'no_head_committed':True,
   'primary_float32_acceptance_unchanged':True,'float64_aggregation_is_diagnostic_only':True,'full_model_float64_forward_tested':False,
   'zero_controls':zero_info,'cached_geometry_states':15,'new_Jacobian_builds':0,'new_backward_calls':0,'new_linear_solver_calls':0,'new_training_steps':0,
   'new_evaluation_states':0,'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
   'parent_C22_decision_unchanged':parent_info['summary']['decision'],'scope':SETTINGS['scope']}
  dump(out/'SUMMARY.json',summary);snapshot(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
