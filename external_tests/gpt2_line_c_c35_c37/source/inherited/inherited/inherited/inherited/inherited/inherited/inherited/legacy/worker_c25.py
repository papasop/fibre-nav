"""C25 guarded OFFLINE fitting; online branch is only features -> one head -> commit."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,time,zipfile
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from raw_writer import StateWriter,route_from_features,prepared_input,one_raw_proposal
from functional_writer import chart_spec,last_logits
from guarded_update import cpu_state
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from joint_calibration import prepare_system,response_prediction,linear_protection
from budget_calibration import interval_coefficients,reconstruct_direction,scale_direction
from c25_guard import merge_route,check_replay,choose_case
from c17_protocol import quality as original_quality
from c25_protocol import (PROTOCOL,RESULT,SETTINGS,REVISION,CASES,STATES,GATES,
 START_HEAD_SHA,original_training_specs,original60_specs,promoted_specs,training_specs,test_specs,validate_capsule,source_hashes,inherited_check,
 sha,endpoint_gate,episode_gate,target_from_reads,snapshot,quality,guard,counts,decide)
from c25_identity import validate_start
from c25_scan import scan_round
from correction_engine import run_correction
from geometry import null_project
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
 ident={'protocol':PROTOCOL,'settings':SETTINGS,'parent_capsule_sha256':sha(out/'parent_capsule.zip'),'parent_c24_sha256':parent_info['capsule_identity']['parent_c24_sha256'],'start_head_sha256':START_HEAD_SHA,'source_hashes':source_hashes(ROOT),
  'gpu':gpu,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'model_revision':REVISION}
 if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('Runtime/source changed; refuse incompatible resume')
 dump(out/'identity.json',ident)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_*'))
 split={'training':training_specs(),'original60':original60_specs(),'promoted_inspected16':promoted_specs(),'prospective_new_states':test_specs(),'old_inspected16_used_for_training':True,'new_generation_after_final_writer_freeze':True,'test_based_checkpoint_selection':False,'fresh_evaluation':False}
 if (out/'SPLIT.json').exists() and json.loads((out/'SPLIT.json').read_text())!=split:raise RuntimeError('Fixed split changed')
 dump(out/'SPLIT.json',split)
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,args.drive_dir);return
 snapshot(out,args.drive_dir)
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION);w.set_seed(82931)
 print('C25 Loading pinned GPT-2 once; actual C24 final head, original60 + promoted16, four-route offline calibration; online no repair.',flush=True)
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
 spec=chart_spec(model,params);starts=pins['starts'];targets={c:target_from_reads(starts[c]) for c in CASES}
 def sync():torch.cuda.synchronize(device)
 def frozen_model_check():
  if any(p._version!=base_versions[n] for n,p in model.named_parameters() if n in base_versions):raise RuntimeError('Frozen base/A changed')
  if any(p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError('Unexpected model gradients outside offline preparation')
 @torch.no_grad()
 def evaluate(theta,target,start=None):
  if not torch.isfinite(theta).all():raise RuntimeError('Nonfinite proposal; no fallback allowed')
  sync();t0=time.perf_counter();w.assign_flat(params,theta.to(device));sync();commit_seconds=time.perf_counter()-t0
  bits=audit.content_bits(w,model,tok,exact,ids,device);ev=audit.evaluate(w,model,tok,exact,dev,[ids[b] for b in target],ids,device)
  drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
  v={'theta_sha256':tensor_hash(theta),'bits':bits,'evaluation':ev,'drift':float(drift),'kl':float(kl),'pass':False}
  v['pass']=endpoint_gate(v,target);v['gates']={'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005,
   'exact_read_margin':min(ev['exact_margins'])>=.05,'development_read_margin':min(ev['heldout_margins'])>=.05}
  if start is not None:v['gates'].update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3]);v['episode_pass']=episode_gate(start,v)
  al=w.next_logits(model,tok,anchors,device)
  residual=al[:,response_ids].mean(0)-response0
  v['response_residual']=residual.cpu().double().tolist()
  v['response_residual_float64_aggregation']=(al[:,response_ids].double().mean(0)-logits0[:,response_ids].double().mean(0)).cpu().tolist()
  if abs(float(residual.abs().max())-v['drift'])>1e-7:raise RuntimeError('Physical response measurement paths disagree')
  pr=(al[:,response_ids]-logits0[:,response_ids]).abs().max(1).values
  pk=(logits0.softmax(-1)*(logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
  v['per_anchor_diagnostics']={'maximum_selected_logit_change':float(pr.max()),'maximum_KL':float(pk.max()),
   'selected_logit_changes':pr.cpu().tolist(),'KL_values':pk.cpu().tolist(),'not_additional_gates':True}
  sync();v['timing']={'commit_seconds':commit_seconds,'verification_seconds':time.perf_counter()-t0-commit_seconds};frozen_model_check();return v

 with ExitStack() as stack:
  parent=stack.enter_context(zipfile.ZipFile(out/'parent_capsule.zip'))
  inherited=stack.enter_context(zipfile.ZipFile(io.BytesIO(parent.read('c24_capsule.zip'))))
  c23capsule=stack.enter_context(zipfile.ZipFile(io.BytesIO(inherited.read('c23_capsule.zip'))))
  training_parent=stack.enter_context(zipfile.ZipFile(io.BytesIO(c23capsule.read('training_capsule.zip'))))
  capsule=stack.enter_context(zipfile.ZipFile(io.BytesIO(training_parent.read('training_capsule.zip'))))
  basecp=load_obj(capsule,'c9_checkpoint.pt');startcp=load_obj(parent,'start_checkpoint.pt');initial_state=validate_start(startcp)
  original_ids=[t['id'] for t in original60_specs()];train_ids=[t['id'] for t in training_specs()]
  c9norm={k:v.to(device) for k,v in basecp['normalisation'].items()};extra_norm={k:v.to(device) for k,v in startcp['extra_norm'].items()}
  if set(extra_norm)!={'mean','scale'} or extra_norm['mean'].shape!=(70,) or extra_norm['scale'].shape!=(70,) or any(not torch.isfinite(v).all() for v in extra_norm.values()) or not (extra_norm['scale']>0).all():raise RuntimeError('Saved extra normalisation invalid')
  sketchcp=load_obj(capsule,'STATE_SKETCH.pt');sketch=sketchcp['matrix'].to(device)
  if sketch.shape!=(36864,64) or not torch.isfinite(sketch).all() or sketchcp['seed']!=615031:raise RuntimeError('Saved state sketch invalid')
  baseline=StateWriter().to(device);baseline.load_state_dict(initial_state);baseline.eval()
  for p in baseline.parameters():p.requires_grad_(False)
  if head_hash(baseline)!=START_HEAD_SHA:raise RuntimeError('Actual C24 head mismatch')
  head=StateWriter().to(device);head.load_state_dict(initial_state);head.eval()
  for p in head.parameters():p.requires_grad_(False);p.grad=None
  hidden_versions={n:p._version for n,p in head.named_parameters() if n.split('.')[2]=='0'}
  baseline_versions=[p._version for p in baseline.parameters()]
  def baseline_check():
   if any(p._version!=hidden_versions[n] for n,p in head.named_parameters() if n in hidden_versions):raise RuntimeError('Frozen writer input layer changed')
   if [p._version for p in baseline.parameters()]!=baseline_versions or any(p.requires_grad or p.grad is not None for p in baseline.parameters()):raise RuntimeError('Frozen C24 control changed')
  @torch.no_grad()
  def fresh_inputs(theta,desired):
   sync();feature_start=time.perf_counter()
   w.assign_flat(params,theta.to(device));r=model(**exact_batch,use_cache=False,output_hidden_states=True)
   last=exact_batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=device)
   f=torch.cat([r.hidden_states[-1][rows,last].reshape(-1),r.logits[rows,last][:,ids][:,1]-r.logits[rows,last][:,ids][:,0],torch.tensor([desired],device=device,dtype=torch.float32)])
   response=w.response_vector(model,tok,anchors,response_ids,device)
   residual=(response-response0).cpu().double().numpy()
   extra=torch.cat([torch.from_numpy(residual).float().to(device)/.002,theta.to(device)@sketch])
   sync();elapsed=time.perf_counter()-feature_start
   return {'features':f.cpu(),'extras':extra.cpu(),'input':prepared_input(f,extra,c9norm,extra_norm).cpu(),'route':route_from_features(f).cpu(),'feature_seconds':elapsed}

  data=[];base_data={}
  for t in training_specs():
   sid=t['id'];case=t['case'];promoted=sid not in original_ids;source=inherited if promoted else capsule;b=('evaluation_states/' if promoted else 'training/')+sid+'/'
   prep=vector(load_obj(source,b+'prepared.pt'));teacher=vector(load_obj(source,b+'teacher.pt'));f=load_obj(source,b+'INPUT.pt');ref=json.loads(source.read(b+'REFERENCE.json'))
   if f['input'].shape!=(3147,) or not torch.isfinite(f['input']).all() or int(route_from_features(f['features']))!=int(f['route']):raise RuntimeError('Saved training input invalid')
   if int(f['route'])!=CASES.index(case):raise RuntimeError('Actual operand route differs from declared training case')
   folder=out/'training_replay'/sid;folder.mkdir(parents=True,exist_ok=True)
   if (folder/'REFERENCE.json').exists():replay=json.loads((folder/'REFERENCE.json').read_text())
   else:
    pm=evaluate(prep,starts[case]);tm=evaluate(teacher,targets[case],starts[case]);fresh=fresh_inputs(prep,targets[case][3])
    err=float((fresh['input']-f['input']).abs().max())
    replay={'preparation':pm,'teacher':tm,'reference_valid':bool(ref['valid'] and pm['pass'] and tm['episode_pass']),
     'fresh_input_max_error':err,'input_match':bool(err<=1e-4 and int(fresh['route'])==int(f['route']))};dump(folder/'REFERENCE.json',replay)
   if not replay['reference_valid'] or not replay['input_match']:raise RuntimeError('Training reference/input replay failed: '+sid)
   data.append({'id':sid,'case':case,'prep':prep,'teacher':teacher,'input':f['input'].to(device),'route':f['route'].to(device),'reference_valid':True})
   if sid in STATES:base_data[sid]=(prep,teacher)
  counters=json.loads((out/'OFFLINE_PREPARATION_COUNTERS.json').read_text()) if (out/'OFFLINE_PREPARATION_COUNTERS.json').exists() else {'jacobian_builds':0,'response_backward_calls':0}
  @torch.no_grad()
  def measured_response(theta):
   w.assign_flat(params,theta.to(device));response=w.response_vector(model,tok,anchors,response_ids,device);drift,kl=w.anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
   return (response-response0).cpu().double().numpy(),float(drift),float(kl)
  def offline_jacobian(theta):
   w.assign_flat(params,theta.to(device))
   for p in params:p.requires_grad_(True)
   try:
    with torch.enable_grad():
     response=w.response_vector(model,tok,anchors,response_ids,device);rr=[]
     for i in range(response.numel()):
      gg=torch.autograd.grad(response[i],params,retain_graph=i+1<response.numel());rr.append(torch.cat([g.reshape(-1) for g in gg]).detach().cpu().double().numpy())
     j=np.stack(rr)
    counters['jacobian_builds']+=1;counters['response_backward_calls']+=len(rr)
   finally:
    for p in params:p.requires_grad_(False);p.grad=None
   dump(out/'OFFLINE_PREPARATION_COUNTERS.json',counters);return j
  def offline_correct(folder,initial):
   folder.mkdir(parents=True,exist_ok=True);f=folder/'CORRECTION_STATE.pt';resume=torch.load(f,map_location='cpu',weights_only=True) if f.exists() else None
   if resume and resume.get('initial_sha256')!=tensor_hash(initial):raise RuntimeError('Offline preparation checkpoint mismatch')
   def persist(cp):cp['initial_sha256']=tensor_hash(initial);save_tensor(f,cp)
   return run_correction(initial,measured_response,offline_jacobian,lambda t,d:t+torch.from_numpy(d).float(),vector,persist,SETTINGS,resume,sync)
  def prepare_state(t,frozen_hash=None):
   if frozen_hash is None or not (out/'WRITER_FROZEN.json').exists() or head_hash(head)!=frozen_hash:raise RuntimeError('C25 new-state generation requires the actual final frozen writer')
   writer_frozen_check()
   folder=out/'states'/t['cohort']/t['id'];folder.mkdir(parents=True,exist_ok=True);rf=folder/'REFERENCE.json'
   if rf.exists():
    stored=json.loads(rf.read_text())
    expected_hash=frozen_hash if frozen_hash is not None else START_HEAD_SHA
    if stored['spec']!=t or stored['writer_head_at_generation']!=expected_hash:raise RuntimeError('Saved prepared state phase/head/spec mismatch')
    return folder,stored
   prep0,teacher0=base_data[t['base_id']];qf=folder/'perturbation.pt'
   if qf.exists():
    qc=torch.load(qf,map_location='cpu',weights_only=True)
    if qc['spec']!=t:raise RuntimeError('Fixed new-state seed/spec changed')
    q=vector(qc);geom=qc['geometry']
   else:
    j=load_obj(capsule,'base_geometry/'+t['base_id']+'/jacobian.pt')['J'].double().numpy();projected,geom=null_project(j,np.random.default_rng(t['seed']).normal(size=36864));qn=float(np.linalg.norm(projected))
    if not qn>0:raise RuntimeError('Zero fixed perturbation; no replacement allowed')
    q=torch.from_numpy(projected*(t['norm']/qn)).float();save_tensor(qf,{'theta':q,'spec':t,'geometry':geom})
   prep,ph,ps=offline_correct(folder/'offline_preparation',prep0+q);teacher,th,ts=offline_correct(folder/'offline_teacher',teacher0+q)
   save_tensor(folder/'prepared.pt',{'theta':prep});save_tensor(folder/'teacher.pt',{'theta':teacher})
   pm=evaluate(prep,starts[t['case']]);tm=evaluate(teacher,targets[t['case']],starts[t['case']]);f=fresh_inputs(prep,targets[t['case']][3]);save_tensor(folder/'INPUT.pt',f)
   r={'spec':t,'valid':bool(pm['pass'] and tm['episode_pass'] and int(f['route'])==CASES.index(t['case'])),'preparation':pm,'teacher':tm,'generation_retries':0,'replacement_used':False,
    'writer_head_at_generation':frozen_hash if frozen_hash is not None else START_HEAD_SHA,'writer_frozen_before_generation':frozen_hash,'generation_phase':'final_frozen_evaluation' if frozen_hash is not None else 'supplemental_training_preparation','offline_geometry':True,'projection':geom,'preparation_seconds':ps,'teacher_seconds':ts};dump(rf,r);baseline_check();frozen_model_check()
   if frozen_hash is not None and head_hash(head)!=frozen_hash:raise RuntimeError('Final writer changed while generating evaluation states')
   return folder,r

  def measure_training(writer,items=None):
   ans={}
   for j,t in enumerate(data if items is None else items):
    delta=one_raw_proposal(writer,t['input'],t['route'],c9norm['output_scale']).cpu();m=evaluate(t['prep']+delta,targets[t['case']],starts[t['case']])
    ans[t['id']]={'reference_valid':t['reference_valid'],'metrics':m}
    if (j+1)%19==0 or j+1==len(data if items is None else items):print('C25 TRAIN GATE',j+1,'/',len(data if items is None else items),flush=True)
   baseline_check();return ans
  initialfile=out/'C24_EXPANDED76_BASELINE.json'
  if initialfile.exists():initial=json.loads(initialfile.read_text())
  else:initial=measure_training(baseline);dump(initialfile,initial)
  initial_quality=quality(initial);print('C25 ACTUAL C24 EXPANDED76',json.dumps(initial_quality),flush=True)
  original_initial={sid:initial[sid] for sid in original_ids}
  if len(initial_quality['passed_ids'])!=69 or not set(original_ids).issubset(initial_quality['passed_ids']):raise RuntimeError('Actual C24 original60 plus nine promoted passes did not replay')
  check_replay(json.loads(parent.read('C24_RECORDS/FINAL_TRAINING.json')),original_initial)
  for t in promoted_specs():
   r=json.loads(parent.read('C24_RECORDS/evaluation/inspected/'+t['id']+'/COMPLETE.json'))
   check_replay({t['id']:{'reference_valid':r['reference_valid'],'metrics':r['arms']['calibrated']}},{t['id']:initial[t['id']]})
  if any(not all(row['metrics']['gates'][g] for g in GATES if g!='response') for row in initial.values()):raise RuntimeError('C24 baseline lost a non-response gate')
  folder=out/'heads'/'calibrated';folder.mkdir(parents=True,exist_ok=True);progress=folder/'PROGRESS.json'
  rounds=accepted=0;history=[];current=initial;stop=None;stalled=[];last_case_index=-1
  def checkpoint(done=False):
   name='checkpoint_%06d%s.pt'%(rounds,'_final' if done else '');f=folder/name
   save_tensor(f,{'head':cpu_state(head),'rounds':rounds,'accepted_updates':accepted,'history':history,'stalled_cases':stalled,'last_case_index':last_case_index,
    'current_training':current,'train_ids':train_ids,'start_head_sha256':START_HEAD_SHA,'head_sha256':head_hash(head),
    'extra_norm':cpu_tree(extra_norm),'done':done,'stop':stop,'optimiser':None,'domain':'existing four expert output layers only'})
   dump(progress,{'checkpoint':name,'checkpoint_sha256':sha(f),'rounds':rounds,'accepted_updates':accepted,'done':done})
   for old in folder.glob('checkpoint_*.pt'):
    if old!=f:old.unlink()
  if progress.exists():
   pr=json.loads(progress.read_text());n=pr['checkpoint']
   if Path(n).name!=n or '\\' in n or sha(folder/n)!=pr['checkpoint_sha256']:raise RuntimeError('Actual C25 checkpoint corrupt/unsafe')
   cp=torch.load(folder/n,map_location='cpu',weights_only=True)
   if cp['train_ids']!=train_ids or cp['start_head_sha256']!=START_HEAD_SHA or cp['rounds']!=pr['rounds'] or any(not torch.equal(cp['extra_norm'][k].to(device),v) for k,v in extra_norm.items()):raise RuntimeError('Actual C25 identity mismatch')
   state=cp['head']
   for n,p in head.named_parameters():
    if n.split('.')[2]=='0' and not torch.equal(p.detach().cpu(),state[n]):raise RuntimeError('Checkpoint changed a frozen writer input layer')
   for route in range(4):set_layer_matrix(head,route,torch.cat([state['experts.%d.2.weight'%route],state['experts.%d.2.bias'%route].unsqueeze(1)],dim=1))
   rounds=cp['rounds'];accepted=cp['accepted_updates'];history=cp['history'];current=cp['current_training'];stop=cp['stop'];stalled=cp['stalled_cases'];last_case_index=cp['last_case_index']
   if head_hash(head)!=cp['head_sha256'] or not set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']):raise RuntimeError('Restored true calibrated head/pass set mismatch')
   print('C25 RESTORE true rounds',rounds,'accepted',accepted,flush=True)
  else:checkpoint()
  replayed=measure_training(head);check_replay(current,replayed);current=replayed

  def build_route_geometry(case,round_dir):
   group=[t for t in data if t['case']==case];route=CASES.index(case);before_hash=head_hash(head);states=[]
   for index,t in enumerate(group):
    state_dir=round_dir/'states'/t['id'];state_dir.mkdir(parents=True,exist_ok=True);jf=state_dir/'JACOBIAN.pt'
    if jf.exists():
     r=torch.load(jf,map_location='cpu',weights_only=True)
     if r['head_sha256']!=before_hash or r['input_sha256']!=tensor_hash(t['input']):raise RuntimeError('Cached calibration derivative at wrong head/input')
    else:
     sync();ts=time.perf_counter();delta=one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale']).cpu();theta=t['prep']+delta
     candidate=theta.to(device).detach().requires_grad_(True)
     with torch.enable_grad():
      logits=last_logits(model,anchor_batch,candidate,spec);residual=logits[:,response_ids].mean(0)-response0;rows=[]
      for k in range(residual.numel()):
       grad=torch.autograd.grad(residual[k],candidate,retain_graph=k+1<residual.numel())[0];rows.append(grad.detach().cpu())
      J=torch.stack(rows);functional_e=residual.detach().cpu()
     if J.shape!=(6,36864) or not torch.isfinite(J).all():raise RuntimeError('Invalid offline candidate-response Jacobian')
     physical_e=torch.tensor(current[t['id']]['metrics']['response_residual'],dtype=torch.float64)
     err=float((functional_e.double()-physical_e).abs().max())
     if err>SETTINGS['functional_response_replay_tolerance']:raise RuntimeError('Derivative/physical response paths disagree')
     H=hidden_augmented(head,t['input'],route);sync()
     r={'id':t['id'],'head_sha256':before_hash,'input_sha256':tensor_hash(t['input']),'theta':theta,'J':J,'H':H,
      'residual':physical_e,'functional_residual_error':err,'elapsed_seconds':time.perf_counter()-ts,'offline_backward_calls':6}
     save_tensor(jf,r);frozen_model_check();baseline_check()
    if r['J'].shape!=(6,36864) or r['H'].shape!=(65,) or r['residual'].shape!=(6,) or any(not torch.isfinite(r[k]).all() for k in ('J','H','residual','theta')):raise RuntimeError('Invalid saved calibration tensor')
    actual_delta=one_raw_proposal(head,t['input'],t['route'],c9norm['output_scale']).cpu()
    if not torch.equal(vector(r['theta']),t['prep']+actual_delta) or not torch.equal(r['H'],hidden_augmented(head,t['input'],route).cpu()):raise RuntimeError('Cached derivative parameter/hidden state changed')
    if np.max(np.abs(r['residual'].numpy()-np.asarray(current[t['id']]['metrics']['response_residual'])))>1e-7:raise RuntimeError('Cached derivative residual differs from current physical state')
    states.append(r);print('C25 OFFLINE GEOMETRY',case,index+1,'/19',flush=True)
    if (index+1)%SETTINGS['archive_every_geometry_states']==0:snapshot(out,args.drive_dir)
   return group,states

  if not (out/'WRITER_FROZEN.json').exists():
   while rounds<SETTINGS['max_calibration_rounds'] and stop is None:
    q=quality(current)
    if len(q['passed_ids'])==76:stop='ALL_76_TRAINING_STATES_ACCEPTED';break
    case=choose_case(current,stalled,last_case_index)
    if case is None:stop='ALL_REMAINING_ROUTE_GRIDS_STALLED';break
    route=CASES.index(case);last_case_index=route
    round_dir=out/'calibration'/('round_%06d'%(rounds+1));round_dir.mkdir(parents=True,exist_ok=True)
    before_hash=head_hash(head);old_layer=layer_matrix(head,route);start_record={'round':rounds+1,'case':case,'head_sha256':before_hash,'quality':q,'candidate_grid_reset':True}
    if (round_dir/'START.json').exists() and json.loads((round_dir/'START.json').read_text())!=start_record:raise RuntimeError('Incomplete round starts from a different actual head')
    dump(round_dir/'START.json',start_record);save_tensor(round_dir/'START_LAYER.pt',old_layer)
    print('C25 FRESH GEOMETRY ROUND',rounds+1,'protected',len(q['passed_ids']),flush=True)
    group,states=build_route_geometry(case,round_dir);group_ids=[t['id'] for t in group]
    J=np.stack([r['J'].double().numpy() for r in states]);H=np.stack([r['H'].double().numpy() for r in states]);E=np.stack([r['residual'].numpy() for r in states]);scale=float(c9norm['output_scale'])
    system=prepare_system(J,H,E,scale,SETTINGS['normalisation_floor'],SETTINGS['gram_rank_relative_cutoff']);bases=[];solver_infos=[]
    protected_versions={n:p._version for n,p in head.named_parameters() if not n.startswith('experts.%d.2.'%route)}
    def invariant_routes_check():
     baseline_check();frozen_model_check()
     if any(p._version!=protected_versions[n] for n,p in head.named_parameters() if n in protected_versions):raise RuntimeError('Unselected writer tensors changed during physical scan')
     if any(p.requires_grad or p.grad is not None for p in head.parameters()):raise RuntimeError('Unexpected writer gradients')
    for ri,ridge in enumerate(SETTINGS['ridge_grid']):
     f=round_dir/('COEFFICIENTS_%02d.pt'%ri)
     if f.exists():
      saved=torch.load(f,map_location='cpu',weights_only=True)
      if saved['start_head_sha256']!=before_hash or saved['case']!=case or saved['ridge']!=ridge or saved['training_ids']!=group_ids:raise RuntimeError('Cached coefficients identity changed')
      coefficients=saved['coefficients'].double().numpy();solver_info=saved['solver_info']
     else:
      coefficients,solver_info=interval_coefficients(system,ridge,SETTINGS['target_response_interval'],SETTINGS['interval_solver_max_iterations'],SETTINGS['interval_solver_tolerance'],SETTINGS['interval_solver_polish_iterations'])
      save_tensor(f,{'coefficients':torch.from_numpy(coefficients),'ridge':ridge,'case':case,'start_head_sha256':before_hash,'training_ids':group_ids,'solver_info':solver_info})
     if coefficients.shape!=(114,) or not np.isfinite(coefficients).all():raise RuntimeError('Invalid exact coefficients')
     bases.append(reconstruct_direction(system,coefficients));solver_infos.append(solver_info)
     print('C25 INTERVAL SOLVER',ridge,json.dumps(solver_info),flush=True)
    winner,scan=scan_round(head,route,group,states,system,bases,current,round_dir,c9norm['output_scale'],
     lambda theta,t:evaluate(theta,targets[t['case']],starts[t['case']]),measure_training,invariant_routes_check,dump,lambda:snapshot(out,args.drive_dir))
    if head_hash(head)!=before_hash:raise RuntimeError('Physical search failed to restore round origin')
    if winner is not None:
     if not winner['full76_verified'] or not guard(current,winner['rows'])['accepted']:raise RuntimeError('Commit requires full76 accepted candidate')
     try:
      set_layer_matrix(head,route,winner['layer']);invariant_routes_check()
      if head_hash(head)!=winner['head_sha256']:raise RuntimeError('Actual winner changed before commit')
      save_tensor(round_dir/'WINNER_LAYER.pt',{'layer':winner['layer'],'head_sha256':winner['head_sha256'],'start_head_sha256':before_hash,'spec':winner['spec']})
     except BaseException:set_layer_matrix(head,route,old_layer);raise
     current=winner['rows'];accepted+=1
    else:stalled.append(case)
    rounds+=1;record={'round':rounds,'case':case,'accepted':winner is not None,'chosen':winner['spec'] if winner else None,'head_sha256':head_hash(head),'quality':quality(current),'scan':scan,'solver_infos':solver_infos,'stalled_cases':list(stalled)}
    history.append(record);dump(round_dir/'RESULT.json',record);checkpoint();snapshot(out,args.drive_dir)
    print('C25 COMMITTED ROUND',json.dumps(record),flush=True)
   if stop is None:stop='ALL_76_TRAINING_STATES_ACCEPTED' if len(quality(current)['passed_ids'])==76 else 'MAX_CALIBRATION_ROUNDS_REACHED'
   current=measure_training(head)
   if not set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']):raise RuntimeError('Final replay lost protected passes')
   checkpoint(True);dump(out/'FINAL_TRAINING.json',current)
   dump(out/'WRITER_FROZEN.json',{'head_sha256':head_hash(head),'selected_checkpoint':'heads/calibrated/'+json.loads(progress.read_text())['checkpoint'],
    'start_head_sha256':START_HEAD_SHA,'selection_uses_training_only':True,'training_quality':quality(current),'rounds':rounds,'accepted_updates':accepted,'stop':stop,'four_route_output_layer_domain':True})
  freeze=json.loads((out/'WRITER_FROZEN.json').read_text())
  if head_hash(head)!=freeze['head_sha256']:raise RuntimeError('Final writer differs from real calibrated checkpoint')
  for p in head.parameters():p.requires_grad_(False);p.grad=None
  head.eval();versions=[p._version for p in head.parameters()]
  def writer_frozen_check():
   if [p._version for p in head.parameters()]!=versions or any(p.requires_grad or p.grad is not None for p in head.parameters()):raise RuntimeError('Writer changed after freeze')
   baseline_check();frozen_model_check()

  def online_raw_arm(writer,prep,inputs,target,start,folder,expected_hash):
   """Exactly one head proposal per completed arm; no geometry, autograd or retry here."""
   folder.mkdir(parents=True,exist_ok=True);pf=folder/'PROPOSAL.pt';rf=folder/'RESULT.json'
   if rf.exists():
    result=json.loads(rf.read_text())
    if result['head_sha256']!=expected_hash or result.get('input_sha256')!=tensor_hash(inputs['input']) or result.get('prepared_sha256')!=tensor_hash(prep):raise RuntimeError('Cached online head changed')
    return result
   if pf.exists():
    saved=torch.load(pf,map_location='cpu',weights_only=True)
    if saved['head_sha256']!=expected_hash or saved['input_sha256']!=tensor_hash(inputs['input']) or saved['prepared_sha256']!=tensor_hash(prep):raise RuntimeError('Cached proposal identity changed')
    candidate=vector(saved);proposal_seconds=saved['proposal_seconds']
   else:
    sync();ts=time.perf_counter();delta=one_raw_proposal(writer,inputs['input'].to(device),inputs['route'].to(device),c9norm['output_scale']).cpu();sync();proposal_seconds=time.perf_counter()-ts
    candidate=prep+delta;save_tensor(pf,{'theta':candidate,'head_sha256':expected_hash,'input_sha256':tensor_hash(inputs['input']),'prepared_sha256':tensor_hash(prep),'proposal_seconds':proposal_seconds})
   metrics=evaluate(candidate,target,start);writer_frozen_check();result={'head_sha256':expected_hash,'input_sha256':tensor_hash(inputs['input']),'prepared_sha256':tensor_hash(prep),'metrics':metrics,'proposal_seconds':proposal_seconds,
    'proposals':1,'Jacobian_builds':0,'SVD_calls':0,'correction_steps':0,'backward_calls':0,'teacher_fallback':False};dump(rf,result);return result
  trainrows=[]
  for t in data:
   sid=t['id'];trainrows.append({'id':sid,'case':t['case'],'reference_valid':t['reference_valid'],'arms':{'c24':initial[sid]['metrics'],'calibrated':current[sid]['metrics']}})
  newrows=[]
  for index,t in enumerate(test_specs()):
   folder,reference=prepare_state(t,freeze['head_sha256']);ef=out/'evaluation'/'new'/t['id'];ef.mkdir(parents=True,exist_ok=True);complete=ef/'COMPLETE.json'
   if complete.exists():
    row=json.loads(complete.read_text())
    if row['id']!=t['id'] or row['spec']!=t or row['final_head_sha256']!=freeze['head_sha256']:raise RuntimeError('Cached prospective evaluation identity changed')
    newrows.append(row);continue
   prep=vector(torch.load(folder/'prepared.pt',map_location='cpu',weights_only=True));inputs=torch.load(folder/'INPUT.pt',map_location='cpu',weights_only=True)
   writer_frozen_check();arms={}
   for name,writer,h in [('c24',baseline,START_HEAD_SHA),('calibrated',head,freeze['head_sha256'])]:
    result=online_raw_arm(writer,prep,inputs,targets[t['case']],starts[t['case']],ef/name,h);arms[name]=result['metrics']
    print('C25 RAW NEW',t['id'],name,'valid',reference['valid'],'pass',result['metrics']['episode_pass'],'R',result['metrics']['drift'],flush=True)
   row={'id':t['id'],'spec':t,'case':t['case'],'reference_valid':reference['valid'],'arms':arms,'final_head_sha256':freeze['head_sha256'],
    'writer_frozen_before_generation':True,'generation_retries':0,'replacement_used':False,'fresh_state_seed':True,'fresh_evaluation':False}
   dump(complete,row);newrows.append(row)
   if (index+1)%SETTINGS['archive_every_new_states']==0:snapshot(out,args.drive_dir)
  if head_hash(head)!=freeze['head_sha256']:raise RuntimeError('Frozen writer altered during evaluation')
  summary={'protocol':PROTOCOL,'complete':True,'decision':decide(trainrows,newrows),'training':{a:counts(trainrows,a) for a in ('c24','calibrated')},
   'original60_training':{a:counts([r for r in trainrows if r['id'] in original_ids],a) for a in ('c24','calibrated')},
   'promoted16_training':{a:counts([r for r in trainrows if r['id'] not in original_ids],a) for a in ('c24','calibrated')},
   'prospective_local_state_development':{a:counts(newrows,a) for a in ('c24','calibrated')},
   'new_states_by_case':{c:{a:counts([r for r in newrows if r['case']==c],a) for a in ('c24','calibrated')} for c in CASES},
   'actual_start_head_sha256':START_HEAD_SHA,'writer_head_sha256':freeze['head_sha256'],'offline_calibration_rounds':rounds,'accepted_layer_updates':accepted,'offline_stop':freeze['stop'],'stalled_cases':stalled,
   'completed_calibration_Jacobian_states':len(list((out/'calibration').rglob('JACOBIAN.pt'))),'completed_calibration_backward_calls':6*len(list((out/'calibration').rglob('JACOBIAN.pt'))),
   'all90_specs_compared_each_completed_round':all(r['scan']['n_completed_candidate_specs']==90 for r in history),'trust_grid_reset_each_round':True,'linear_rejection_used':False,
   'full76_physical_audit_before_each_commit':True,'new_Adam_steps':0,'four_route_output_layer_domain':True,'normalisation_refitted':False,'guard_change_from_C24':SETTINGS['guard_change_from_C24'],
   'original60_complete_training_passes_preserved':set(original_ids).issubset(quality(current)['passed_ids']),'all_initial69_passes_preserved':set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']),
   'online_write_Jacobian_builds':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,'online_write_backward_calls':0,'online_proposals_per_arm':1,'teacher_fallback_used':False,
   'promoted_inspected_evaluation_states':16,'new_training_preparation_states':0,'new_evaluation_preparation_states':16,'evaluation_state_generation_retries':0,'replacement_states_used':False,
   'offline_evaluation_preparation_counters':counters,'test_based_checkpoint_selection':False,'old_inspected_states_promoted_to_training':True,'evaluation_influenced_protocol_design':True,
   'fresh_state_seeds':True,'fresh_evaluation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,'requested_output_bit_supplied':True,
   'parent_C24_decision_unchanged':parent_info['summary']['decision'],'scope':SETTINGS['scope']}
  writer_frozen_check();w.assign_flat(params,torch.zeros(36864,device=device));dump(out/'SUMMARY.json',summary);snapshot(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
