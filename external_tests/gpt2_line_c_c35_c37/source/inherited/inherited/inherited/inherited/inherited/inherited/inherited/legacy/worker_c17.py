"""C17 guarded OFFLINE fitting; online branch is only features -> one head -> commit."""
import argparse,fcntl,hashlib,io,json,os,random,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch,transformers,peft
import writer_core as w
import run_l5_audit as audit
from runtime_a100 import a100_device
from raw_writer import StateWriter,route_from_features,prepared_input,one_raw_proposal
from functional_writer import chart_spec,last_logits
from guarded_update import attempt_update,cpu_state
from correction_engine import run_correction
from geometry import null_project
from c17_protocol import (PROTOCOL,RESULT,SETTINGS,C15_SHA,C9_SHA,HEAD_SHA,CHECKPOINT,REVISION,CASES,STATES,GATES,
 WARM_MEMBER,WARM_CP_SHA,WARM_HEAD_SHA,training_specs,test_specs,validate_c15,source_hashes,inherited_check,
 sha,endpoint_gate,episode_gate,target_from_reads,snapshot,quality,guard,counts,decide)
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
 inherited_check(ROOT);parent_info=validate_c15(out/'parent_c15.zip');device,gpu=a100_device(torch)
 ident={'protocol':PROTOCOL,'settings':SETTINGS,'parent_c15_sha256':C15_SHA,'source_hashes':source_hashes(ROOT),
  'gpu':gpu,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'model_revision':REVISION}
 if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('Runtime/source changed; refuse incompatible resume')
 dump(out/'identity.json',ident)
 if not (out/'source').exists():shutil.copytree(ROOT,out/'source',ignore=shutil.ignore_patterns('__pycache__','build_*'))
 split={'training':training_specs(),'prospective_new_states':test_specs(),'old_inspected16_used_for_tuning':False,'generate_new_after_final_freeze':True}
 if (out/'SPLIT.json').exists() and json.loads((out/'SPLIT.json').read_text())!=split:raise RuntimeError('Fixed split changed')
 dump(out/'SPLIT.json',split)
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,args.drive_dir);return
 snapshot(out,args.drive_dir)
 cfg=json.loads((ROOT/'config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION);w.set_seed(82931)
 print('C17 Loading pinned GPT-2 once; actual warm1000 start, guarded offline training; online no repair.',flush=True)
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
  v={'bits':bits,'evaluation':ev,'drift':float(drift),'kl':float(kl),'pass':False}
  v['pass']=endpoint_gate(v,target);v['gates']={'bits_match':bits==target,'response':drift<=.002,'KL':kl<=.005,
   'exact_read_margin':min(ev['exact_margins'])>=.05,'development_read_margin':min(ev['heldout_margins'])>=.05}
  if start is not None:v['gates'].update(output_flip=bits[3]!=start[3],unselected_preserved=bits[:3]==start[:3]);v['episode_pass']=episode_gate(start,v)
  al=w.next_logits(model,tok,anchors,device);pr=(al[:,response_ids]-logits0[:,response_ids]).abs().max(1).values
  pk=(logits0.softmax(-1)*(logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
  v['per_anchor_diagnostics']={'maximum_selected_logit_change':float(pr.max()),'maximum_KL':float(pk.max()),
   'selected_logit_changes':pr.cpu().tolist(),'KL_values':pk.cpu().tolist(),'not_additional_gates':True}
  sync();v['timing']={'commit_seconds':commit_seconds,'verification_seconds':time.perf_counter()-t0-commit_seconds};frozen_model_check();return v

 with zipfile.ZipFile(out/'parent_c15.zip') as parent,zipfile.ZipFile(io.BytesIO(parent.read('parent_c9.zip'))) as c9,zipfile.ZipFile(io.BytesIO(c9.read('parent_c8.zip'))) as c8:
  basecp=load_obj(c9,CHECKPOINT);warmcp=load_obj(parent,WARM_MEMBER)
  if hashlib.sha256(parent.read(WARM_MEMBER)).hexdigest()!=WARM_CP_SHA or warmcp['stage']!='warm' or warmcp['steps']!=1000:raise RuntimeError('True warm checkpoint mismatch')
  train_ids=[t['id'] for t in training_specs()]
  if warmcp['train_ids']!=train_ids:raise RuntimeError('Exact 36 training IDs required')
  c9norm={k:v.to(device) for k,v in basecp['normalisation'].items()};extra_norm={k:v.to(device) for k,v in warmcp['extra_norm'].items()}
  if set(extra_norm)!={'mean','scale'} or extra_norm['mean'].shape!=(70,) or extra_norm['scale'].shape!=(70,) or any(not torch.isfinite(v).all() for v in extra_norm.values()) or not (extra_norm['scale']>0).all():raise RuntimeError('Saved extra normalisation invalid')
  sketchcp=load_obj(parent,'STATE_SKETCH.pt');sketch=sketchcp['matrix'].to(device)
  if sketch.shape!=(36864,64) or not torch.isfinite(sketch).all() or sketchcp['seed']!=615031:raise RuntimeError('Saved state sketch invalid')
  baseline=StateWriter().to(device);baseline.load_state_dict(warmcp['head']);baseline.eval()
  for p in baseline.parameters():p.requires_grad_(False)
  if head_hash(baseline)!=WARM_HEAD_SHA:raise RuntimeError('Actual warm head mismatch')
  head=StateWriter().to(device);head.load_state_dict(warmcp['head']);head.eval()
  baseline_versions=[p._version for p in baseline.parameters()]
  def baseline_check():
   if [p._version for p in baseline.parameters()]!=baseline_versions or any(p.requires_grad or p.grad is not None for p in baseline.parameters()):raise RuntimeError('Warm control changed')
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
   sid=t['id'];case=t['case'];b='states/training/'+sid+'/';prep=vector(load_obj(parent,b+'prepared.pt'));teacher=vector(load_obj(parent,b+'teacher.pt'))
   f=load_obj(parent,'evaluation/training/'+sid+'/ONLINE_INPUT.pt');ref=json.loads(parent.read(b+'REFERENCE.json'))
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
  def measure_training(writer):
   ans={}
   for j,t in enumerate(data):
    delta=one_raw_proposal(writer,t['input'],t['route'],c9norm['output_scale']).cpu();m=evaluate(t['prep']+delta,targets[t['case']],starts[t['case']])
    ans[t['id']]={'reference_valid':t['reference_valid'],'metrics':m}
    if (j+1)%9==0:print('C17 TRAIN GATE',j+1,'/36',flush=True)
   baseline_check();return ans
  initialfile=out/'WARM_TRAINING_BASELINE.json'
  if initialfile.exists():initial=json.loads(initialfile.read_text())
  else:initial=measure_training(baseline);dump(initialfile,initial)
  initial_quality=quality(initial);print('C17 ACTUAL WARM TRAINING',json.dumps(initial_quality),flush=True)
  if any(not all(row['metrics']['gates'][g] for g in GATES if g!='response') for row in initial.values()):raise RuntimeError('Warm replay loses a previously satisfied non-response gate')

  folder=out/'heads'/'guarded';folder.mkdir(parents=True,exist_ok=True);progress=folder/'PROGRESS.json'
  optimiser=torch.optim.AdamW(head.parameters(),lr=SETTINGS['head_lr'],weight_decay=0.)
  attempt=accepted=stalls=0;lr=SETTINGS['head_lr'];history=[];current=initial;stop=None
  def checkpoint(done=False):
   name='checkpoint_%06d.pt'%attempt;f=folder/name
   save_tensor(f,{'head':cpu_state(head),'optimiser':cpu_tree(optimiser.state_dict()),'attempts':attempt,'accepted_updates':accepted,
    'consecutive_rejections':stalls,'lr':lr,'history':history,'current_training':current,'train_ids':train_ids,'start_head_sha256':WARM_HEAD_SHA,
    'head_sha256':head_hash(head),'extra_norm':cpu_tree(extra_norm),'done':done,'stop':stop})
   dump(progress,{'checkpoint':name,'checkpoint_sha256':sha(f),'attempts':attempt,'accepted_updates':accepted,'done':done})
   for old in folder.glob('checkpoint_*.pt'):
    if old!=f:old.unlink()
  if progress.exists():
   pr=json.loads(progress.read_text());name=pr['checkpoint']
   if Path(name).name!=name or '\\' in name or sha(folder/name)!=pr['checkpoint_sha256']:raise RuntimeError('Actual C17 checkpoint corrupt/unsafe')
   cp=torch.load(folder/name,map_location='cpu',weights_only=True)
   if cp['train_ids']!=train_ids or cp['start_head_sha256']!=WARM_HEAD_SHA or cp['attempts']!=pr['attempts'] or any(not torch.equal(cp['extra_norm'][k].to(device),v) for k,v in extra_norm.items()):raise RuntimeError('Actual optimizer/training/normalisation mismatch')
   head.load_state_dict(cp['head']);optimiser.load_state_dict(cp['optimiser']);attempt=cp['attempts'];accepted=cp['accepted_updates'];stalls=cp['consecutive_rejections'];lr=cp['lr'];history=cp['history'];current=cp['current_training'];stop=cp['stop']
   if head_hash(head)!=cp['head_sha256'] or not set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']):raise RuntimeError('Actual guarded checkpoint identity/pass-set mismatch')
   print('C17 RESTORE real attempts',attempt,'accepted',accepted,'lr',lr,flush=True)
  else:checkpoint()

  if not (out/'WRITER_FROZEN.json').exists():
   while attempt<SETTINGS['max_offline_attempts'] and stop is None:
    q=quality(current)
    if len(q['passed_ids'])==36:stop='ALL_36_TRAINING_STATES_ACCEPTED';break
    if stalls>=SETTINGS['max_consecutive_rejections'] or lr<SETTINGS['minimum_lr']:stop='GUARDED_SEARCH_STALLED';break
    eligible=[c for c in CASES if any(t['case']==c and t['id'] not in q['passed_ids'] for t in data)]
    case=eligible[attempt%len(eligible)];group=[t for t in data if t['case']==case];attempt_dir=out/'attempts'/('%06d'%(attempt+1));attempt_dir.mkdir(parents=True,exist_ok=True)
    for pg in optimiser.param_groups:pg['lr']=lr
    # Inactive expert Adam moments must not move other operand routes.
    for i,expert in enumerate(head.experts):
     for p in expert.parameters():p.requires_grad_(i==CASES.index(case));p.grad=None
    def loss_fn():
     total=0.;resp_total=mse_total=0.
     for t in group:
      pred=head(t['input'].unsqueeze(0),t['route'].reshape(1))[0];candidate=t['prep'].to(device)+pred*c9norm['output_scale']
      logits=last_logits(model,anchor_batch,candidate,spec);e=logits[:,response_ids].mean(0)-response0
      response=(e.double()/SETTINGS['internal_response_target']).square().mean()
      y=(t['teacher']-t['prep']).to(device)/c9norm['output_scale'];mse=(pred-y).square().mean()
      loss=(response+SETTINGS['teacher_mse_weight']*mse)/len(group);loss.backward()
      resp_total+=float(response.detach())/len(group);mse_total+=float(mse.detach())/len(group);total+=float(loss.detach())
     # Gradients already accumulated sequentially; avoid retaining nine GPT-2 graphs.
     return torch.tensor(total,device=device),{'response_loss':resp_total,'teacher_MSE':mse_total,'case':case,'states':len(group),'state_backward_calls':len(group),'gradients_accumulated':True}
    trial_index=[0]
    def trial_eval():
     r=measure_training(head);dump(attempt_dir/('trial_%02d.json'%trial_index[0]),{'head_sha256':head_hash(head),'rows':r});trial_index[0]+=1;return r
    print('C17 OFFLINE ATTEMPT',attempt+1,case,'lr',lr,'current_passed',len(q['passed_ids']),flush=True)
    updated,record=attempt_update(head,optimiser,loss_fn,trial_eval,lambda r:guard(current,r),SETTINGS['head_backtracking'],SETTINGS['gradient_clip'],
     lambda bt,a:print('C17 HEAD BACKTRACK',bt,'alpha',a,'full 36-state forward audit',flush=True))
    attempt+=1
    if updated is not None:current=updated;accepted+=1;stalls=0
    else:stalls+=1;lr*=.5
    frozen_model_check();baseline_check();record.update(attempt=attempt,case=case,head_sha256=head_hash(head),lr_next=lr,accepted_updates=accepted)
    dump(attempt_dir/'RESULT.json',record);history.append(record);checkpoint();snapshot(out,args.drive_dir)
    print('C17 ACCEPTED STATE',json.dumps({'attempt':attempt,'accepted':record['accepted'],'training_quality':quality(current),'lr_next':lr}),flush=True)
   if stop is None:stop='MAX_OFFLINE_ATTEMPTS_REACHED'
   # Independent actual-forward replay at the final accepted checkpoint, never an unaccepted candidate.
   current=measure_training(head)
   if not set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']):raise RuntimeError('Final replay lost protected training passes')
   checkpoint(True)
   dump(out/'FINAL_TRAINING.json',current)
   dump(out/'WRITER_FROZEN.json',{'head_sha256':head_hash(head),'selected_checkpoint':'heads/guarded/'+json.loads(progress.read_text())['checkpoint'],
    'start_head_sha256':WARM_HEAD_SHA,'selection_uses_training_only':True,'training_quality':quality(current),'attempts':attempt,'accepted_updates':accepted,'stop':stop})
  freeze=json.loads((out/'WRITER_FROZEN.json').read_text())
  if head_hash(head)!=freeze['head_sha256']:raise RuntimeError('Final frozen writer does not match actual checkpoint')
  for p in head.parameters():p.requires_grad_(False);p.grad=None
  head.eval();versions=[p._version for p in head.parameters()]
  def writer_frozen_check():
   if [p._version for p in head.parameters()]!=versions or any(p.requires_grad or p.grad is not None for p in head.parameters()):raise RuntimeError('Writer changed after freeze')
   baseline_check();frozen_model_check()
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
  def prepare_new(t):
   folder=out/'states'/'new'/t['id'];folder.mkdir(parents=True,exist_ok=True);rf=folder/'REFERENCE.json'
   if rf.exists():return folder,json.loads(rf.read_text())
   prep0,teacher0=base_data[t['base_id']];qf=folder/'perturbation.pt'
   if qf.exists():
    qc=torch.load(qf,map_location='cpu',weights_only=True)
    if qc['spec']!=t:raise RuntimeError('Fixed new-state seed/spec changed')
    q=vector(qc);geom=qc['geometry']
   else:
    j=load_obj(c8,'states/'+t['base_id']+'/jacobian.pt')['J'].double().numpy();projected,geom=null_project(j,np.random.default_rng(t['seed']).normal(size=36864));qn=float(np.linalg.norm(projected))
    if not qn>0:raise RuntimeError('Zero fixed perturbation; no replacement allowed')
    q=torch.from_numpy(projected*(t['norm']/qn)).float();save_tensor(qf,{'theta':q,'spec':t,'geometry':geom})
   prep,ph,ps=offline_correct(folder/'offline_preparation',prep0+q);teacher,th,ts=offline_correct(folder/'offline_teacher',teacher0+q)
   save_tensor(folder/'prepared.pt',{'theta':prep});save_tensor(folder/'teacher.pt',{'theta':teacher})
   pm=evaluate(prep,starts[t['case']]);tm=evaluate(teacher,targets[t['case']],starts[t['case']]);f=fresh_inputs(prep,targets[t['case']][3]);save_tensor(folder/'INPUT.pt',f)
   r={'spec':t,'valid':bool(pm['pass'] and tm['episode_pass']),'preparation':pm,'teacher':tm,'generation_retries':0,'replacement_used':False,
    'writer_frozen_before_generation':freeze['head_sha256'],'offline_geometry':True,'projection':geom,'preparation_seconds':ps,'teacher_seconds':ts};dump(rf,r);writer_frozen_check();return folder,r

  def online_raw_arm(writer,prep,inputs,target,start,folder,expected_hash):
   """Exactly one head proposal per completed arm; no geometry, autograd or retry here."""
   folder.mkdir(parents=True,exist_ok=True);pf=folder/'PROPOSAL.pt';rf=folder/'RESULT.json'
   if rf.exists():
    result=json.loads(rf.read_text())
    if result['head_sha256']!=expected_hash:raise RuntimeError('Cached online head changed')
    return result
   if pf.exists():
    saved=torch.load(pf,map_location='cpu',weights_only=True)
    if saved['head_sha256']!=expected_hash or saved['input_sha256']!=tensor_hash(inputs['input']) or saved['prepared_sha256']!=tensor_hash(prep):raise RuntimeError('Cached proposal identity changed')
    candidate=vector(saved);proposal_seconds=saved['proposal_seconds']
   else:
    sync();ts=time.perf_counter();delta=one_raw_proposal(writer,inputs['input'].to(device),inputs['route'].to(device),c9norm['output_scale']).cpu();sync();proposal_seconds=time.perf_counter()-ts
    candidate=prep+delta;save_tensor(pf,{'theta':candidate,'head_sha256':expected_hash,'input_sha256':tensor_hash(inputs['input']),'prepared_sha256':tensor_hash(prep),'proposal_seconds':proposal_seconds})
   metrics=evaluate(candidate,target,start);writer_frozen_check();result={'head_sha256':expected_hash,'metrics':metrics,'proposal_seconds':proposal_seconds,
    'proposals':1,'Jacobian_builds':0,'SVD_calls':0,'correction_steps':0,'backward_calls':0,'teacher_fallback':False};dump(rf,result);return result
  trainrows=[]
  for t in data:
   sid=t['id'];trainrows.append({'id':sid,'case':t['case'],'reference_valid':True,'arms':{'warm':initial[sid]['metrics'],'guarded':current[sid]['metrics']}})
  newrows=[]
  for index,t in enumerate(test_specs()):
   folder,reference=prepare_new(t);ef=out/'evaluation'/'new'/t['id'];ef.mkdir(parents=True,exist_ok=True);complete=ef/'COMPLETE.json'
   if complete.exists():newrows.append(json.loads(complete.read_text()));continue
   prep=vector(torch.load(folder/'prepared.pt',map_location='cpu',weights_only=True));inputs=torch.load(folder/'INPUT.pt',map_location='cpu',weights_only=True)
   # State-level timing includes required feature extraction separately from head/commit/verify.
   arms={}
   for name,writer,h in [('warm',baseline,WARM_HEAD_SHA),('guarded',head,freeze['head_sha256'])]:
    r=online_raw_arm(writer,prep,inputs,targets[t['case']],starts[t['case']],ef/name,h);arms[name]=r['metrics']
    print('C17 RAW NEW',t['id'],name,'pass',r['metrics']['episode_pass'],'R',r['metrics']['drift'],flush=True)
   row={'id':t['id'],'case':t['case'],'reference_valid':reference['valid'],'arms':arms};dump(complete,row);newrows.append(row)
   if (index+1)%SETTINGS['archive_every_new_states']==0:snapshot(out,args.drive_dir)
  if head_hash(head)!=freeze['head_sha256']:raise RuntimeError('Frozen writer altered during evaluation')
  summary={'protocol':PROTOCOL,'complete':True,'decision':decide(trainrows,newrows),'training':{a:counts(trainrows,a) for a in ('warm','guarded')},
   'prospective_local_state_development':{a:counts(newrows,a) for a in ('warm','guarded')},
   'new_states_by_case':{c:{a:counts([r for r in newrows if r['case']==c],a) for a in ('warm','guarded')} for c in CASES},
   'actual_start_head_sha256':WARM_HEAD_SHA,'writer_head_sha256':freeze['head_sha256'],'offline_attempts':attempt,'accepted_head_updates':accepted,'offline_stop':freeze['stop'],
   'completed_offline_state_backward_calls':sum(r['terms']['state_backward_calls'] for r in history),
   'inherited_failed_functional_stage_continued':False,
   'initial_complete_training_passes_preserved':set(initial_quality['passed_ids']).issubset(quality(current)['passed_ids']),
   'online_write_Jacobian_builds':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,'online_write_backward_calls':0,
   'teacher_fallback_used':False,'offline_state_preparation_geometry':True,'fresh_state_seeds':True,'fresh_evaluation':False,
   'internal_parameter_update':False,'controller_dispatch_tested':False,'requested_output_bit_supplied':True,
   'same_four_trained_input_classes':True,'old_inspected16_not_used_for_tuning':True,'test_based_checkpoint_selection':False,
   'parent_C15_decision_unchanged':parent_info['summary']['decision'],'scope':SETTINGS['scope']}
  writer_frozen_check();w.assign_flat(params,torch.zeros(36864,device=device));dump(out/'SUMMARY.json',summary);snapshot(out,args.drive_dir);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
