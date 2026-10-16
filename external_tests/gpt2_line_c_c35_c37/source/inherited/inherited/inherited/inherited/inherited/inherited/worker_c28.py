"""One numerical offline fit per arm, protected true forwards, frozen development evaluation."""
import argparse,io,json,hashlib,random,sys,time,shutil,os,fcntl
from pathlib import Path
import numpy as np
import torch
import writer_core as w
import run_l5_audit as audit
from raw_writer import StateWriter,prepared_input,route_from_features
from runtime_a100 import a100_device
from c28_protocol import *
from c27_features import ResidualWriter,fit_map,map_features,tensor_hash,state_hash,activation_audit
from c28_features import feature_map,FourRouteWriter
from c28_math import directions,candidate
from c27_inputs import validate_saved_input
ROOT=Path(__file__).resolve().parent


def save_tensor(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);os.replace(tmp,path)


def load(path):return torch.load(path,map_location='cpu',weights_only=True)


def vector(v):
 if isinstance(v,dict):v=v['theta']
 require(isinstance(v,torch.Tensor) and v.dtype==torch.float32 and v.shape==(36864,) and torch.isfinite(v).all(),'Finite original memory chart required')
 return v.detach().cpu().clone()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--drive-dir',default='');a=ap.parse_args();out=Path(a.out)
 lock=open(out.parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 check_inherited(ROOT)
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,a.drive_dir);return
 identity=json.loads((out/'identity.json').read_text());require(identity['settings']==SETTINGS and identity['source_hashes']==source_hashes(ROOT),'Runtime source mismatch')
 provenance=json.loads((out/'PARENT_PROVENANCE.json').read_text())
 for n,r in provenance['members'].items():require(file_sha(out/n)==r['sha256'],'Compact original member changed: '+n)
 device,gpu=a100_device(torch);torch.set_num_threads(min(4,os.cpu_count() or 1));w.set_seed(82931)
 import transformers,peft
 runtime={'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':gpu}
 if (out/'RUNTIME.json').exists():require(json.loads((out/'RUNTIME.json').read_text())==runtime,'C28 resume runtime differs')
 else:dump(out/'RUNTIME.json',runtime)
 print('C28 Loading pinned GPT-2 once; frozen C27 base, four-route matched residual features.',flush=True)
 cfg=json.loads((ROOT/'inherited/legacy/config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
 tok,model,params=w.make_model(cfg,device);model.config.use_cache=False
 for p in model.parameters():p.requires_grad_(False)
 pins=json.loads((out/'parent/MODEL_PINS.json').read_text());ah=hashlib.sha256()
 for n,p in model.named_parameters():
  if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
 require(ah.hexdigest()==pins['lora_A_sha256'] and getattr(model.config,'_commit_hash',None)==REVISION,'Original model/A differs')
 require([list(p.shape) for p in params]==pins['chart_shapes'] and w.param_flat(params).abs().max()==0,'Original zero-B chart differs')
 symbols=w.single_token_pair(tok);readids=[s[1] for s in symbols];require(json.loads(json.dumps(symbols))==pins['symbols'],'Read token identity differs')
 rng=random.Random(82931);anchors=rng.sample(w.ANCHORS,cfg['anchor_count']);responseids=rng.sample(list(range(100,min(50000,tok.vocab_size))),cfg['response_dim'])
 _,exact,_,dev=audit.prompts()
 with torch.no_grad():logits0=w.next_logits(model,tok,anchors,device).detach();response0=logits0[:,responseids].mean(0)
 model_versions={n:p._version for n,p in model.named_parameters() if 'lora_B.default.' not in n}
 cp=load(out/'parent/C25_FINAL_CHECKPOINT.pt');require(file_sha(out/'parent/C25_FINAL_CHECKPOINT.pt')==PARENT_CP_SHA and state_hash(cp['head'])==HEAD_SHA,'Actual C25 frozen head differs')
 raw_base=StateWriter().to(device);raw_base.load_state_dict(cp['head']);raw_base.eval()
 branch=load(out/'parent/C27_BRANCH.pt');fm0=load(out/'parent/C27_FEATURE_MAP.pt')
 require(file_sha(out/'parent/C27_BRANCH.pt')==PARENT_BRANCH_CP_SHA and tensor_hash(branch['residual_matrix'])==PARENT_MATRIX_SHA,'C27 frozen residual differs')
 base=ResidualWriter(raw_base,fm0,'state_bypass').to(device).eval();base.set_matrix(branch['residual_matrix'])
 for p in base.parameters():p.requires_grad_(False);p.grad=None
 base_versions={n:p._version for n,p in base.named_parameters()}
 norm=load(out/'parent/C9_CHECKPOINT.pt')['normalisation'];extra_norm=cp['extra_norm']
 gnorm={k:v.to(device) for k,v in norm.items()};gextra={k:v.to(device) for k,v in extra_norm.items()}
 require(float(norm['output_scale'])==SETTINGS['output_scale'],'Output scale changed')
 split=json.loads((out/'parent/SPLIT.json').read_text());train_ids=[t['id'] for t in split['training']];original=[t['id'] for t in split['original60']]
 parent_final=branch['training_rows'];targets={c:STARTS[c][:3]+[STARTS[c][0]^STARTS[c][1]] for c in CASES}
 require(len(train_ids)==76 and len(set(train_ids))==76 and len(original)==60,'Cohort changed')
 def invariant():
  require(all(p._version==base_versions[n] for n,p in base.named_parameters()),'Original C25 writer changed')
  require(tensor_hash(base.matrix)==PARENT_MATRIX_SHA,'Frozen C27 branch changed')
  require(all(p._version==model_versions[n] for n,p in model.named_parameters() if n in model_versions),'Frozen model/base A changed')
  require(all(not p.requires_grad and p.grad is None for p in base.parameters()) and all(not p.requires_grad and p.grad is None for p in model.parameters()),'Unexpected model/writer gradients')
 def sync():torch.cuda.synchronize(device)
 @torch.no_grad()
 def evaluate(theta,target,start):
  theta=vector(theta);sync();ts=time.perf_counter();w.assign_flat(params,theta.to(device))
  # Same prompt batches, padding, read positions and dtype as C25; eliminate only duplicate forwards.
  el=w.next_logits(model,tok,exact,device)[:,readids];dl=w.next_logits(model,tok,dev,device)[:,readids];al=w.next_logits(model,tok,anchors,device)
  bits=[int(v) for v in el.argmax(-1).cpu().tolist()];tt=target*3
  em=[float((row[b]-row[1-b]).cpu()) for row,b in zip(el,target)];dm=[float((row[b]-row[1-b]).cpu()) for row,b in zip(dl,tt)]
  residual=al[:,responseids].mean(0)-response0;drift=float(residual.abs().max().cpu())
  kvals=(logits0.softmax(-1)*(logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1);kl=max(0.,float(kvals.mean().cpu()))
  gates={'bits_match':bits==target,'output_flip':bits[3]!=start[3],'unselected_preserved':bits[:3]==start[:3],
         'response':drift<=.002,'KL':kl<=.005,'exact_read_margin':min(em)>=.05,'development_read_margin':min(dm)>=.05}
  v={'theta_sha256':tensor_hash(theta),'bits':bits,'evaluation':{'exact_margins':em,'heldout_margins':dm,'minimum_exact_margin':min(em),'minimum_heldout_margin':min(dm)},
     'drift':drift,'kl':kl,'gates':gates,'pass':all(gates[g] for g in ('bits_match','response','KL','exact_read_margin','development_read_margin')),
     'episode_pass':all(gates.values()),'response_residual':residual.cpu().double().tolist(),
     'response_residual_float64_aggregation':(al[:,responseids].double().mean(0)-logits0[:,responseids].double().mean(0)).cpu().tolist(),
     'per_anchor_diagnostics':{'maximum_selected_logit_change':float((al[:,responseids]-logits0[:,responseids]).abs().max().cpu()),'maximum_KL':float(kvals.max().cpu()),'not_additional_gates':True}}
  sync();v['verification_seconds']=time.perf_counter()-ts;invariant();return v
 def item(t,cohort):
  sid=t['id'];folder=out/'data'/cohort/sid;f=load(folder/'INPUT.pt');prep=vector(load(folder/'prepared.pt'));ref=json.loads((folder/'REFERENCE.json').read_text())
  x,route,input_audit=validate_saved_input(f,gnorm,gextra,CASES.index(t['case']),device)
  af=folder/'C28_INPUT_SCHEMA_AUDIT.json'
  if af.exists():require(json.loads(af.read_text())==input_audit,'Input audit differs')
  else:dump(af,input_audit)
  return {'id':sid,'case':t['case'],'prep':prep,'x':x,'route':route,'valid':bool(ref['valid'])}
 data=[item(t,'training') for t in split['training']];bycase={c:[t for t in data if t['case']==c] for c in CASES}
 require(all(len(ts)==19 for ts in bycase.values()) and all(t['valid'] for t in data),'Fixed76 training references invalid')
 calls={'offline_or_audit':0,'new_frozen_evaluation':0,'resume_evaluation_audit':0}
 @torch.no_grad()
 def proposal(writer,t,purpose='offline_or_audit'):
  calls[purpose]+=1;return t['prep']+(writer(t['x'].unsqueeze(0),t['route'].reshape(1))[0]*SETTINGS['output_scale']).cpu()
 def measure(writer,items,cache=None):
  rows={}
  for index,t in enumerate(items,1):
   theta=proposal(writer,t)
   if cache is not None and t['id'] in cache:
    require(cache[t['id']]['metrics']['theta_sha256']==tensor_hash(theta),'Cached physical theta differs');rows[t['id']]=cache[t['id']]
   else:rows[t['id']]={'reference_valid':t['valid'],'metrics':evaluate(theta,targets[t['case']],STARTS[t['case']])}
   if index%19==0:print('C28 ACTUAL GATE',index,'/',len(items),flush=True)
  return rows
 bf=out/'BASELINE_REPLAY.json'
 if bf.exists():baseline=json.loads(bf.read_text());measure(base,data,baseline)
 else:baseline=measure(base,data);replay_equal(parent_final,baseline);dump(bf,baseline);snapshot(out,a.drive_dir)
 require(metrics_counts(baseline)['passed']==76,'Frozen C27 training76 did not replay')
 counters_file=out/'OFFLINE_COUNTERS.json'
 counters=json.loads(counters_file.read_text()) if counters_file.exists() else {'new_Jacobian_builds':0,'response_backward_calls':0}
 geometry={}
 # Only training-state written endpoints enter this new offline geometry.
 for index,t in enumerate(data,1):
  gf=out/'geometry'/(t['id']+'.pt');theta=proposal(base,t)
  if gf.exists():
   g=load(gf);require(g['theta_sha256']==tensor_hash(theta) and g['input_sha256']==tensor_hash(t['x']) and g['C27_matrix_sha256']==PARENT_MATRIX_SHA,'Saved geometry identity differs')
  else:
   sync();geometry_started=time.perf_counter();w.assign_flat(params,theta.to(device))
   for p in params:p.requires_grad_(True)
   try:
    with torch.enable_grad():
     response=w.response_vector(model,tok,anchors,responseids,device);rows=[]
     for k in range(response.numel()):
      grads=torch.autograd.grad(response[k],params,retain_graph=k+1<response.numel(),allow_unused=False);rows.append(torch.cat([v.reshape(-1) for v in grads]).detach().cpu().double())
     J=torch.stack(rows);rr=(response.detach().cpu()-response0.detach().cpu()).double()
    require(J.shape==(6,36864) and bool(torch.isfinite(J).all()),'Offline Jacobian invalid')
    require(np.max(np.abs(rr.numpy()-np.array(baseline[t['id']]['metrics']['response_residual'])))<=2e-6,'Offline response replay differs')
    g={'J':J,'residual':rr,'theta_sha256':tensor_hash(theta),'input_sha256':tensor_hash(t['x']),'C27_matrix_sha256':PARENT_MATRIX_SHA,'offline_training_only':True,'committed_backward_calls':6,'geometry_seconds':time.perf_counter()-geometry_started}
    save_tensor(gf,g);counters['new_Jacobian_builds']+=1;counters['response_backward_calls']+=6;dump(counters_file,counters)
   finally:
    for p in params:p.requires_grad_(False);p.grad=None
   invariant()
  geometry[t['id']]=g
  require(g['J'].shape==(6,36864) and bool(torch.isfinite(g['J']).all()),'Cached Jacobian invalid')
  print('C28 OFFLINE TRAINING GEOMETRY',index,'/76',t['id'],flush=True)
  if index%19==0:snapshot(out,a.drive_dir)
 writers={};arm_results={};freeze={}
 for arm in ARMS:
  folder=out/'arms'/arm;folder.mkdir(parents=True,exist_ok=True);maps=[]
  for case in CASES:
   p=folder/case/'FEATURE_MAP.pt';ts=bycase[case];ids=[t['id'] for t in ts]
   if p.exists():fm=load(p)
   else:fm=feature_map(torch.stack([t['x'] for t in ts]).cpu().double().numpy(),ids,arm);save_tensor(p,fm)
   require(fm['kind']==arm and fm['fit_ids']==ids and not fm['fit_uses_development'],'Feature fit scope differs');maps.append(fm)
  writer=FourRouteWriter(base,maps).to(device).eval();writers[arm]=writer;frfile=folder/'FROZEN.json'
  if frfile.exists():
   fr=json.loads(frfile.read_text());cpfile=folder/'FINAL_CHECKPOINT.pt';require(file_sha(cpfile)==fr['checkpoint_sha256'],'Frozen checkpoint digest differs');acp=load(cpfile)
   require(acp['arm']==arm and acp['parent_matrix_sha256']==PARENT_MATRIX_SHA,'Frozen arm parent differs')
   for route in range(4):
    require(file_sha(folder/CASES[route]/'FEATURE_MAP.pt')==fr['feature_map_sha256'][route],'Frozen feature-map identity differs');writer.set_route(route,acp['matrices'][route])
   require(tensor_hash(writer.matrices)==fr['matrices_sha256'],'Frozen matrices differ');measure(writer,data,acp['training_rows']);arm_results[arm]=acp['training_rows'];freeze[arm]=fr;continue
  route_selections={}
  for route,case in enumerate(CASES):
   routefolder=folder/case;selected_file=routefolder/'ROUTE_SELECTION.json';ts=bycase[case];ids=[t['id'] for t in ts];before={sid:baseline[sid] for sid in ids}
   if selected_file.exists():
    selected=json.loads(selected_file.read_text());mp=routefolder/selected['matrix_member'];require(file_sha(mp)==selected['matrix_file_sha256'],'Selected route checkpoint differs');m=load(mp)['matrix'];require(tensor_hash(m)==selected['matrix_sha256'],'Selected route matrix differs');writer.set_route(route,m);route_selections[case]=selected;continue
   with torch.no_grad():Phi=torch.stack([writer.maps[route](t['x'].unsqueeze(0))[0].cpu() for t in ts]).double().numpy()
   J=np.stack([geometry[sid]['J'].numpy() for sid in ids]);E=np.array([baseline[sid]['metrics']['response_residual'] for sid in ids]);solverfile=routefolder/'SAVED_DIRECTIONS.pt'
   if solverfile.exists():
    sv=load(solverfile);require(sv['Phi_sha256']==tensor_hash(torch.from_numpy(Phi)) and sv['ids']==ids,'Saved solve feature identity differs');from joint_calibration import prepare_system
    system=prepare_system(J,Phi,E,SETTINGS['output_scale'],1e-12,1e-8);bases=[b.numpy() for b in sv['bases']]
   else:
    system,bases,infos=directions(J,Phi,E);save_tensor(solverfile,{'Phi_sha256':tensor_hash(torch.from_numpy(Phi)),'ids':ids,'bases':[torch.from_numpy(b) for b in bases],'infos':infos})
   records=[];aliases={}
   for k,spec in enumerate(specs(),1):
    m,linear=candidate(system,bases,spec);mh=tensor_hash(m);mp=routefolder/'matrices'/(mh+'.pt')
    if not mp.exists():save_tensor(mp,{'matrix':m,'parent_C27_matrix_sha256':PARENT_MATRIX_SHA})
    else:require(tensor_hash(load(mp)['matrix'])==mh,'Saved actual candidate matrix differs')
    writer.set_route(route,load(mp)['matrix']);rf=routefolder/'candidates'/spec['id']/'COMPLETE.json'
    if rf.exists():
     rec=json.loads(rf.read_text());require(rec['spec']==spec and rec['matrix_sha256']==mh,'Candidate identity differs');measure(writer,ts,rec['rows'])
    else:
     actual=measure(writer,ts,aliases.get(mh));g=route_guard(before,actual,ids)
     rec={'spec':spec,'matrix_sha256':mh,'matrix_member':mp.relative_to(routefolder).as_posix(),'matrix_file_sha256':file_sha(mp),'rows':actual,'guard':g,'linear':linear,'linear_is_not_finite_certificate':True};dump(rf,rec)
    require(rec['guard']==route_guard(before,rec['rows'],ids),'Stored candidate guard differs');aliases[mh]=rec['rows'];records.append(rec)
    print('C28 SCAN',arm,case,k,'/18 eligible',rec['guard']['accepted'],'max_R',rec['guard']['after']['maximum_R'],flush=True)
    if k%SETTINGS['archive_every_candidates']==0:snapshot(out,a.drive_dir)
   eligible=sorted([r for r in records if r['guard']['accepted']],key=lambda r:(r['guard']['after']['maximum_R'],r['guard']['after_response_score'],r['spec']['id']))
   if eligible:
    best=eligible[0];selected={'accepted':True,**{key:best[key] for key in ('spec','matrix_sha256','matrix_member','matrix_file_sha256','guard')}}
   else:
    zero=torch.zeros_like(writer.matrices[route]).cpu();mh=tensor_hash(zero);mp=routefolder/'matrices'/(mh+'.pt');save_tensor(mp,{'matrix':zero,'parent_C27_matrix_sha256':PARENT_MATRIX_SHA})
    selected={'accepted':False,'spec':None,'matrix_sha256':mh,'matrix_member':mp.relative_to(routefolder).as_posix(),'matrix_file_sha256':file_sha(mp),'reason':'No fixed-grid candidate preserves all19 route passes and improves response score without increasing worst response.'}
   writer.set_route(route,load(routefolder/selected['matrix_member'])['matrix']);dump(selected_file,selected);route_selections[case]=selected;snapshot(out,a.drive_dir)
  actual=measure(writer,data);require(metrics_counts(actual)['passed']==76,'Full76 physical audit failed; no arm committed')
  require(metrics_counts(actual)['maximum_R']<=metrics_counts(baseline)['maximum_R']+1e-12,'Full76 worst response increased')
  for route,case in enumerate(CASES):
   if not route_selections[case]['accepted']:
    replay_equal({sid:baseline[sid] for sid in [t['id'] for t in bycase[case]]},{sid:actual[sid] for sid in [t['id'] for t in bycase[case]]})
  acp={'arm':arm,'matrices':writer.matrices.detach().cpu().clone(),'feature_maps':maps,'training_rows':actual,'route_selections':route_selections,'parent_matrix_sha256':PARENT_MATRIX_SHA,'done':True}
  save_tensor(folder/'FINAL_CHECKPOINT.pt',acp);fr={'arm':arm,'checkpoint_sha256':file_sha(folder/'FINAL_CHECKPOINT.pt'),'matrices_sha256':tensor_hash(writer.matrices),'feature_map_sha256':[file_sha(folder/c/'FEATURE_MAP.pt') for c in CASES],'route_selections':route_selections,'training':metrics_counts(actual)}
  dump(frfile,fr);arm_results[arm]=actual;freeze[arm]=fr;invariant();snapshot(out,a.drive_dir)
 # Freeze both arms before any inspected input tensor is loaded.
 fz=out/'WRITERS_FROZEN.json'
 if fz.exists():require(json.loads(fz.read_text())==freeze,'Arm selection changed after development access')
 else:dump(fz,freeze)
 inspected=[item(t,'inspected') for t in split['prospective_new_states']];require(len(inspected)==16,'Inspected denominator differs')
 dev_rows={name:{} for name in ('c27',)+ARMS}
 for t in inspected:
  ef=out/'evaluation'/t['id']/'COMPLETE.json'
  if ef.exists():
   rec=json.loads(ef.read_text());require(rec['frozen']==freeze,'Cached development writer differs')
   for name,wr in [('c27',base)]+list(writers.items()):require(rec['arms'][name]['metrics']['theta_sha256']==tensor_hash(proposal(wr,t,'resume_evaluation_audit')),'Cached development displacement differs')
  else:
   rec={'id':t['id'],'reference_valid':t['valid'],'frozen':freeze,'arms':{}}
   for name,wr in [('c27',base)]+list(writers.items()):rec['arms'][name]={'reference_valid':t['valid'],'metrics':evaluate(proposal(wr,t,'new_frozen_evaluation'),targets[t['case']],STARTS[t['case']])}
   old=json.loads((out/'data/inspected'/t['id']/'C27_COMPLETE.json').read_text());replay_equal({t['id']:old['arms']['state_bypass']},{t['id']:rec['arms']['c27']});dump(ef,rec)
  for name in dev_rows:dev_rows[name][t['id']]=rec['arms'][name]
  print('C28 INSPECTED',t['id'],{name:complete(dev_rows[name][t['id']]) for name in dev_rows},flush=True)
  if len(dev_rows['c27'])%4==0:snapshot(out,a.drive_dir)
 invariant()
 for arm in ARMS:require(tensor_hash(writers[arm].matrices)==freeze[arm]['matrices_sha256'],'Frozen matrices changed during evaluation')
 training={'c27':metrics_counts(baseline),**{arm:metrics_counts(v) for arm,v in arm_results.items()}};development={name:metrics_counts(v) for name,v in dev_rows.items()}
 decision='C28_INSPECTED_DEVELOPMENT_PASS' if development['response_aware']['passed']==16 else 'C28_INSPECTED_DEVELOPMENT_INCOMPLETE'
 summary={'protocol':PROTOCOL,'complete':True,'decision':decision,'training':training,'inspected_development':development,'by_case':{case:{name:metrics_counts({sid:r for sid,r in rows.items() if sid.startswith(case+'__')}) for name,rows in dev_rows.items()} for case in CASES},
  'frozen_arms':freeze,'parent_C27_matrix_sha256':PARENT_MATRIX_SHA,'parent_C27_unchanged':True,'all76_training_passes_preserved':True,'offline_geometry':{'saved_states':len(geometry),'committed_Jacobian_builds':len(geometry),'committed_backward_calls':sum(g.get('committed_backward_calls',6) for g in geometry.values()),'completed_geometry_seconds':sum(g.get('geometry_seconds',0.) for g in geometry.values()),'counter_checkpoint':counters,'interrupted_uncommitted_work_not_in_completed_counts':True},'proposal_calls_this_process':calls,
  'online_correction_steps':0,'online_jacobian_builds':0,'online_SVD_calls':0,'online_backtracking_steps':0,'teacher_fallback_used':False,'cached_preparation_features_used':True,'requested_output_bit_supplied':True,'external_runtime_commit':True,'fresh_evaluation':False,'new_state_generation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,'LC1_closed':False,'scope':SETTINGS['scope']}
 dump(out/'SUMMARY.json',summary);w.assign_flat(params,torch.zeros(36864,device=device));snapshot(out,a.drive_dir);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
