"""One numerical offline fit per arm, protected true forwards, frozen development evaluation."""
import argparse,io,json,hashlib,random,sys,time,shutil,os,fcntl
from pathlib import Path
import numpy as np
import torch
import writer_core as w
import run_l5_audit as audit
from raw_writer import StateWriter,prepared_input,route_from_features
from runtime_a100 import a100_device
from c27_protocol import *
from c27_features import ResidualWriter,fit_map,map_features,tensor_hash,state_hash,activation_audit
from c27_math import residual_candidates,trial_matrix
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
 check_legacy(ROOT)
 if (out/'SUMMARY.json').exists():print((out/'SUMMARY.json').read_text(),flush=True);snapshot(out,a.drive_dir);return
 identity=json.loads((out/'identity.json').read_text());require(identity['settings']==SETTINGS and identity['source_hashes']==source_hashes(ROOT),'Runtime source mismatch')
 provenance=json.loads((out/'PARENT_PROVENANCE.json').read_text())
 for n,r in provenance['members'].items():require(file_sha(out/n)==r['sha256'],'Compact original member changed: '+n)
 device,gpu=a100_device(torch);torch.set_num_threads(min(4,os.cpu_count() or 1));w.set_seed(82931)
 import transformers,peft
 runtime={'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':gpu}
 if (out/'RUNTIME.json').exists():require(json.loads((out/'RUNTIME.json').read_text())==runtime,'C27 resume runtime differs')
 else:dump(out/'RUNTIME.json',runtime)
 print('C27 Loading pinned GPT-2 once; immutable C25 writer + two numerical residual branches.',flush=True)
 cfg=json.loads((ROOT/'legacy/config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
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
 base=StateWriter().to(device);base.load_state_dict(cp['head']);base.eval()
 for p in base.parameters():p.requires_grad_(False);p.grad=None
 base_versions={n:p._version for n,p in base.named_parameters()}
 norm=load(out/'parent/C9_CHECKPOINT.pt')['normalisation'];extra_norm=cp['extra_norm']
 gnorm={k:v.to(device) for k,v in norm.items()};gextra={k:v.to(device) for k,v in extra_norm.items()}
 require(float(norm['output_scale'])==SETTINGS['output_scale'],'Output scale changed')
 split=json.loads((out/'parent/SPLIT.json').read_text());train_ids=[t['id'] for t in split['training']];original=[t['id'] for t in split['original60']]
 parent_final=json.loads((out/'parent/FINAL_TRAINING.json').read_text());targets={c:STARTS[c][:3]+[STARTS[c][0]^STARTS[c][1]] for c in CASES}
 require(len(train_ids)==76 and len(set(train_ids))==76 and len(original)==60,'Cohort changed')
 def invariant():
  require(all(p._version==base_versions[n] for n,p in base.named_parameters()),'Original C25 writer changed')
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
  auditfile=folder/'C27_INPUT_SCHEMA_AUDIT.json'
  if auditfile.exists():require(json.loads(auditfile.read_text())==input_audit,'Saved input audit differs')
  else:dump(auditfile,input_audit)
  return {'id':sid,'case':t['case'],'prep':prep,'x':x,'route':route,'valid':bool(ref['valid'])}
 data=[item(t,'training') for t in split['training']];byid={t['id']:t for t in data};route_data=[t for t in data if t['case']==SETTINGS['route']];route_ids=[t['id'] for t in route_data]
 require(len(route_data)==19 and all(t['valid'] for t in data),'Original training references invalid')
 proposal_counts={'offline_or_audit':0,'new_frozen_evaluation':0,'resume_evaluation_audit':0}
 @torch.no_grad()
 def proposal(writer,t,purpose='offline_or_audit'):
  # One combined module call; supplied cached preparation features, no online geometry or retries.
  proposal_counts[purpose]+=1
  delta=writer(t['x'].unsqueeze(0),t['route'].reshape(1))[0]*SETTINGS['output_scale'];return t['prep']+delta.cpu()
 def measure(writer,items,cache=None):
  rows={}
  for i,t in enumerate(items,1):
   theta=proposal(writer,t)
   if cache is not None and t['id'] in cache:
    r=cache[t['id']];require(r['metrics']['theta_sha256']==tensor_hash(theta),'Cached physical theta differs');rows[t['id']]=r
   else:rows[t['id']]={'reference_valid':t['valid'],'metrics':evaluate(theta,targets[t['case']],STARTS[t['case']])}
   if i%19==0:print('C27 ACTUAL GATE',i,'/',len(items),flush=True)
  return rows
 baselinefile=out/'BASELINE_REPLAY.json'
 if baselinefile.exists():baseline=json.loads(baselinefile.read_text());measure(base,data,baseline)
 else:
  baseline=measure(base,data);replay_equal(parent_final,baseline);dump(baselinefile,baseline);snapshot(out,a.drive_dir)
 require(len(quality(baseline,train_ids,original)['passed_ids'])==74,'C25 baseline74 did not replay')
 auditfile=out/'TRAINING_ACTIVATION_AUDIT.json'
 xby={c:torch.stack([t['x'] for t in data if t['case']==c]) for c in CASES};idby={c:[t['id'] for t in data if t['case']==c] for c in CASES}
 if not auditfile.exists():dump(auditfile,activation_audit(base,xby,idby))
 xroute=torch.stack([t['x'] for t in route_data])
 # Reproduce single-input batch geometry features exactly; no batched-head rounding assumptions.
 with torch.no_grad():hroute=torch.stack([base.experts[2][1](base.experts[2][0](t['x'])) for t in route_data])
 J=[];E=[]
 for t,h in zip(route_data,hroute):
  r=load(out/'geometry'/(t['id']+'.pt'));theta=proposal(base,t)
  require(r['head_sha256']==HEAD_SHA and torch.equal(theta,r['theta']) and r['input_sha256']==tensor_hash(t['x']),'Saved Jacobian state/head/input differs')
  require(float((torch.cat([h.cpu(),torch.ones(1)])-r['H']).abs().max())<=2e-6,'Saved hidden feature replay differs')
  require(np.max(np.abs(r['residual'].double().numpy()-np.array(baseline[t['id']]['metrics']['response_residual'])))<=2e-6,'Saved residual replay differs')
  J.append(r['J'].double().numpy());E.append(r['residual'].double().numpy())
 J=np.stack(J);E=np.stack(E)
 arm_results={};writers={}
 for arm in ARMS:
  folder=out/'arms'/arm;folder.mkdir(parents=True,exist_ok=True);feature_file=folder/'FEATURE_MAP.pt'
  if feature_file.exists():fm=load(feature_file)
  else:
   values=hroute.cpu().double().numpy() if arm=='hidden_control' else xroute.cpu().double().numpy()
   fm=fit_map(values,route_ids,arm,SETTINGS['feature_components'],SETTINGS['feature_scale_floor']);save_tensor(feature_file,fm)
  require(fm['kind']==arm and fm['fit_ids']==route_ids and not fm['fit_uses_development'],'Feature fit IDs/scope changed')
  writer=ResidualWriter(base,fm,arm).to(device).eval();writers[arm]=writer
  with torch.no_grad():Phi=torch.stack([writer.features(t['x'].unsqueeze(0))[0].cpu() for t in route_data]).double().numpy()
  frozenfile=folder/'FROZEN.json'
  if frozenfile.exists():
   fr=json.loads(frozenfile.read_text());cpfile=folder/'FINAL_CHECKPOINT.pt';require(file_sha(cpfile)==fr['checkpoint_sha256'],'Frozen branch checkpoint digest differs')
   bcp=load(cpfile);require(bcp['arm']==arm and bcp['parent_head_sha256']==HEAD_SHA and fr['feature_map_sha256']==file_sha(feature_file),'Frozen arm/base/feature identity differs')
   writer.set_matrix(bcp['residual_matrix']);require(tensor_hash(writer.matrix)==fr['matrix_sha256'],'Frozen matrix differs')
   require(guard(baseline,bcp['training_rows'],train_ids,original)['accepted'] if bcp['accepted_refinement'] else bcp['training_rows']==baseline,'Frozen training rows/guard differ')
   measure(writer,data,bcp['training_rows']);arm_results[arm]=bcp['training_rows'];continue
  cache_solver=folder/'SAVED_DIRECTIONS.pt'
  if cache_solver.exists():
   saved=load(cache_solver);require(saved['Phi_sha256']==tensor_hash(torch.from_numpy(Phi)) and saved['fit_ids']==route_ids,'Saved direction feature/ID mismatch')
   from joint_calibration import prepare_system
   system=prepare_system(J,Phi,E,SETTINGS['output_scale'],1e-12,1e-8);bases=[x.numpy() for x in saved['bases']];infos=saved['infos']
  else:
   system,bases,infos=residual_candidates(J,Phi,E)
   save_tensor(cache_solver,{'bases':[torch.from_numpy(x) for x in bases],'infos':infos,'Phi_sha256':tensor_hash(torch.from_numpy(Phi)),'fit_ids':route_ids})
  dump(folder/'SOLVER.json',infos);records=[];seen={}
  for index,spec in enumerate(specs(),1):
   cf=folder/'candidates'/spec['id'];mf=cf/'MATRIX.pt';dest=cf/'COMPLETE.json'
   if mf.exists():
    mc=load(mf);require(mc['spec']==spec and mc['arm']==arm and mc['parent_head_sha256']==HEAD_SHA,'Candidate matrix checkpoint identity differs')
    matrix=mc['matrix'];linear=mc['linear']
   else:
    matrix,linear=trial_matrix(system,bases,spec)
    save_tensor(mf,{'matrix':matrix,'linear':linear,'spec':spec,'arm':arm,'parent_head_sha256':HEAD_SHA})
   mh=tensor_hash(matrix)
   try:
    writer.set_matrix(matrix)
    if dest.exists():
     rec=json.loads(dest.read_text());require(rec['spec']==spec and rec['matrix_sha256']==mh,'Candidate resume identity differs');measure(writer,route_data,rec['route_rows'])
    else:
     alias=seen.get(mh);rr=measure(writer,route_data,alias['route_rows'] if alias else None)
     merged={**baseline,**rr};g=guard(baseline,merged,train_ids,original)
     actual_displacements=np.stack([(proposal(writer,t).double()-load(out/'geometry'/(t['id']+'.pt'))['theta'].double()).numpy() for t in route_data])
     predicted=E+np.einsum('npd,nd->np',J,actual_displacements)
     measured_residuals=np.asarray([rr[t['id']]['metrics']['response_residual'] for t in route_data])
     prediction={'actual_memory_step_norm_max':float(np.max(np.linalg.norm(actual_displacements,axis=1))),
                 'actual_displacement_linear_R':float(np.max(np.abs(predicted))),
                 'actual_route_R':max(rr[t['id']]['metrics']['drift'] for t in route_data),
                 'linear_prediction_error_max':float(np.max(np.abs(measured_residuals-predicted))),
                 'actual_matrix_Frobenius_norm':float(matrix.double().norm())}
     rec={'prediction':prediction,'spec':spec,'matrix_sha256':mh,'route_rows':rr,'guard':g,'alias_of':alias['spec']['id'] if alias else None,'linear':linear,'full76_verified':False}
     dump(dest,rec)
    require(rec['guard']==guard(baseline,{**baseline,**rec['route_rows']},train_ids,original),'Candidate guard changed')
    records.append(rec);seen.setdefault(mh,rec)
    print('C27 SCAN',arm,index,'/18 eligible',rec['guard']['accepted'],'pass',len(rec['guard']['after']['passed_ids']),flush=True)
   finally:writer.set_matrix(torch.zeros_like(writer.matrix))
   if index%SETTINGS['archive_every_candidates']==0:snapshot(out,a.drive_dir)
  eligible=sorted([r for r in records if r['guard']['accepted']],key=lambda r:(-len(r['guard']['after']['passed_ids']),r['guard']['after']['maximum_drift'],r['guard']['after']['violation_score'],r['spec']['id']))
  winner=None;audited=set()
  for r in eligible:
   if r['matrix_sha256'] in audited:continue
   audited.add(r['matrix_sha256']);matrix=load(folder/'candidates'/r['spec']['id']/'MATRIX.pt')['matrix'];require(tensor_hash(matrix)==r['matrix_sha256'],'Actual candidate matrix checkpoint differs');writer.set_matrix(matrix)
   af=folder/'audits'/r['spec']['id']/'AUDIT.json'
   if af.exists():
    av=json.loads(af.read_text());require(av['matrix_sha256']==r['matrix_sha256'],'Audit matrix differs');actual=av['rows'];measure(writer,data,actual)
   else:
    actual=measure(writer,data);replay_equal({**baseline,**r['route_rows']},actual);av={'matrix_sha256':r['matrix_sha256'],'rows':actual,'guard':guard(baseline,actual,train_ids,original)};dump(af,av)
   require(av['guard']==guard(baseline,actual,train_ids,original),'Full76 audit guard mismatch')
   if av['guard']['accepted']:winner=(r,matrix,actual);break
  if winner is None:writer.set_matrix(torch.zeros_like(writer.matrix));selected=None;actual=baseline;accepted=False
  else:r,matrix,actual=winner;writer.set_matrix(matrix);selected=r['spec'];accepted=True
  bcp={'residual_matrix':writer.matrix.detach().cpu().clone(),'feature_map':fm,'training_rows':actual,'selected_spec':selected,
       'parent_head_sha256':HEAD_SHA,'arm':arm,'done':True,'accepted_refinement':accepted,'original_base_head_frozen':True}
  save_tensor(folder/'FINAL_CHECKPOINT.pt',bcp);fr={'arm':arm,'accepted_refinement':accepted,'selected_spec':selected,'matrix_sha256':tensor_hash(writer.matrix),
      'checkpoint_sha256':file_sha(folder/'FINAL_CHECKPOINT.pt'),'feature_map_sha256':file_sha(feature_file),'training_quality':quality(actual,train_ids,original),'n_specs':18,'n_unique_matrices':len(seen),'new_backward_calls':0,'saved_geometry_used':True}
  dump(frozenfile,fr);arm_results[arm]=actual;invariant();snapshot(out,a.drive_dir)
 # Both selections are immutable BEFORE any inspected-development tensor is loaded.
 freeze={arm:json.loads((out/'arms'/arm/'FROZEN.json').read_text()) for arm in ARMS}
 if (out/'WRITERS_FROZEN.json').exists():require(json.loads((out/'WRITERS_FROZEN.json').read_text())==freeze,'Final branch selection changed after development access')
 else:dump(out/'WRITERS_FROZEN.json',freeze)
 invariant();require(state_hash(base.state_dict())==HEAD_SHA,'C25 base head identity changed')
 inspected=[item(t,'inspected') for t in split['prospective_new_states']]
 require(len(inspected)==16,'All16 inspected states required')
 dx={c:torch.stack([t['x'] for t in inspected if t['case']==c]) for c in CASES};di={c:[t['id'] for t in inspected if t['case']==c] for c in CASES}
 dump(out/'INSPECTED_ACTIVATION_AUDIT.json',activation_audit(base,dx,di))
 dev_rows={name:{} for name in ('c25',)+ARMS}
 for t in inspected:
  ef=out/'evaluation'/t['id']/'COMPLETE.json'
  if ef.exists():
   r=json.loads(ef.read_text());require(r['frozen']==freeze,'Development result belongs to different branches')
   for name,writer in [('c25',base)]+[(arm,writers[arm]) for arm in ARMS]:require(r['arms'][name]['metrics']['theta_sha256']==tensor_hash(proposal(writer,t,'resume_evaluation_audit')),'Cached development theta differs')
  else:
   r={'id':t['id'],'frozen':freeze,'reference_valid':t['valid'],'arms':{}}
   for name,writer in [('c25',base)]+[(arm,writers[arm]) for arm in ARMS]:
    theta=proposal(writer,t,'new_frozen_evaluation');r['arms'][name]={'reference_valid':t['valid'],'metrics':evaluate(theta,targets[t['case']],STARTS[t['case']])}
   old=json.loads((out/'data/inspected'/t['id']/'C25_COMPLETE.json').read_text())
   replay_equal({t['id']:{'reference_valid':old['reference_valid'],'metrics':old['arms']['calibrated']}},{t['id']:r['arms']['c25']})
   dump(ef,r)
  for name in dev_rows:dev_rows[name][t['id']]=r['arms'][name]
  print('C27 INSPECTED',t['id'],{name:complete(dev_rows[name][t['id']]) for name in dev_rows},flush=True)
  if sum(len(x) for x in dev_rows.values())%12==0:snapshot(out,a.drive_dir)
 invariant();require(state_hash(base.state_dict())==HEAD_SHA,'Original C25 writer changed after evaluation')
 for arm in ARMS:require(tensor_hash(writers[arm].matrix)==freeze[arm]['matrix_sha256'],'Frozen residual branch changed')
 training={'c25':metrics_counts(baseline),**{arm:metrics_counts(arm_results[arm]) for arm in ARMS}};development={name:metrics_counts(rows) for name,rows in dev_rows.items()}
 summary={'protocol':PROTOCOL,'complete':True,'decision':decide(training,development),'training':training,'inspected_development':development,
  'training_by_case':{c:{name:metrics_counts({s:r for s,r in rows.items() if s.startswith(c+'__')}) for name,rows in {'c25':baseline,**arm_results}.items()} for c in CASES},
  'proposal_calls_this_process_by_purpose':proposal_counts,'cached_record_replay_is_audit_not_new_online_write':True,
  'training_activation_summary':{c:{k:v for k,v in r.items() if k in ('z_abs_maximum','hidden_abs_above_0999_fraction','tanh_derivative_maximum','hidden_pairwise_maximum_L2')} for c,r in json.loads(auditfile.read_text()).items()},
  'n_candidates_per_arm':18,'n_offline_rounds_per_arm':1,'base_head_sha256':HEAD_SHA,'base_head_unchanged':True,'all_initial74_passes_preserved':all(set(quality(baseline,train_ids,original)['passed_ids']).issubset(quality(rows,train_ids,original)['passed_ids']) for rows in arm_results.values()),
  'arms':freeze,'gpu':gpu,'new_model_backward_calls':0,'new_model_Jacobians':0,'saved_Jacobian_states':19,
  'online_proposals_per_completed_arm':1,'online_correction_steps':0,'online_SVD_calls':0,'online_backward_calls':0,'teacher_fallback_used':False,
  'cached_preparation_features_used':True,'external_runtime_commit':True,'requested_output_bit_supplied':True,'fresh_evaluation':False,'new_state_generation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,'scope':SETTINGS['scope']}
 dump(out/'SUMMARY.json',summary);w.assign_flat(params,torch.zeros(36864,device=device));snapshot(out,a.drive_dir);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
