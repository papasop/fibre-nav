"""Frozen endpoint diagnosis only: no fit, no candidates and no writer selection."""
import argparse,io,json,hashlib,random,sys,time,shutil,os,fcntl
from pathlib import Path
import numpy as np
import torch
import writer_core as w
import run_l5_audit as audit
from raw_writer import StateWriter,prepared_input,route_from_features
from runtime_a100 import a100_device
from c29_protocol import *
from c27_features import ResidualWriter,fit_map,map_features,tensor_hash,state_hash,activation_audit
from c28_features import feature_map,FourRouteWriter
from c29_math import paired,aggregate
from c29_coverage import coverage
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
 if (out/'RUNTIME.json').exists():require(json.loads((out/'RUNTIME.json').read_text())==runtime,'C29 resume runtime differs')
 else:dump(out/'RUNTIME.json',runtime)
 print('C29 Loading pinned GPT-2 once; all C27/C28 writers frozen; diagnostic only.',flush=True)
 cfg=json.loads((ROOT/'inherited/inherited/legacy/config_audit.json').read_text());cfg.update(seed=82931,model_revision=REVISION)
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
 writers={'c27':base};frozen_meta={};arm_cps={}
 for arm in ARMS:
  folder=out/'parent'/arm;fr=json.loads((folder/'FROZEN.json').read_text());cpfile=folder/'FINAL_CHECKPOINT.pt'
  require(all(fr[k]==v for k,v in PINS[arm].items()) and file_sha(cpfile)==PINS[arm]['checkpoint_sha256'],'Actual frozen C28 checkpoint differs')
  acp=load(cpfile);require(acp['arm']==arm and acp['parent_matrix_sha256']==PARENT_MATRIX_SHA,'C28 parent/arm differs')
  maps=[]
  for route,case in enumerate(CASES):
   file=folder/case/'FEATURE_MAP.pt';require(file_sha(file)==fr['feature_map_sha256'][route],'Actual frozen feature file differs');fm=load(file)
   require(fm['fit_ids']==[t['id'] for t in split['training'] if t['case']==case] and not fm['fit_uses_development'],'Frozen feature fit IDs differ');maps.append(fm)
  wr=FourRouteWriter(base,maps).to(device).eval()
  for route in range(4):wr.set_route(route,acp['matrices'][route])
  require(tensor_hash(wr.matrices)==PINS[arm]['matrices_sha256'],'Actual frozen matrix tensor differs')
  writers[arm]=wr;arm_cps[arm]=acp;frozen_meta[arm]={'checkpoint_sha256':file_sha(cpfile),'matrices_sha256':tensor_hash(wr.matrices),'feature_map_sha256':fr['feature_map_sha256']}
 def frozen_check():
  invariant()
  for arm in ARMS:require(tensor_hash(writers[arm].matrices)==PINS[arm]['matrices_sha256'],'C29 changed a frozen writer')
 ffile=out/'FROZEN_IDENTITIES.json'
 if ffile.exists():require(json.loads(ffile.read_text())==frozen_meta,'C29 frozen identities differ')
 else:dump(ffile,frozen_meta)
 def item(t,cohort):
  sid=t['id'];folder=out/'data'/cohort/sid;f=load(folder/'INPUT.pt');prep=vector(load(folder/'prepared.pt'));ref=json.loads((folder/'REFERENCE.json').read_text())
  x,route,inp=validate_saved_input(f,gnorm,gextra,CASES.index(t['case']),device);dump(folder/'C29_INPUT_AUDIT.json',inp)
  return {'id':sid,'case':t['case'],'prep':prep,'x':x,'route':route,'valid':bool(ref['valid']),'cohort':cohort}
 training=[item(t,'training') for t in split['training']];inspected=[item(t,'inspected') for t in split['prospective_new_states']]
 require(len(training)==76 and len(inspected)==16,'Fixed92-state denominator differs')
 oldtrain={'c27':json.loads((out/'parent/C28_BASELINE_REPLAY.json').read_text()),**{arm:arm_cps[arm]['training_rows'] for arm in ARMS}}
 counters={'new_committed_inspected_Jacobians':0,'diagnostic_backward_rows':0,'midpoint_response_probes':0}
 cohorts={'training':training,'inspected':inspected};allrows={};metrics={cohort:{name:{} for name in writers} for cohort in cohorts}
 @torch.no_grad()
 def proposal(wr,t):return t['prep']+(wr(t['x'].unsqueeze(0),t['route'].reshape(1))[0]*SETTINGS['output_scale']).cpu()
 def jacobian(t,theta,e0):
  path=out/'geometry'/(t['id']+'.pt') if t['cohort']=='training' else out/'geometry/inspected'/(t['id']+'.pt')
  if path.exists():g=load(path)
  else:
   require(t['cohort']=='inspected','Missing required archived training geometry')
   sync();started=time.perf_counter();w.assign_flat(params,theta.to(device))
   for p in params:p.requires_grad_(True)
   try:
    with torch.enable_grad():
     response=w.response_vector(model,tok,anchors,responseids,device);rows=[]
     for k in range(6):
      grads=torch.autograd.grad(response[k],params,retain_graph=k<5,allow_unused=False);rows.append(torch.cat([v.reshape(-1) for v in grads]).detach().cpu().double())
     J=torch.stack(rows);rr=(response.detach().cpu()-response0.detach().cpu()).double()
    g={'J':J,'residual':rr,'theta_sha256':tensor_hash(theta),'input_sha256':tensor_hash(t['x']),'C27_matrix_sha256':PARENT_MATRIX_SHA,'diagnostic_only':True,'not_training_or_online_write':True,'committed_backward_rows':6,'diagnostic_seconds':time.perf_counter()-started};save_tensor(path,g)
   finally:
    for p in params:p.requires_grad_(False);p.grad=None
   frozen_check()
  require(g['theta_sha256']==tensor_hash(theta) and g['input_sha256']==tensor_hash(t['x']) and g['C27_matrix_sha256']==PARENT_MATRIX_SHA,'Diagnostic geometry belongs to another state/writer')
  require(g['J'].shape==(6,36864) and bool(torch.isfinite(g['J']).all()),'Diagnostic Jacobian invalid')
  require(np.max(np.abs(g['residual'].double().numpy()-np.asarray(e0)))<=SETTINGS['metric_comparison_tolerance'],'Baseline geometry residual differs')
  return g
 for cohort,items in cohorts.items():
  cohort_rows={}
  for index,t in enumerate(items,1):
   sid=t['id'];dest=out/'diagnosis'/cohort/sid/'COMPLETE.json'
   theta={name:proposal(wr,t) for name,wr in writers.items()};sha={name:tensor_hash(v) for name,v in theta.items()}
   points_file=dest.parent/'DIAGNOSTIC_POINTS.pt'
   points={'endpoints':theta,'midpoints':{}}
   if cohort=='inspected':
    points['midpoints']={arm:(theta['c27'].double()+SETTINGS['diagnostic_half_step']*(theta[arm].double()-theta['c27'].double())).float() for arm in ARMS}
   if points_file.exists():
    saved_points=load(points_file)
    for group in points:
     require(set(saved_points[group])==set(points[group]) and all(torch.equal(saved_points[group][name],v) for name,v in points[group].items()),'Saved diagnostic parameter point differs')
   else:save_tensor(points_file,points)
   if dest.exists():
    rec=json.loads(dest.read_text());require(rec['theta_sha256']==sha and rec['frozen']==frozen_meta and rec['diagnostic_points_sha256']==file_sha(points_file),'Cached diagnosis belongs to other writers/states')
   else:
    endpoint={name:{'reference_valid':t['valid'],'metrics':evaluate(v,targets[t['case']],STARTS[t['case']])} for name,v in theta.items()}
    old=oldtrain if cohort=='training' else None
    if cohort=='inspected':old=json.loads((out/'data/inspected'/sid/'C28_COMPLETE.json').read_text())['arms']
    for name in writers:
     before=old[name][sid] if cohort=='training' else old[name];replay_equal({sid:before},{sid:endpoint[name]})
    e0=endpoint['c27']['metrics']['response_residual'];g=jacobian(t,theta['c27'],e0);J=g['J'].numpy();paired_results={}
    for arm in ARMS:
     delta=(theta[arm].double()-theta['c27'].double()).numpy();half=None
     if cohort=='inspected':
      halfway=points['midpoints'][arm]
      w.assign_flat(params,halfway.to(device))
      with torch.no_grad():hr=(w.response_vector(model,tok,anchors,responseids,device)-response0).cpu().double().numpy()
      half={'residual':hr,'delta':(halfway.double()-theta['c27'].double()).numpy()}
     paired_results[arm]=paired(e0,endpoint[arm]['metrics']['response_residual'],J,delta,half,SETTINGS['response_budget'],SETTINGS['diagnostic_change_tolerance'])
    rec={'id':sid,'case':t['case'],'cohort':cohort,'reference_valid':t['valid'],'theta_sha256':sha,'diagnostic_points_sha256':file_sha(points_file),'frozen':frozen_meta,'endpoints':endpoint,'paired':paired_results,'new_writer_committed':False,'midpoint_not_an_accepted_write':True};dump(dest,rec)
   cohort_rows[sid]=rec
   for name in writers:metrics[cohort][name][sid]=rec['endpoints'][name]
   print('C29 DIAGNOSIS',cohort,index,'/',len(items),sid,{arm:rec['paired'][arm]['diagnostic_category'] for arm in ARMS},flush=True)
   if index%SETTINGS['archive_every_completed_states']==0:snapshot(out,a.drive_dir)
  allrows[cohort]=cohort_rows
 # Pure feature diagnostics on the immutable original maps; no fitting occurs.
 feature_rows=[]
 for arm in ARMS:
  for case in CASES:
   tr=[t for t in training if t['case']==case];dv=[t for t in inspected if t['case']==case];fm=load(out/'parent'/arm/case/'FEATURE_MAP.pt')
   tx=torch.stack([t['x'].cpu() for t in tr]).double().numpy();dx=torch.stack([t['x'].cpu() for t in dv]).double().numpy();cc=coverage(tx,dx,fm)
   for t,c in zip(dv,cc):feature_rows.append({'id':t['id'],'arm':arm,'case':case,**c,'paired':allrows['inspected'][t['id']]['paired'][arm]})
 dump(out/'FEATURE_COVERAGE.json',feature_rows)
 new_geom=list((out/'geometry/inspected').glob('*.pt'));counters['new_committed_inspected_Jacobians']=len(new_geom);counters['diagnostic_backward_rows']=6*len(new_geom);counters['midpoint_response_probes']=2*len(inspected)
 grouped={cohort:{arm:aggregate([r['paired'][arm] for r in rr.values()]) for arm in ARMS} for cohort,rr in allrows.items()}
 summary={'protocol':PROTOCOL,'complete':True,'decision':'C29_FROZEN_DIRECTION_DIAGNOSIS_COMPLETE','endpoint_replay':{c:{name:metrics_counts(v) for name,v in ms.items()} for c,ms in metrics.items()},'direction_diagnosis':grouped,
  'by_case':{case:{arm:aggregate([r['paired'][arm] for r in allrows['inspected'].values() if r['case']==case]) for arm in ARMS} for case in CASES},
  'frozen_identities':frozen_meta,'all_parent_endpoint_records_replayed':True,'all_writers_unchanged':True,'parent_archives_modified':False,'training_Jacobians_reused':76,'committed_diagnostic_work':counters,
  'new_head_training_steps':0,'new_candidates':0,'new_heads_committed':0,'new_writing_method':False,'reference_validity_reused_from_parent':True,
  'fresh_evaluation':False,'new_state_generation':False,'internal_parameter_update':False,'LC1_closed':False,'scope':SETTINGS['scope'],'interpretation':'Jacobian/Taylor discrepancies include nonlinear response, finite-precision and derivative evaluation error. No global curvature certificate or exclusive causal attribution.'}
 frozen_check();dump(out/'SUMMARY.json',summary);dump(out/'DIRECTION_ROWS.json',allrows);w.assign_flat(params,torch.zeros(36864,device=device));snapshot(out,a.drive_dir);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
