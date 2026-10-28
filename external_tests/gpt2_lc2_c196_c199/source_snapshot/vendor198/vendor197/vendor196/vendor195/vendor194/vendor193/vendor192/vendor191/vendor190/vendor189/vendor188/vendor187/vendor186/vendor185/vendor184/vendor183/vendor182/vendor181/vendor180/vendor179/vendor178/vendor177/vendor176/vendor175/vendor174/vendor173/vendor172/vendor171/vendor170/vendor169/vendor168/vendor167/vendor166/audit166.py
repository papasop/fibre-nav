"""C165 archive, actual point/head and complete observed arithmetic reconstruction."""
from core166 import *
from audit147 import source_check
from audit164 import seal,actual_batch
from baseline166 import baseline_audit

def finite_diagnostics(anchor,repeat,probes,d,ds,native):
 fd=[];repeats=[];nrepeat=[];summaries={};paired={}
 for a,b in zip(anchor,repeat):
  compare(a['state_id'],b['state_id']);compare(a['FP64_point_sha256'],b['FP64_point_sha256']);err=float(np.max(abs(np.asarray(a['observed'])-b['observed'])));kl=abs(a['KL']-b['KL']);need(err<=2e-9 and kl<=2e-9,'Original same-run FP64 anchor repeat');repeats.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=kl,within_original_repeat_tolerance=True))
 for label,h in zip(('1e-6','5e-6','1e-5'),H):
  minus=probes['minus_'+label];plus=probes['plus_'+label]
  for i,sid in enumerate(IDS):
   parts={}
   for name,keys,gradient in [('response',('response_loss',),'response_gradient'),('guard',PARTS[1:],'guard_gradient')]:
    lo=sum(minus[i]['objective_parts'][k] for k in keys);hi=sum(plus[i]['objective_parts'][k] for k in keys);ad=float(np.sum(np.asarray(anchor[i][gradient]).ravel()*d.ravel()));central=(hi-lo)/(2.*h);parts[name]=dict(AD_directional=ad,central_difference=central,difference=central-ad,absolute_difference=abs(central-ad),no_derivative_error_or_curvature_cause_claim=True)
   fd.append(dict(state_id=sid,h=h,checks=parts))
 for arm in ['anchor']+[a['name'] for a in ARMS]:
  summaries[arm]={str(t):parent.parent.native_summary(native[(arm,t)]) for t in (0,1)}
  for a,b in zip(native[(arm,0)],native[(arm,1)]):
   r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'All original native repeat checks');nrepeat.append(dict(arm=arm,state_id=a['state_id'],**r))
  if arm!='anchor':paired[arm]={str(t):parent.parent.paired(native[('anchor',t)],native[(arm,t)]) for t in (0,1)}
 diag=dict(native_summaries=summaries,paired_vs_anchor=paired,all84_all6_arms_retained=True,computed_strict_common_response_descent=ds['computed_strict_common_response_descent'],direction_zero=ds['direction_zero'],all6_FP64_maximum_response={name:max(float(max(abs(np.asarray(r['observed'])[:6]))) for r in rr) for name,rr in probes.items()},any_probe_pass_is_training_diagnostic_only=True,step_or_head_selected=False,LC2_closed=False)
 return diag,fd,repeats,nrepeat

def audit(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with archive155.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source165'],'C165');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['decision'],'C165_FIXED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['scope'],parent.design()['scope']);compare(s['stage'],'complete')
  for n,v in dict(new_training_steps=0,new_head_fits=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,temporary_probe_heads=6,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_model_forwards=3699,functional_FP64_forwards=1346,new_backward_calls=168,scalar_backward_attempts=168,records_completed=1848,requests_consumed=1848,actual_FP64_point_tensors_saved=672).items():compare(s[n],v,'Original denominator/'+n)
  for n in ('candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decisions_reclassified','archived_source_executed','chronology_or_nonaccess_proved','preparation_replay_performed','FP64_acceptance_evaluated','LC2_closed','LC2_single_fold_passed'):compare(s[n],False)
  for n in ('direction_computed_from_training_measurements','all_requested_states_and_arms_retained','all6_probes_sealed_before_first_probe','actual_FP64_points_checked_before_model_forward','initial_native_and_FP64_replay_matched_C163','native_repeats_consistent','FP64_anchor_repeat_consistent'):compare(s[n],True)
  compare(s['C164_parent_decision'],ctx['C164_receipt']['decision']);compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in parent.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'Original link/'+k)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design());compare(z.json('ALL1848_REQUESTS_FROZEN.json'),parent.requests());chain=parent.parent.parent.previous.constants(ctx['data']);actual=z.arrays('FROZEN_FULL84_CHAIN_CONSTANTS.npz');compare(set(actual),set(chain))
  for n,v in chain.items():exact(actual[n],v,'Original full84 constants/'+n)
  p=ctx['initial_head'].copy();head=z.arrays('ANCHOR_C163_FIXED_FINAL_HEAD.npz');compare(set(head),{'P'});exact(head['P'],p,'C163 fixed final head identity');compare(s['initial_C163_fixed_final_head_digest'],parameter_hash(p))
  if c['production']:compare(p.shape,(3,2,36))
  before=z.json('SEALED_BEFORE_MODEL.json');compare(before['model_forwards_so_far'],0);compare(set(before['artifact_seals']),{'DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','ANCHOR_C163_FIXED_FINAL_HEAD.npz','ALL1848_REQUESTS_FROZEN.json'})
  for n,h in before['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h)
  base=baseline_audit(z,ctx,out/'baseline');rows=[clean_json(line) for line in z.raw('ALL1848_RECORDS.jsonl').splitlines()];compare([{k:r[k] for k in ('stage','arm','state_id')} for r in rows],parent.requests());groups={}
  for r in rows:compare(r['case'],r['state_id'].split('__')[0]);groups.setdefault((r['stage'],r['arm']),[]).append(r)
  anchor_points,anchor_native=seal(z,'anchor',chain,p,'C163_fixed_final_anchor',True);seals=[];point_rows=[]
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(anchor_native[sid]),ctx['parent_final_native_records']['0'][i]['metrics']['theta_sha256'],'C163 final/native anchor exact')
  def continuous(rr,head,points,folder,tag,F,B,grad):
   compare([r['state_id'] for r in rr],IDS);pts,receipt=actual_batch(z,folder,head,points,tag,F,B,budget);seals.append(receipt)
   for r in rr:
    budget();sid=r['state_id'];compare(r['parameters_digest'],parameter_hash(head));compare(r['sealed_numpy_point_sha256'],point_hash(points[sid]));compare(r['FP64_point_sha256'],point_hash(pts[sid]));compare(r['actual_FP64_point_artifact'],folder+'/ACTUAL_TORCH_FP64_POINTS.npz');err=float(max(abs(pts[sid]-points[sid])));compare(r['map_Linf'],err,atol=1e-18,rtol=1e-12);compare(r['FP64_acceptance_evaluated'],False);_,e=mapping(chain,head,sid);expected=losses(r['observed'],float(r['KL']),pts[sid],chain[sid+'__prepared'],e,r['case']);compare(r['objective_parts'],expected,'Independent full loss',atol=1e-10,rtol=1e-9)
    if grad:
     for key in ('response_gradient','guard_gradient'):farray(np.asarray(r[key],float),p.shape)
    point_rows.append(dict(folder=folder,state_id=sid,head_sha256=parameter_hash(head),actual_FP64_point_sha256=point_hash(pts[sid]),NumPy_point_sha256=point_hash(points[sid]),recomputed_map_Linf=err))
  gradient=groups[('anchor_gradient','anchor')];continuous(gradient,p,anchor_points,'anchor','anchor_gradient',3,0,True)
  for i,r in enumerate(gradient):
   old=ctx['parent_final_FP64_records'][i];re=dict(observed_Linf=float(max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;compare(r['C163_final_FP64_replay'],re);need(re['within_original_replay_tolerance'],'Original C163 FP64 replay')
  saved=z.arrays('ONE_FIXED_DIRECTION.npz');compare(set(saved),{'direction','anchor'});exact(saved['anchor'],p,'Direction file anchored to exact C163 final head');d=saved['direction'];hist=z.json('ALL4096_DIRECTION_ITERATIONS.json');ds,dd,trace,contractions=replay_direction(gradient,p,d,hist);compare(z.json('DIRECTION_DIAGNOSTICS.json'),ds,'Complete original direction summary');compare(s['offline_direction_optimization_iterations'],len(hist));compare(s['computed_strict_common_response_descent'],ds['computed_strict_common_response_descent']);compare(s['direction_zero'],ds['direction_zero'])
  heads={};phys={'anchor':anchor_native};head_seals={};probes={}
  for arm in ARMS:
   name=arm['name'];q=p+float(arm['alpha'])*d;need(np.max(abs(q))<=.125,'Unclipped original symmetric head box');heads[name]=q;pts,nt=seal(z,name,chain,q,name,True);phys[name]=nt;head_seals[name]=z.json(name+'/SEALED.json');rr=groups[('FP64_probe',name)];continuous(rr,q,pts,name,name,339,168,False);probes[name]=rr
  repeat_points,_=seal(z,'anchor_repeat',chain,p,'anchor_repeat',False);repeat=groups[('FP64_anchor_repeat','anchor')];continuous(repeat,p,repeat_points,'anchor_repeat','anchor_repeat',339,168,False);compare(z.json('ALL8_ACTUAL_FP64_POINT_SEALS.json'),seals);compare(len(point_rows),672)
  all6=z.json('ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json');compare(all6['model_forwards_so_far'],339);compare(all6['backward_calls_so_far'],168);compare(all6['all6_head_digests'],{n:parameter_hash(q) for n,q in heads.items()});compare(all6['head_seals'],head_seals);compare(all6['actual_point_seals'],seals);compare(all6['all6_before_first_probe'],True);compare(set(all6['direction_artifact_seals']),{'ONE_FIXED_DIRECTION.npz','DIRECTION_DIAGNOSTICS.json','ALL4096_DIRECTION_ITERATIONS.json'})
  for n,h in all6['direction_artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h)
  native={}
  for arm in ['anchor']+[r['name'] for r in ARMS]:
   for rnd in (0,1):
    stage=('native_anchor' if arm=='anchor' else 'native_probe')+str(rnd);rr=groups[(stage,arm)];compare([r['state_id'] for r in rr],IDS)
    for i,r in enumerate(rr):
     budget();sid=r['state_id'];pt=phys[arm][sid];met=archive155.full_metric(r['metrics']['observed'],r['metrics']['KL'],pt,r['case']);qual=archive155.endpoint_gates(met,r['case'],pt,chain[sid+'__prepared'].astype(np.float32),True);compare(r['metrics'],met,'Independent native metrics');compare(r['qualification'],qual,'All original native gates')
     if c['production']:compare(r['actual_physical_FP32_commit'],True)
     if arm=='anchor':
      re=archive155.repeat_check(ctx['parent_final_native_records'][str(rnd)][i]['metrics'],met);compare(r['C163_final_native_replay'],re);need(re['within_replay_tolerance'] and re['same_point_identity'],'Original native C163 replay')
    native[(arm,rnd)]=rr
  diag,fd,fr,nr=finite_diagnostics(gradient,repeat,probes,d,ds,native);compare(z.json('DIAGNOSTIC_SUMMARY.json'),diag);compare(s['diagnostics'],diag);compare(z.json('ALL252_CENTRAL_DIFFERENCE_CHECKS.json'),fd);compare(z.json('ALL84_FP64_ANCHOR_REPEATS.json'),fr);compare(z.json('ALL588_NATIVE_REPEAT_CHECKS.json'),nr);compare(z.json('ALL1176_NATIVE_ENDPOINTS.json'),{name+'__round'+str(t):[{k:v for k,v in r.items() if k not in ('stage','arm')} for r in rr] for (name,t),rr in native.items()});compare(z.json('ALL6_FP64_PROBE_RECORDS.json'),{name:[{k:v for k,v in r.items() if k not in ('stage','arm')} for r in rr] for name,rr in probes.items()})
  dump(out/'ALL672_ACTUAL_FP64_POINT_AUDIT_ROWS.json',point_rows);dump(out/'ALL8_ACTUAL_POINT_SEALS_AUDITED.json',seals);dump(out/'ALL4096_FW_ITERATION_AUDIT.json',trace);dump(out/'ALL84_DIRECTION_CONTRACTIONS.json',contractions);dump(out/'INDEPENDENT_DIRECTION_SUMMARY.json',ds);dump(out/'DIRECTION_NUMERIC_DIAGNOSTICS.json',dd);dump(out/'ALL252_SCALAR_CENTRAL_DIFFERENCES.json',fd);dump(out/'ALL84_FP64_REPEATS.json',fr);dump(out/'ALL588_NATIVE_REPEATS.json',nr);dump(out/'PARENT_DIAGNOSTICS_RECOMPUTED.json',diag)
  return dict(summary=s,members_hashed=members,diagnostics=diag,direction=dd,contractions=contractions,trace=trace,fd=fd,gradient_records=gradient,repeat_records=repeat,probe_records=probes,native_records=native,maximum_map_error=max(r['recomputed_map_Linf'] for r in point_rows))
