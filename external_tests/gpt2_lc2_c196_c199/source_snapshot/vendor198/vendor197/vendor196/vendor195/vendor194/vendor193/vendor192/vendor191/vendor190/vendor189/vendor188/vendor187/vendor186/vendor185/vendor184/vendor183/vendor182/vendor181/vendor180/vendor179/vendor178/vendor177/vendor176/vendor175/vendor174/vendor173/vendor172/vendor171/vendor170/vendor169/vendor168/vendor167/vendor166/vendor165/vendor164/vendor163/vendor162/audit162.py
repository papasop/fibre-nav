"""C161 archive, actual head, full loss/update and native endpoint reconstruction."""
from core162 import *
from audit147 import source_check
from baseline158 import record
def baseline_audit(z,ctx,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);first=None;diagnostics={}
 configs=[('native_logits','NATIVE_FP32_BASELINE_LOGITS',ctx['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('native_cached_response','NATIVE_FP32_CACHED_RESPONSE',np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'cross_run_descriptive_only'),('FP64_cross_run','FUNCTIONAL_FP64_BASELINE_LOGITS',ctx['baseline_logits']['FP64_logits'],'cross_run_descriptive_only'),('FP64_repeat','WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',None,'within_run_repeat')]
 for key,name,expected,role in configs:
  a=z.arrays('model_runtime/'+name+'.npz');compare(set(a),{'actual','archived'});expected=first if key=='FP64_repeat' else expected;exact(a['archived'],expected,'C161 unchanged reference/'+key)
  with contextlib.redirect_stdout(__import__('io').StringIO()):r=record(out,name,a['actual'],a['archived'],role)
  compare(z.json('model_runtime/'+name+'.json'),r,'C161 baseline arithmetic');diagnostics[key]=r
  if key=='FP64_cross_run':first=a['actual']
 need(diagnostics['FP64_repeat']['passed_original_replay_tolerance'],'Within-run FP64 zero repeat')
 refs=z.json('model_runtime/REFERENCES_FROZEN_BEFORE_FOCUS.json');compare(refs['protocol'],parent.PROTOCOL);compare(refs['model_forwards_so_far'],3);compare(refs['state_probes_so_far'],0);R={n:farray(np.asarray(v,np.float64),(6,)) for n,v in refs['references'].items()};compare(set(R),{'current_FP64','archived_FP64','archived_FP32'});compare(refs['reference_tensor_sha256'],{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in R.items()});exact(R['archived_FP32'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'Original cached response');exact(R['archived_FP64'],np.asarray(ctx['baseline']['FP64_response_reference'],float),'Original double reference');close(R['current_FP64'],first[:,ctx['modelpins']['response_ids']].mean(0),'Same-run zero reference identity',1e-12)
 return diagnostics
def seal(z,folder,c,p,tag,physical=False):
 h=z.arrays(folder+'/HEAD.npz');compare(set(h),{'P'});exact(h['P'],p,'Actual sealed head/'+folder);receipt=z.json(folder+'/SEALED.json');compare(receipt['tag'],tag);compare(receipt['head_digest'],parameter_hash(p));compare(receipt['state_ids'],IDS);compare(receipt['before_measurement'],True);expected={'HEAD.npz','POINT_BINDINGS.json'}|({'PHYSICAL_FP32_POINTS.npz'} if physical else set());compare(set(receipt['artifact_seals']),expected)
 for n,digest in receipt['artifact_seals'].items():compare(hashlib.sha256(z.raw(folder+'/'+n)).hexdigest(),digest,'Actual head seal/'+folder+'/'+n)
 rows=z.json(folder+'/POINT_BINDINGS.json');compare([r['state_id'] for r in rows],IDS,'All84 sealed point order');native=z.arrays(folder+'/PHYSICAL_FP32_POINTS.npz') if physical else {};points={}
 if physical:compare(set(native),set(IDS),'All84 actual native points')
 for row in rows:
  sid=row['state_id'];point,e=mapping(c,p,sid);points[sid]=point;compare(row['continuous_FP64_sha256'],point_hash(point),'Reconstructed NumPy continuous point hash/'+sid)
  if physical:
   exact(native[sid],point.astype(np.float32),'Original physical FP32 formula/'+sid);compare(row['physical_FP32_sha256'],safe.tensor_hash(native[sid]),'Actual saved physical point/'+sid)
 return points,native
def audit(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with archive155.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source161'],'C161');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production'])
  for key,value in dict(training_states_requested=84,new_training_steps=8,new_head_fits=1,final_checkpoint_step=8,new_model_forwards=2187,functional_FP64_forwards=1514,new_backward_calls=1344,scalar_backward_attempts=1344,records_completed=1092,requests_consumed=1092,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0).items():compare(s[key],value,'C161 fixed denominator/'+key)
  for key in ('candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decisions_reclassified','preparation_replay_performed','FP64_acceptance_evaluated','LC2_closed','LC2_single_fold_passed'):compare(s[key],False)
  for key in ('final_head_frozen','native_repeats_consistent','all_requested_states_retained'):compare(s[key],True)
  compare(s['scope'],parent.design()['scope']);compare(s['C160_parent_decision'],ctx['completed160']['decision']);compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for key in parent.INPUT_KEYS:compare(links[key]['sha256'],c['pins'][key]['sha256'],'C161 original link/'+key)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design(),'Frozen C161 protocol');compare(z.json('ALL1092_REQUESTS_FROZEN.json'),parent.requests());chain=parent.previous.constants(ctx['data']);actual=z.arrays('FROZEN_FULL84_CHAIN_CONSTANTS.npz');compare(set(actual),set(chain))
  for n,a in chain.items():exact(actual[n],a,'C161 actual original tensor/'+n)
  if c['production']:compare(chain['P0'].shape,(3,2,36))
  refs=z.arrays('FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz');compare(set(refs),{'response','baseline_logits'});exact(refs['response'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],float),'Unchanged primary response');exact(refs['baseline_logits'],ctx['baseline_logits']['FP32_logits_cast64'],'Unchanged primary logits')
  before=z.json('SEALED_BEFORE_MODEL.json');compare(before['model_forwards_so_far'],0);compare(set(before['artifact_seals']),{'DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','ALL1092_REQUESTS_FROZEN.json'})
  for n,h in before['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Actual before-model artifact/'+n)
  base=baseline_audit(z,ctx,out/'baseline');rows=[clean_json(line) for line in z.raw('ALL1092_RECORDS.jsonl').splitlines()];compare([{k:r[k] for k in ('stage','state_id','step')} for r in rows],parent.requests(),'All1092 ordered observations');groups={}
  for r in rows:
   compare(r['case'],r['state_id'].split('__')[0],'Actual case binding');groups.setdefault((r['stage'],r['step']),[]).append(r)
  p=chain['P0'].copy();m=np.zeros_like(p);v=np.zeros_like(p);heads=[p.copy()];updates=[];aggs=[];batches=[];history=z.json('ALL8_UPDATE_HISTORY.json');compare(len(history),8);points0,native0=seal(z,'initial_head',chain,p,'initial0',True)
  for sid in IDS:exact(native0[sid],ctx['data']['points']['zero__'+sid],'Original alpha0 FP32 point')
  def check_continuous(rr,head,points,gradient):
   compare([r['state_id'] for r in rr],IDS)
   for r in rr:
    sid=r['state_id'];compare(r['parameters_digest'],parameter_hash(head),'Actual recorded head digest');compare(r['sealed_numpy_point_sha256'],point_hash(points[sid]),'Recorded NumPy point binding');need(isinstance(r['FP64_point_sha256'],str) and len(r['FP64_point_sha256'])==64 and all(k in '0123456789abcdef' for k in r['FP64_point_sha256']),'Recorded Torch point hash encoding');need(np.isfinite(r['map_Linf']) and 0<=r['map_Linf']<=1e-12,'Archived map agreement declaration');compare(r['FP64_acceptance_evaluated'],False);_,e=mapping(chain,head,sid);parts=losses(r['observed'],float(r['KL']),points[sid],chain[sid+'__prepared'],e,r['case']);compare(r['objective_parts'],parts,'Independent observed full-loss arithmetic',atol=1e-10,rtol=1e-9)
    if gradient:
     for name in ('response_gradient','guard_gradient'):farray(np.asarray(r[name],np.float64),head.shape)
  for t in range(8):
   budget();rr=groups[('training',t)];points,_=seal(z,'step_%02d_before'%(t+1),chain,p,'step%d_before'%(t+1));check_continuous(rr,p,points,True);g,agg=aggregate(rr,p.shape);saved=z.arrays('UPDATE_STEP_%02d.npz'%(t+1));compare(set(saved),{'before','gradient','m_before','v_before','after','m_after','v_after'});exact(saved['before'],p,'Step before exact chain');exact(saved['m_before'],m,'Step first moment chain');exact(saved['v_before'],v,'Step second moment chain');close(saved['gradient'],g,'All84 independently aggregated gradient',1e-10);want,mm,vv,diag=adam(p,g,m,v,t+1);errs={}
   for name,a in [('after',want),('m_after',mm),('v_after',vv)]:errs[name]=close(saved[name],a,'Independent Adam '+name,2e-16)
   expected=dict(step=t+1,head_before=parameter_hash(p),head_after=parameter_hash(saved['after']),aggregation=agg,optimizer=diag);compare(history[t],expected,'C161 update history recomputed',atol=1e-10,rtol=1e-9);updates.append(dict(step=t+1,gradient=g,delta=saved['after']-p,arithmetic_errors=errs,optimizer=diag));p=saved['after'];m=saved['m_after'];v=saved['v_after'];heads.append(p.copy());aggs.append(agg);batches.append(rr);print('C162_UPDATE_AUDIT',t+1,'/8',flush=True)
  finalpoints,native8=seal(z,'final_step8',chain,p,'fixed_final_step8',True);finalhead=z.arrays('FIXED_FINAL_HEAD_STEP8.npz');compare(set(finalhead),{'P'});exact(finalhead['P'],p,'Actual fixed final head equals step8');compare(s['final_head_digest'],parameter_hash(p));compare(z.json('FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json'),dict(step=8,head_digest=parameter_hash(p),selected_from_candidates=False,optimizer_updates_completed=8,record_count_before_final_test=756));compare(s['head_updated'],bool(np.any(p!=chain['P0'])));final=groups[('final_FP64',8)];check_continuous(final,p,finalpoints,False);compare(z.json('FINAL_FP64_OBJECTIVE_ALL84.json'),[{k:v for k,v in r.items() if k not in ('stage','step')} for r in final]);aggs.append(objective(final));batches.append(final);compare(s['initial_FP64_objective'],aggs[0]);compare(s['final_FP64_objective'],aggs[8]);compare(s['objective_change_final_minus_initial'],aggs[8]['objective']-aggs[0]['objective'])
  natives={};summaries={}
  for name,t,points,key in [('native_initial0',0,native0,'native_initial_round0'),('native_initial1',0,native0,'native_initial_round1'),('native_final0',8,native8,'native_final_round0'),('native_final1',8,native8,'native_final_round1')]:
   rr=groups[(name,t)];compare([r['state_id'] for r in rr],IDS)
   for r in rr:
    sid=r['state_id'];pt=points[sid];met=archive155.full_metric(r['metrics']['observed'],r['metrics']['KL'],pt,r['case']);qual=archive155.endpoint_gates(met,r['case'],pt,chain[sid+'__prepared'].astype(np.float32),True);compare(r['metrics'],met,'Native metrics recomputed');compare(r['qualification'],qual,'All original native gates recomputed')
    if c['production']:compare(r['actual_physical_FP32_commit'],True)
   natives[name]=rr;summaries[key]=parent.native_summary(rr);compare(s[key],summaries[key])
  compare(z.json('ALL336_NATIVE_ENDPOINTS.json'),{k:[{n:v for n,v in r.items() if n not in ('stage','step')} for r in rr] for k,rr in natives.items()});repeats=[]
  for arm in ('initial','final'):
   for a,b in zip(natives['native_'+arm+'0'],natives['native_'+arm+'1']):
    r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Original native repeats/gates');repeats.append(dict(arm=arm,state_id=a['state_id'],**r))
  compare(z.json('ALL168_NATIVE_REPEAT_CHECKS.json'),repeats);pairs={str(t):parent.paired(natives['native_initial'+str(t)],natives['native_final'+str(t)]) for t in (0,1)};compare(s['paired_by_round'],pairs);passed=all(summaries['native_final_round'+str(t)]['passed']==84 for t in (0,1));compare(s['final_training_all84_passed_both_rounds'],passed);compare(s['decision'],'C161_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION' if passed else 'C161_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE_NO_CONFIRMATION')
  return dict(summary=s,members_hashed=members,chain=chain,heads=heads,updates=updates,aggregations=aggs,batches=batches,natives=natives,native_summaries=summaries,paired=pairs,repeat_checks=repeats,baseline=base,maximum_update_arithmetic_error_Linf=max(max(u['arithmetic_errors'].values()) for u in updates),archived_Torch_FP64_point_bytes_available=False)
