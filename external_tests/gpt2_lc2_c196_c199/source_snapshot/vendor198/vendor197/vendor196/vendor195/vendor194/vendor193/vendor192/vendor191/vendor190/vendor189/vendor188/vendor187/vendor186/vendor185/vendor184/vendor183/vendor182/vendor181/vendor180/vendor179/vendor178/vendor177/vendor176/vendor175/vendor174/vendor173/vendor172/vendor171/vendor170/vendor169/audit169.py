"""Bind original C168 points, heads, records and gates to all thirteen predecessors."""
from core169 import *
from audit147 import source_check
from audit164 import seal,actual_batch
from baseline169 import audit as baseline_audit
from recompute169 import reconstruct
def audit(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source168'],'C168');s=z.json('SUMMARY.json');compare(s['protocol'],prior.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['decision'],'C168_FIXED_CERTIFIED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['stage'],'complete');compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['scope'],prior.SCOPE)
  for k,v in dict(new_training_steps=0,new_head_fits=0,new_model_parameter_gradients=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_linear_program_calls=0,new_direction_optimization_iterations=0,temporary_probe_heads=6,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_model_forwards=3699,functional_FP64_forwards=1346,new_backward_calls=0,scalar_backward_attempts=0,records_completed=1848,requests_consumed=1848,actual_FP64_point_tensors_saved=672).items():compare(s[k],v,'C168 fixed count/'+k)
  for k in ('candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decisions_reclassified','archived_source_executed','chronology_or_nonaccess_proved','preparation_replay_performed','FP64_acceptance_evaluated','new_AD_evaluated','LC2_closed','LC2_single_fold_passed'):compare(s[k],False,'C168 scope/'+k)
  for k in ('exact_recorded_gradient_certificate_reverified','all6_NumPy_heads_sealed_before_model','all_requested_states_and_arms_retained','all6_actual_point_batches_sealed_before_first_state','actual_FP64_points_checked_before_model_forward','initial_native_and_FP64_replay_matched_C165','native_repeats_consistent','FP64_anchor_repeat_consistent'):compare(s[k],True)
  if c['production']:compare(s['all_thirteen_originals_independently_bound'],True)
  compare(s['C165_parent_decision'],'C165_FIXED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED');compare(s['C166_parent_decision'],ctx['C166_receipt']['decision']);compare(s['C167_parent_decision'],ctx['C167_receipt']['decision']);compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C168 original link/'+k)
  compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),prior.design());compare(z.json('ALL1848_REQUESTS_FROZEN.json'),prior.requests())
  chain=prior.old.parent.parent.previous.constants(ctx['data']);saved=z.arrays('FROZEN_FULL84_CHAIN_CONSTANTS.npz');compare(set(saved),set(chain))
  for k,a in chain.items():exact(saved[k],a,'Fixed full84 tensor/'+k)
  p=ctx['initial_head'];d=ctx['fixed_C167_direction'];a=z.arrays('ANCHOR_C163_FIXED_FINAL_HEAD.npz');compare(set(a),{'P'});exact(a['P'],p,'Exact C163 anchor');a=z.arrays('ONE_FIXED_C167_DIRECTION.npz');compare(set(a),{'anchor','direction'});exact(a['anchor'],p,'Direction anchor');exact(a['direction'],d,'Exact C167 certificate direction');compare(s['initial_C163_fixed_final_head_digest'],parameter_hash(p));compare(s['C167_direction_digest'],array_digest(d))
  if c['production']:compare(p.shape,(3,2,36))
  heads,panel=prior.step_panel(p,d);compare(z.json('FIXED6_STEP_PANEL_BEFORE_MODEL.json'),panel,'Original prewrite-only step rule');compare(s['positive_steps'],panel['positive_steps'])
  files={'DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','ANCHOR_C163_FIXED_FINAL_HEAD.npz','ONE_FIXED_C167_DIRECTION.npz','FIXED6_STEP_PANEL_BEFORE_MODEL.json','ALL1848_REQUESTS_FROZEN.json'};before=z.json('ALL6_NUMPY_HEADS_SEALED_BEFORE_MODEL.json');compare(set(before['artifact_seals']),files);compare(before['model_forwards_so_far'],0);compare(before['backward_calls_so_far'],0);compare(before['all6_head_digests'],{n:parameter_hash(q) for n,q in heads.items()})
  for n,h in before['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Pre-model artifact/'+n)
  baseline=baseline_audit(z,ctx,out/'baseline');rows=[clean_json(line) for line in z.raw('ALL1848_RECORDS.jsonl').splitlines()];compare([{k:r[k] for k in ('stage','arm','state_id')} for r in rows],prior.requests(),'Complete ordered1848 records');groups={}
  for r in rows:compare(r['case'],r['state_id'].split('__')[0]);groups.setdefault((r['stage'],r['arm']),[]).append(r)
  pts,nt=seal(z,'anchor',chain,p,'C163_fixed_final_anchor',True);points={'anchor':pts};native_points={'anchor':nt};seals=[];audit_rows=[];head_seals={}
  for i,sid in enumerate(IDS):compare(safe.tensor_hash(nt[sid]),ctx['C165_native_anchor_records']['0'][i]['metrics']['theta_sha256'],'Original C165 anchor native identity')
  for name,q in heads.items():points[name],native_points[name]=seal(z,name,chain,q,name,True);head_seals[name]=z.json(name+'/SEALED.json')
  compare(before['head_seals'],head_seals)
  points['anchor_repeat'],_=seal(z,'anchor_repeat',chain,p,'anchor_repeat',False);actuals={}
  def continuous(rr,head,folder,tag):
   compare([r['state_id'] for r in rr],IDS);actual,receipt=actual_batch(z,folder,head,points[folder],tag,3,0,budget);seals.append(receipt);actuals[folder]=actual
   for r in rr:
    budget();sid=r['state_id'];compare(r['parameters_digest'],parameter_hash(head));compare(r['sealed_numpy_point_sha256'],point_hash(points[folder][sid]));compare(r['FP64_point_sha256'],point_hash(actual[sid]));compare(r['actual_FP64_point_artifact'],folder+'/ACTUAL_TORCH_FP64_POINTS.npz');err=float(max(abs(actual[sid]-points[folder][sid])));compare(r['map_Linf'],err,atol=1e-18,rtol=1e-12);compare(r['FP64_acceptance_evaluated'],False);need('response_gradient' not in r and 'guard_gradient' not in r,'C168 performs no new AD');_,e=mapping(chain,head,sid);o=farray(np.asarray(r['observed'],float),(22,));kl=float(r['KL']);need(np.isfinite(kl) and kl>=0,'Finite archived KL');parts=losses(o,kl,actual[sid],chain[sid+'__prepared'],e,r['case']);compare(r['objective_parts'],parts,'Independent loss reconstruction',atol=1e-10,rtol=1e-9);audit_rows.append(dict(folder=folder,state_id=sid,head_sha256=parameter_hash(head),actual_FP64_point_sha256=point_hash(actual[sid]),NumPy_point_sha256=point_hash(points[folder][sid]),recomputed_map_Linf=err))
  anchor=groups[('FP64_anchor','anchor')];continuous(anchor,p,'anchor','FP64_anchor')
  for i,r in enumerate(anchor):
   old=ctx['C165_FP64_anchor_records'][i];re=dict(observed_Linf=float(max(abs(np.asarray(r['observed'])-old['observed']))),KL_absolute=abs(r['KL']-old['KL']));re['within_original_replay_tolerance']=re['observed_Linf']<=2e-9 and re['KL_absolute']<=2e-9;compare(r['C165_FP64_anchor_replay'],re);need(re['within_original_replay_tolerance'],'C165 FP64 anchor replay')
  probes={}
  for name,q in heads.items():probes[name]=groups[('FP64_probe',name)];continuous(probes[name],q,name,name)
  repeat=groups[('FP64_anchor_repeat','anchor')];continuous(repeat,p,'anchor_repeat','anchor_repeat');compare(z.json('ALL8_ACTUAL_FP64_POINT_SEALS.json'),seals);compare(len(audit_rows),672)
  receipt=z.json('ALL6_PROBES_SEALED_BEFORE_FIRST_STATE.json');compare(receipt,dict(model_forwards_so_far=3,backward_calls_so_far=0,all6_head_digests={n:parameter_hash(q) for n,q in heads.items()},actual_point_seals=seals,all6_before_first_state=True))
  native={}
  for name in ['anchor']+prior.ARM_NAMES:
   for rnd in (0,1):
    stage=('native_anchor' if name=='anchor' else 'native_probe')+str(rnd);rr=groups[(stage,name)];compare([r['state_id'] for r in rr],IDS)
    for i,r in enumerate(rr):
     budget();sid=r['state_id'];pt=native_points[name][sid];met=archive155.full_metric(r['metrics']['observed'],r['metrics']['KL'],pt,r['case']);qual=archive155.endpoint_gates(met,r['case'],pt,chain[sid+'__prepared'].astype(np.float32),True);compare(r['metrics'],met,'Recomputed native measurements arithmetic');compare(r['qualification'],qual,'All original native gates')
     if c['production']:compare(r['actual_physical_FP32_commit'],True)
     if name=='anchor':
      re=archive155.repeat_check(ctx['C165_native_anchor_records'][str(rnd)][i]['metrics'],met);compare(r['C165_native_anchor_replay'],re);need(re['within_replay_tolerance'] and re['same_point_identity'],'C165 native anchor replay')
    native[(name,rnd)]=rr
  diag,fd,changes,fr,nr=reconstruct(anchor,repeat,probes,d,ctx['C165_gradient_records'],native,panel)
  for name,v in [('DIAGNOSTIC_SUMMARY.json',diag),('ALL252_CENTRAL_DIFFERENCES_VS_ARCHIVED_AD.json',fd),('ALL504_SIGNED_LOSS_CHANGES.json',changes),('ALL84_FP64_ANCHOR_REPEATS.json',fr),('ALL588_NATIVE_REPEAT_CHECKS.json',nr),('ALL1176_NATIVE_ENDPOINTS.json',{n+'__round'+str(t):stripped(rr) for (n,t),rr in native.items()}),('ALL6_FP64_PROBE_RECORDS.json',{n:stripped(rr) for n,rr in probes.items()})]:compare(z.json(name),v,'Original duplicate/summary recomputed/'+name)
  compare(s['diagnostics'],diag)
  for name,v in [('ALL672_ACTUAL_FP64_POINT_AUDIT_ROWS.json',audit_rows),('ALL8_ACTUAL_POINT_SEALS_RECHECKED.json',seals),('PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json',diag),('ALL252_CENTRAL_DIFFERENCES_RECOMPUTED.json',fd),('ALL504_SIGNED_LOSS_CHANGES_RECOMPUTED.json',changes),('ALL84_FP64_REPEATS_RECOMPUTED.json',fr),('ALL588_NATIVE_REPEATS_RECOMPUTED.json',nr)]:dump(out/name,v)
  return dict(summary=s,members_hashed=members,chain=chain,head=p,direction=d,panel=panel,anchor=anchor,repeat=repeat,probes=probes,native=native,points=points,actual_points=actuals,native_points=native_points,archived_gradients=ctx['C165_gradient_records'],diagnostics=diag,fd=fd,changes=changes,baseline=baseline,maximum_map_error=max(r['recomputed_map_Linf'] for r in audit_rows))
