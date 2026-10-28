"""Recompute C168 saved summaries without importing its measurement worker."""
from core173 import *
def reconstruct(anchor,repeat,probes,d,native,panel,p,heads):
 archived=anchor
 compare([r['state_id'] for r in anchor],IDS);compare([r['state_id'] for r in archived],IDS)
 for name in prior.ARM_NAMES:compare([r['state_id'] for r in probes[name]],IDS)
 active=[r['state_id'] for r in archived if r['objective_parts']['response_loss']>0];fd=[];changes=[];fr=[];nr=[];summaries={};pairs={}
 for i,h in enumerate(panel['positive_steps'],1):
  minus=probes['minus_s'+str(i)];plus=probes['plus_s'+str(i)]
  for j,sid in enumerate(IDS):
   checks={}
   for name,gkey in [('response','response_gradient'),('guard','guard_gradient')]:
    lo=float(minus[j]['objective_parts']['response_loss']) if name=='response' else guard(minus[j]);hi=float(plus[j]['objective_parts']['response_loss']) if name=='response' else guard(plus[j]);ad=dot(archived[j][gkey],d);cd=(hi-lo)/(2*h)
    checks[name]=dict(fresh_C172_AD_directional=ad,current_C172_central_difference=cd,difference=cd-ad,absolute_difference=abs(cd-ad),new_AD_evaluated=True,no_derivative_accuracy_or_error_cause_claim=True)
   fd.append(dict(state_id=sid,h=h,checks=checks))
 for arm in panel['arms']:
  for j,sid in enumerate(IDS):
   a=anchor[j];b=probes[arm['name']][j];r0=float(a['objective_parts']['response_loss']);r1=float(b['objective_parts']['response_loss']);g0=guard(a);g1=guard(b)
   changes.append(dict(arm=arm['name'],alpha=arm['alpha'],state_id=sid,active_at_C170_final_anchor=sid in active,response_loss_before=r0,response_loss_after=r1,observed_response_loss_change=r1-r0,archived_linear_response_loss_change=arm['alpha']*dot(archived[j]['response_gradient'],d),guard_loss_before=g0,guard_loss_after=g1,observed_guard_loss_change=g1-g0,archived_linear_guard_loss_change=arm['alpha']*dot(archived[j]['guard_gradient'],d),strict_measured_response_loss_decrease=bool(r1<r0)))
 for a,b in zip(anchor,repeat):
  compare(a['state_id'],b['state_id']);compare(a['FP64_point_sha256'],b['FP64_point_sha256']);err=float(max(abs(np.asarray(a['observed'])-b['observed'])));ke=abs(a['KL']-b['KL']);need(err<=2e-9 and ke<=2e-9,'Original FP64 repeat tolerance');fr.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=ke,within_original_repeat_tolerance=True))
 for name in ['anchor']+prior.ARM_NAMES:
  summaries[name]={str(t):prior.native_summary(native[(name,t)]) for t in (0,1)}
  for a,b in zip(native[(name,0)],native[(name,1)]):
   r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Original native repeat tolerance and gates');nr.append(dict(arm=name,state_id=a['state_id'],**r))
  if name!='anchor':pairs[name]={str(t):prior.paired(native[('anchor',t)],native[(name,t)]) for t in (0,1)}
 stats={name:dict(maximum_FP64_response=max(float(max(abs(np.asarray(r['observed'])[:6]))) for r in rr),FP64_response_only_passed=int(sum(max(abs(np.asarray(r['observed'])[:6]))<=.002 for r in rr)),FP64_full_acceptance_claim=False) for name,rr in {'anchor':anchor,**probes}.items()};positive={}
 for name in ['plus_s1','plus_s2','plus_s3']:
  rr=[r for r in changes if r['arm']==name and r['active_at_C170_final_anchor']];positive[name]=dict(active_requested=len(active),active_measured=len(rr),strict_response_loss_decreases=sum(r['strict_measured_response_loss_decrease'] for r in rr),all_active_losses_strictly_decreased=bool(rr and all(r['strict_measured_response_loss_decrease'] for r in rr)),descriptive_only_no_new_acceptance_gate=True)
 diag=dict(native_summaries=summaries,paired_vs_anchor=pairs,FP64_response_statistics=stats,original_active_state_ids=active,all3_positive_arms_active_loss_changes=positive,all84_all6_arms_retained=True,all252_central_difference_rows_retained=True,all504_one_sided_loss_changes_retained=True,fresh_gradients_at_C170_final_anchor=True,new_AD_evaluated=True,any_probe_pass_is_training_diagnostic_only=True,step_or_head_selected=False,LC2_closed=False)
 for row in changes:
  j=IDS.index(row['state_id']);q=heads[row['arm']];terms={}
  for kind,key in [('response','response_gradient'),('guard','guard_gradient')]:
   exact_dot=sum((rational(g)*(rational(y)-rational(x)) for g,x,y in zip(np.asarray(anchor[j][key]).ravel(),p.ravel(),q.ravel())),rational(0));terms[kind]=dict(exact_recorded_gradient_dot_actual_binary64_head_difference=encoded(exact_dot),finite_minus_recorded_linear_change=row['observed_'+kind+'_loss_change']-float(exact_dot),not_a_curvature_bound=True)
  row['actual_binary64_head_change_diagnostic']=terms
 frames={'anchor':anchor,**probes}
 diag.update(anchor='Original C170 fixed final step8',fresh_anchor_scalar_backwards=168,temporary_probe_heads=6,all84_all6_original_native_gates_measured_both_rounds=True,all_probe_heads_from_one_fresh_direction=True,guard_parts_by_arm={name:{part:dict(mean=float(np.mean([r['objective_parts'][part] for r in rows])),maximum=float(max(r['objective_parts'][part] for r in rows))) for part in PARTS[1:]} for name,rows in frames.items()},FP64_objectives_by_arm={name:control.objective(rows) for name,rows in frames.items()},guard_increase_does_not_imply_hard_gate_failure=True,linear_residual_is_not_a_curvature_or_derivative_error_bound=True,solver_optimality_or_infeasibility_certified=False,curvature_or_convergence_certified=False,causal_explanation_established=False,no_future_step_or_closure_time_extrapolation=True)
 compare((len(fd),len(changes),len(fr),len(nr)),(252,504,84,588));return diag,fd,changes,fr,nr

def dot(g,d):return float(np.sum(np.asarray(g)*d))
