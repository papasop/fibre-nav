"""All fixed arms, central differences and original gates; no selection."""
from core172 import *
def legacy(anchor,repeat,probes,d,archived,native,panel):
 compare([r['state_id'] for r in anchor],IDS);compare([r['state_id'] for r in archived],IDS);fd=[];changes=[];active=[r['state_id'] for r in archived if r['objective_parts']['response_loss']>0]
 for i,h in enumerate(panel['positive_steps']):
  for j,sid in enumerate(IDS):
   a=probes['minus_s'+str(i+1)][j];b=probes['plus_s'+str(i+1)][j];parts={}
   for name,terms,key in [('response',('response_loss',),'response_gradient'),('guard',PARTS[1:],'guard_gradient')]:
    lo=sum(a['objective_parts'][k] for k in terms);hi=sum(b['objective_parts'][k] for k in terms);ad=float(np.sum(np.asarray(archived[j][key])*d));central=(hi-lo)/(2*h);parts[name]=dict(fresh_C172_AD_directional=ad,current_C172_central_difference=central,difference=central-ad,absolute_difference=abs(central-ad),new_AD_evaluated=True,no_derivative_accuracy_or_error_cause_claim=True)
   fd.append(dict(state_id=sid,h=h,checks=parts))
 for arm in panel['arms']:
  for i,sid in enumerate(IDS):
   oldR=anchor[i]['objective_parts']['response_loss'];newR=probes[arm['name']][i]['objective_parts']['response_loss'];rd=float(np.sum(np.asarray(archived[i]['response_gradient'])*d));gd=float(np.sum(np.asarray(archived[i]['guard_gradient'])*d));oldG=sum(anchor[i]['objective_parts'][k] for k in PARTS[1:]);newG=sum(probes[arm['name']][i]['objective_parts'][k] for k in PARTS[1:]);changes.append(dict(arm=arm['name'],alpha=arm['alpha'],state_id=sid,active_at_C170_final_anchor=sid in active,response_loss_before=oldR,response_loss_after=newR,observed_response_loss_change=newR-oldR,archived_linear_response_loss_change=arm['alpha']*rd,guard_loss_before=oldG,guard_loss_after=newG,observed_guard_loss_change=newG-oldG,archived_linear_guard_loss_change=arm['alpha']*gd,strict_measured_response_loss_decrease=bool(newR<oldR)))
 repeats=[]
 for a,b in zip(anchor,repeat):
  compare(a['state_id'],b['state_id']);compare(a['FP64_point_sha256'],b['FP64_point_sha256']);err=float(np.max(abs(np.asarray(a['observed'])-b['observed'])));kl=abs(a['KL']-b['KL']);need(err<=2e-9 and kl<=2e-9,'Original same-run FP64 anchor repeat');repeats.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=kl,within_original_repeat_tolerance=True))
 summaries={};pairs={};nr=[]
 for name in ['anchor']+ARM_NAMES:
  summaries[name]={str(t):native_summary(native[(name,t)]) for t in (0,1)}
  for a,b in zip(native[(name,0)],native[(name,1)]):
   r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Original native repeated same-point/gates');nr.append(dict(arm=name,state_id=a['state_id'],**r))
  if name!='anchor':pairs[name]={str(t):paired(native[('anchor',t)],native[(name,t)]) for t in (0,1)}
 frames={'anchor':anchor,**probes};stats={}
 for name,rr in frames.items():
  stats[name]=dict(maximum_FP64_response=max(float(max(abs(np.asarray(r['observed'])[:6]))) for r in rr),FP64_response_only_passed=int(sum(max(abs(np.asarray(r['observed'])[:6]))<=.002 for r in rr)),FP64_full_acceptance_claim=False)
 plus={}
 for name in ('plus_s1','plus_s2','plus_s3'):
  rows=[r for r in changes if r['arm']==name and r['active_at_C170_final_anchor']];plus[name]=dict(active_requested=len(active),active_measured=len(rows),strict_response_loss_decreases=sum(r['strict_measured_response_loss_decrease'] for r in rows),all_active_losses_strictly_decreased=bool(rows and all(r['strict_measured_response_loss_decrease'] for r in rows)),descriptive_only_no_new_acceptance_gate=True)
 diag=dict(native_summaries=summaries,paired_vs_anchor=pairs,FP64_response_statistics=stats,original_active_state_ids=active,all3_positive_arms_active_loss_changes=plus,all84_all6_arms_retained=True,all252_central_difference_rows_retained=True,all504_one_sided_loss_changes_retained=True,fresh_gradients_at_C170_final_anchor=True,new_AD_evaluated=True,any_probe_pass_is_training_diagnostic_only=True,step_or_head_selected=False,LC2_closed=False)
 return diag,fd,changes,repeats,nr

def analyse(anchor,repeat,probes,d,native,panel,p,heads):
 compare(panel['positive_steps'],list(H));compare([a['name'] for a in panel['arms']],ARM_NAMES)
 for name in ARM_NAMES:compare([r['state_id'] for r in probes[name]],IDS,'Full84 every probe arm')
 diag,fd,changes,fr,nr=legacy(anchor,repeat,probes,d,anchor,native,panel)
 from core171 import rational,encoded
 for row in changes:
  i=IDS.index(row['state_id']);q=heads[row['arm']];old=anchor[i];terms={}
  for field,key in [('response','response_gradient'),('guard','guard_gradient')]:
   exact_dot=sum((rational(g)*(rational(b)-rational(a)) for g,a,b in zip(np.asarray(old[key]).ravel(),p.ravel(),q.ravel())),rational(0));actual=row['observed_'+field+'_loss_change'];terms[field]=dict(exact_recorded_gradient_dot_actual_binary64_head_difference=encoded(exact_dot),finite_minus_recorded_linear_change=actual-float(exact_dot),not_a_curvature_bound=True)
  row['actual_binary64_head_change_diagnostic']=terms
 diag.update(anchor='Original C170 fixed final step8',fresh_anchor_scalar_backwards=168,temporary_probe_heads=6,all84_all6_original_native_gates_measured_both_rounds=True,all_probe_heads_from_one_fresh_direction=True,guard_parts_by_arm={name:{part:dict(mean=float(np.mean([r['objective_parts'][part] for r in rows])),maximum=float(max(r['objective_parts'][part] for r in rows))) for part in PARTS[1:]} for name,rows in dict(anchor=anchor,**probes).items()},FP64_objectives_by_arm={name:control.objective(rows) for name,rows in dict(anchor=anchor,**probes).items()},guard_increase_does_not_imply_hard_gate_failure=True,linear_residual_is_not_a_curvature_or_derivative_error_bound=True,solver_optimality_or_infeasibility_certified=False,curvature_or_convergence_certified=False,causal_explanation_established=False,no_future_step_or_closure_time_extrapolation=True)
 return diag,fd,changes,fr,nr
