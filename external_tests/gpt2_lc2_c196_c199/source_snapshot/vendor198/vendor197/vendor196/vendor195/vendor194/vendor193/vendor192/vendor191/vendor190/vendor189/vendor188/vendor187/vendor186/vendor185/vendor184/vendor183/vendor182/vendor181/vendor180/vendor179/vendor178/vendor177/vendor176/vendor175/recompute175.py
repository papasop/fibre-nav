"""Recompute C168 saved summaries without importing its measurement worker."""
from core175 import *
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
    checks[name]=dict(fresh_C174_AD_directional=ad,current_C174_central_difference=cd,difference=cd-ad,absolute_difference=abs(cd-ad),new_AD_evaluated=True,no_derivative_accuracy_or_error_cause_claim=True)
   fd.append(dict(state_id=sid,h=h,checks=checks))
 for arm in panel['arms']:
  for j,sid in enumerate(IDS):
   a=anchor[j];b=probes[arm['name']][j];r0=float(a['objective_parts']['response_loss']);r1=float(b['objective_parts']['response_loss']);g0=guard(a);g1=guard(b)
   changes.append(dict(arm=arm['name'],alpha=arm['alpha'],state_id=sid,active_at_C170_final_anchor=sid in active,response_loss_before=r0,response_loss_after=r1,observed_response_loss_change=r1-r0,archived_linear_response_loss_change=arm['alpha']*dot(archived[j]['response_gradient'],d),guard_loss_before=g0,guard_loss_after=g1,observed_guard_loss_change=g1-g0,archived_linear_guard_loss_change=arm['alpha']*dot(archived[j]['guard_gradient'],d),strict_measured_response_loss_decrease=bool(r1<r0)))
 for a,b in zip(anchor,repeat):
  compare(a['state_id'],b['state_id']);compare(a['FP64_point_sha256'],b['FP64_point_sha256']);err=float(max(abs(np.asarray(a['observed'])-b['observed'])));ke=abs(a['KL']-b['KL']);need(err<=2e-9 and ke<=2e-9,'Original FP64 repeat tolerance');fr.append(dict(state_id=a['state_id'],observed_Linf=err,KL_absolute=ke,within_original_repeat_tolerance=True))
 for name in ['anchor']+prior.ARM_NAMES:
  summaries[name]={str(t):control.native_summary(native[(name,t)]) for t in (0,1)}
  for a,b in zip(native[(name,0)],native[(name,1)]):
   r=archive155.repeat_check(a['metrics'],b['metrics']);need(r['within_replay_tolerance'] and r['same_point_identity'],'Original native repeat tolerance and gates');nr.append(dict(arm=name,state_id=a['state_id'],**r))
  if name!='anchor':pairs[name]={str(t):control.paired(native[('anchor',t)],native[(name,t)]) for t in (0,1)}
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

def coordinate_affine(diag,anchor,probes,p,heads,pan,pr,certs,out):
 from diagnostics173 import write_csv
 out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];summaries={}
 for arm in pan['arms']:
  name=arm['name'];delta=[rational(y)-rational(x) for x,y in zip(p.ravel(),heads[name].ravel())];frame=[]
  for i,sid in enumerate(IDS):
   observed=np.asarray(probes[name][i]['observed'])[:6]
   for j in range(6):
    prediction=rational(pr['response'][i,j])+sum((rational(g)*v for g,v in zip(pr['J'][i,j],delta)),Fraction(0));value=float(observed[j]);row=dict(arm=name,alpha=arm['alpha'],state_id=sid,coordinate=j,anchor_signed_response=float(pr['response'][i,j]),recorded_linear_signed_response=encoded(prediction),actual_probe_signed_response=value,actual_minus_recorded_affine_response=value-float(prediction),training_linearisation_cap=float(pr['caps'][i]),actual_cap_slack=float(pr['caps'][i]-abs(observed[j])),original_hard_response_slack=float(.002-abs(observed[j])),training_safety_slack=float(SAFETY-abs(observed[j])),not_a_finite_error_bound=True);rows.append(row);frame.append(row)
  summaries[name]=dict(actual_all504_coordinate_caps_satisfied=all(r['actual_cap_slack']>=0 for r in frame),actual_cap_violation_coordinates=sum(r['actual_cap_slack']<0 for r in frame),actual_cap_violation_state_ids=[sid for sid in IDS if any(r['state_id']==sid and r['actual_cap_slack']<0 for r in frame)],maximum_absolute_observed_minus_affine_response=max(abs(r['actual_minus_recorded_affine_response']) for r in frame),exact_recorded_affine_minimum_margin=certs[name]['minimum_normalized_recorded_cap_slack'],exact_recorded_affine_active_margin=certs[name]['strict_recorded_active_margin'],training_caps_are_not_new_native_acceptance_gates=True)
 dump(out/'ALL3024_RESPONSE_COORDINATE_AFFINE_ROWS.json',rows);write_csv(out/'ALL3024_RESPONSE_COORDINATE_AFFINE_ROWS.csv',rows)
 diag.update(fresh_response_coordinate_Jacobian_blocks=84,fresh_response_coordinate_Jacobian_rows=504,coefficient_columns=p.size,scalar_gradient_chain_checks=[dict(state_id=r['state_id'],**chain_check(r,p.shape)) for r in anchor],all6_observed_vs_recorded_coordinate_affine_summaries=summaries,response_coordinate_affine_rows_saved=len(rows),original_acceptance_budgets_unchanged=True,all_states_including_zero_scalar_loss_constrain_LP=True,all1008_signed_rows_in_LP=True,coefficient_response_Jacobians_are_offline_development=True,online_LC1_execution_demonstrated=False)
 compare(len(rows),3024);return diag,rows

def supplemental(a,out):
 """Fixed full-coordinate secants versus archived AD; no accuracy gate or selection."""
 from diagnostics173 import write_csv,maxima
 out=Path(out);out.mkdir(parents=True,exist_ok=True);central=[];native=[];allstate={};frames={'anchor':a['anchor'],**a['probes']};frame_stats=[]
 for k,h in enumerate(H,1):
  for i,sid in enumerate(IDS):
   lo=np.asarray(a['probes']['minus_s'+str(k)][i]['observed'])[:6];hi=np.asarray(a['probes']['plus_s'+str(k)][i]['observed'])[:6]
   for j in range(6):
    ad=float(np.sum(a['problem']['J'][i,j].reshape(a['direction'].shape)*a['direction']));fd=float((hi[j]-lo[j])/(2*h));central.append(dict(state_id=sid,h=h,coordinate=j,archived_C174_coordinate_AD_directional=ad,current_C175_archived_central_difference=fd,central_minus_archived_AD=fd-ad,absolute_difference=abs(fd-ad),coordinate_AD_independently_recomputed=False,no_derivative_accuracy_or_curvature_bound=True))
 for name,rows in frames.items():
  values=[maxima(np.asarray(r['observed'])[:6])[0] for r in rows];top=max(values);frame_stats.append(dict(arm=name,maximum_FP64_response=top,worst_response_state_ids=[IDS[i] for i,x in enumerate(values) if x==top],FP64_response_only_passed=sum(x<=.002 for x in values),FP64_full_acceptance_claim=False))
  if name!='anchor':
   increases=[];decreases=[]
   for old,new in zip(a['anchor'],rows):
    before=old['objective_parts']['response_loss'];after=new['objective_parts']['response_loss']
    if after>before:increases.append(new['state_id'])
    if after<before:decreases.append(new['state_id'])
   allstate[name]=dict(all84_loss_increase_state_ids=increases,all84_strict_loss_decrease_state_ids=decreases,all84_unchanged_loss_count=84-len(increases)-len(decreases),no_new_acceptance_gate=True)
  for rnd in (0,1):
   for i,sid in enumerate(IDS):
    f=np.asarray(rows[i]['observed'])[:6];r=a['native'][(name,rnd)][i];n=np.asarray(r['metrics']['observed'])[:6];fg=bool(max(abs(f))<=.002);ng='response' not in r['qualification']['failed_gates'];native.append(dict(arm=name,round=rnd,state_id=sid,FP64_response_Linf=float(max(abs(f))),native_response_Linf=float(max(abs(n))),maximum_absolute_response_coordinate_difference=float(max(abs(n-f))),native_response_gate=ng,FP64_response_only_gate=fg,response_gate_agreement=ng==fg,native_full_acceptance=bool(r['qualification']['complete_pass']),FP64_full_acceptance_claim=False))
 compare(len(central),1512);compare(len(native),1176)
 for name,rows in [('ALL1512_COORDINATE_SECANTS_VS_ARCHIVED_AD',central),('ALL1176_NATIVE_VS_FP64_RESPONSE',native)]:dump(out/(name+'.json'),rows);write_csv(out/(name+'.csv'),rows)
 result=dict(all7_frame_statistics=frame_stats,all6_arm_all84_loss_changes=allstate,coordinate_secant_comparisons=1512,native_vs_FP64_comparisons=1176,native_vs_FP64_response_gate_disagreements=sum(not r['response_gate_agreement'] for r in native),maximum_native_vs_FP64_response_coordinate_difference=max(r['maximum_absolute_response_coordinate_difference'] for r in native),recorded_coordinate_gradients_available=True,new_derivatives_or_model_measurements=False,derivative_accuracy_or_curvature_bound_established=False,step_or_head_selected=False,LC2_closed=False)
 dump(out/'SUPPLEMENTAL_SUMMARY.json',result);return result
