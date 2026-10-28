"""All-state retrospective diagnostics; recorded gradients are not derivative proof."""
from core164 import *
from diagnostics162 import diagnostics as legacy_diagnostics

def diagnostics(data,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 # Preserve independent existing full84/native diagnostics using the original hardmax score.
 legacy=dict(data,aggregations=[old.objective(rr) for rr in data['batches']])
 d=legacy_diagnostics(legacy,out/'original_hardmax_and_precision')
 smooth_steps=[];allrows=[];trajectory=[];focus=[]
 for t,rr in enumerate(data['batches']):
  a=data['aggregations'][t]
  for r in rr:
   obs=np.asarray(r['observed']);slack=.002-abs(obs[:6]);row=dict(head_position=t,state_id=r['state_id'],response_Linf=float(max(abs(obs[:6]))),response_signed_budget_slack=slack.tolist(),response_within_budget=bool(np.all(slack>=0)),response_loss=r['objective_parts']['response_loss'],smooth_response_weight=a['response_weights'][IDS.index(r['state_id'])]);trajectory.append(row)
   if r['state_id'] in FOCUS:focus.append(row)
  if t==8:continue
  u=data['updates'][t];dp=u['delta'];after=data['batches'][t+1];sa=data['aggregations'][t+1];linear=float(np.sum(u['gradient']*dp));change=sa['objective']-a['objective'];hard=sa['original_hardmax_objective']-a['original_hardmax_objective'];avg=sa['mean_response_loss']-a['mean_response_loss'];tail=sa['maximum_response_loss']-a['maximum_response_loss']
  for i,(r,b) in enumerate(zip(rr,after)):
   rg=np.asarray(r['response_gradient']);gg=np.asarray(r['guard_gradient']);dr=float(np.sum(rg*dp));dg=float(np.sum(gg*dp));act=b['objective_parts']['response_loss']-r['objective_parts']['response_loss'];allrows.append(dict(step=t+1,state_id=r['state_id'],response_weight_before=a['response_weights'][i],response_weight_after=sa['response_weights'][i],recorded_response_gradient_dot_actual_update=dr,recorded_guard_gradient_dot_actual_update=dg,observed_response_loss_change=act,response_observed_minus_recorded_linear=act-dr,response_gradient_cosine_actual_update=cosine(rg,dp)))
  smooth_steps.append(dict(step=t+1,smooth_objective_before=a['objective'],smooth_objective_after=sa['objective'],smooth_objective_change=change,original_hardmax_before=a['original_hardmax_objective'],original_hardmax_after=sa['original_hardmax_objective'],original_hardmax_change=hard,mean_response_loss_change=avg,maximum_response_loss_change=tail,smooth_decreases_while_hardmax_increases=bool(change<0 and hard>0),average_response_loss_decreases_while_maximum_increases=bool(avg<0 and tail>0),recorded_smooth_gradient_dot_actual_update=linear,observed_smooth_change_minus_recorded_linear=change-linear,recorded_smooth_gradient_cosine_actual_update=cosine(u['gradient'],dp),response_weight_entropy=float(-np.sum(np.asarray(a['response_weights'])*np.log(a['response_weights']))),largest_response_weight=float(max(a['response_weights'])),next_aggregate_gradient_cosine=cosine(u['gradient'],data['updates'][t+1]['gradient']) if t<7 else None))
 a=data['aggregations'][0];b=data['aggregations'][8]
 d.update(smooth_objective_initial=a['objective'],smooth_objective_final=b['objective'],smooth_objective_relative_change=(b['objective']/a['objective']-1) if a['objective'] else None,original_hardmax_initial=a['original_hardmax_objective'],original_hardmax_final=b['original_hardmax_objective'],original_hardmax_relative_change=(b['original_hardmax_objective']/a['original_hardmax_objective']-1) if a['original_hardmax_objective'] else None,smooth_decrease_hardmax_increase_steps=sum(r['smooth_decreases_while_hardmax_increases'] for r in smooth_steps),recorded_smooth_gradient_uphill_updates=sum(r['recorded_smooth_gradient_dot_actual_update']>0 for r in smooth_steps),all756_response_margin_rows_retained=True,all672_smooth_transition_rows_retained=True,recorded_derivatives_independently_verified=False,optimization_failure_cause_established=False)
 dump(out/'ALL8_SMOOTH_VS_HARDMAX_STEPS.json',smooth_steps);dump(out/'ALL672_SMOOTH_RECORDED_GRADIENT_TRANSITIONS.json',allrows);dump(out/'ALL756_RESPONSE_MARGIN_TRAJECTORY.json',trajectory);dump(out/'FIXED_FIVE_ALL9_HEADS.json',focus);dump(out/'DIAGNOSTIC_SUMMARY.json',d)
 import csv
 with (out/'ALL756_STATE_MARGIN_TRAJECTORY.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['head_position','state_id','response_Linf','response_within_budget','response_loss','smooth_response_weight'])
  for r in trajectory:w.writerow([r[k] for k in ('head_position','state_id','response_Linf','response_within_budget','response_loss','smooth_response_weight')])
 return d
