"""Full state activity histories; training safety is distinct from acceptance."""
from core183 import *
import csv
def activity_diagnostics(a,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);frames=a['frames'];compare(len(frames),9,'Nine complete activity frames');flags=[];drifts=[];loss=[]
 for ff in frames:
  compare([r['state_id'] for r in ff],IDS,'All84 activity frame identities');lv=[];dv=[]
  for r in ff:
   value=float(r['objective_parts']['response_loss']);need(math.isfinite(value) and value>=0,'Finite nonnegative activity loss');obs=farray(np.asarray(r['observed'],np.float64),(22,));lv.append(value);dv.append(float(max(abs(obs[:6]))))
  loss.append(lv);drifts.append(dv);flags.append([v>0 for v in lv])
 for t in range(8):compare(np.flatnonzero(flags[t]).tolist(),np.asarray(a['problems'][t]['active']).tolist(),'Archived LP activity registry from recomputed losses')
 for arm in ('initial','final'):
  for rnd in (0,1):compare([r['state_id'] for r in a['native']['native_'+arm+str(rnd)]],IDS,'Activity native endpoint pairing')
 ever=flags[0].copy();rows=[];steps=[]
 for t in range(8):
  rr=[]
  for i,sid in enumerate(IDS):
   before=flags[t][i];after=flags[t+1][i];entered=not before and after;left=before and not after;reentered=entered and ever[i];r=dict(step=t+1,state_id=sid,active_before=before,active_after=after,left_active_set=left,entered_active_set=entered,reentered_active_set=reentered,response_loss_before=loss[t][i],response_loss_after=loss[t+1][i],FP64_response_Linf_before=drifts[t][i],FP64_response_Linf_after=drifts[t+1][i],FP64_training_safety_slack_after=SAFETY-drifts[t+1][i],FP64_response_budget_slack_after=.002-drifts[t+1][i],entered_training_safety_budget=drifts[t][i]>SAFETY and drifts[t+1][i]<=SAFETY,left_training_safety_budget=drifts[t][i]<=SAFETY and drifts[t+1][i]>SAFETY,crossed_into_FP64_response_budget=drifts[t][i]>.002 and drifts[t+1][i]<=.002,crossed_out_of_FP64_response_budget=drifts[t][i]<=.002 and drifts[t+1][i]>.002,intermediate_native_full_acceptance_evaluated=False,FP64_full_acceptance_evaluated=False);ever[i]=ever[i] or after;rr.append(r);rows.append(r)
  steps.append(dict(step=t+1,active_before=sum(flags[t]),active_after=sum(flags[t+1]),left_active_ids=[r['state_id'] for r in rr if r['left_active_set']],entered_active_ids=[r['state_id'] for r in rr if r['entered_active_set']],reentered_active_ids=[r['state_id'] for r in rr if r['reentered_active_set']],entered_safety_ids=[r['state_id'] for r in rr if r['entered_training_safety_budget']],left_safety_ids=[r['state_id'] for r in rr if r['left_training_safety_budget']],crossed_into_FP64_response_budget_ids=[r['state_id'] for r in rr if r['crossed_into_FP64_response_budget']],crossed_out_of_FP64_response_budget_ids=[r['state_id'] for r in rr if r['crossed_out_of_FP64_response_budget']]))
 states=[]
 for i,sid in enumerate(IDS):
  rr=[r for r in rows if r['state_id']==sid];native={str(t):dict(initial_full_pass=a['native']['native_initial'+str(t)][i]['qualification']['complete_pass'],final_full_pass=a['native']['native_final'+str(t)][i]['qualification']['complete_pass']) for t in (0,1)}
  states.append(dict(state_id=sid,active_by_frame=[f[i] for f in flags],response_loss_by_frame=[f[i] for f in loss],FP64_response_Linf_by_frame=[f[i] for f in drifts],left_active_steps=[r['step'] for r in rr if r['left_active_set']],entered_active_steps=[r['step'] for r in rr if r['entered_active_set']],reentered_active_steps=[r['step'] for r in rr if r['reentered_active_set']],native_full_endpoints_by_round=native,activity_exit_does_not_imply_native_rescue=True))
 compare(len(rows),672);compare(len(states),84);dump(out/'ALL672_ACTIVITY_TRANSITIONS.json',rows);dump(out/'ALL84_ACTIVITY_TRAJECTORIES.json',states);dump(out/'ALL8_ACTIVITY_STEP_SUMMARIES.json',steps)
 with (out/'ALL672_ACTIVITY_TRANSITIONS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);keys=list(rows[0]);w.writerow(keys);w.writerows([[r[k] for k in keys] for r in rows])
 initial={sid for sid,flag in zip(IDS,flags[0]) if flag};final={sid for sid,flag in zip(IDS,flags[-1]) if flag};changed={r['state_id'] for r in rows if r['left_active_set'] or r['entered_active_set']};summary=dict(activity_transition_rows=672,state_trajectories=84,initial_active_count=len(initial),final_active_count=len(final),active_counts_by_frame=[sum(f) for f in flags],initial_active_ids=[sid for sid in IDS if sid in initial],final_active_ids=[sid for sid in IDS if sid in final],initial_active_but_final_inactive_ids=[sid for sid in IDS if sid in initial-final],initial_inactive_but_final_active_ids=[sid for sid in IDS if sid in final-initial],step_summaries=steps,states_with_activity_changes=[r for r in states if r['state_id'] in changed],all84_states_retained=True,training_safety_budget=SAFETY,original_response_budget=.002,training_safety_not_a_new_acceptance_gate=True,no_monotonicity_or_six_to_five_assumption=True,FP64_full_acceptance_evaluated=False,no_new_model_measurements=True,no_future_step_or_convergence_claim=True)
 dump(out/'ACTIVITY_SET_SUMMARY.json',summary);return summary
