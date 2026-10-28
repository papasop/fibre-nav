"""All84 endpoint accounting; dynamic rescue/failure focus, no new measurements."""
from core187 import *
def focus_report(a,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);frames=a['frames'];native=a['native'];compare(len(frames),9,'All nine FP64 frames')
 for rr in frames:compare([r['state_id'] for r in rr],IDS,'Full ordered FP64 focus registry')
 for name in ('native_initial0','native_initial1','native_final0','native_final1'):compare([r['state_id'] for r in native[name]],IDS,'Full ordered native focus registry')
 rows=[];focus=[];counts={str(r):dict(initial_passed=0,final_passed=0,rescued=0,destroyed=0,both_passed=0,both_failed=0) for r in (0,1)}
 for i,sid in enumerate(IDS):
  drift=[float(max(abs(farray(np.asarray(rr[i]['observed'],np.float64),(22,))[:6]))) for rr in frames];rp=[v<=.002 for v in drift];sp=[v<=.0018 for v in drift];native_rows=[];outcomes=[]
  for rnd in (0,1):
   b=native['native_initial'+str(rnd)][i];e=native['native_final'+str(rnd)][i]
   for rr in (b,e):need(type(rr['qualification']['complete_pass']) is bool,'Boolean full native gate')
   bp=b['qualification']['complete_pass'];ep=e['qualification']['complete_pass'];outcome='both_passed' if bp and ep else 'rescued' if ep else 'destroyed' if bp else 'both_failed';outcomes.append(outcome);c=counts[str(rnd)];c['initial_passed']+=bp;c['final_passed']+=ep;c[outcome]+=1
   native_rows.append(dict(round=rnd,initial_full_pass=bp,final_full_pass=ep,outcome=outcome,initial_record=b,final_record=e))
  compare(outcomes[0],outcomes[1],'Full native focus outcome repeated/'+sid)
  entry=[t for t in range(1,9) if not rp[t-1] and rp[t]];exit_=[t for t in range(1,9) if rp[t-1] and not rp[t]]
  row=dict(state_id=sid,dataset_role='exposed_training',native_outcome_both_rounds=outcomes[0],FP64_response_Linf_by_frame=drift,FP64_response_slack_by_frame=[.002-v for v in drift],FP64_safety_slack_by_frame=[.0018-v for v in drift],FP64_response_only_pass_by_frame=rp,FP64_training_safety_by_frame=sp,FP64_response_entry_steps=entry,FP64_response_exit_steps=exit_,FP64_safety_entry_steps=[t for t in range(1,9) if not sp[t-1] and sp[t]],FP64_safety_exit_steps=[t for t in range(1,9) if sp[t-1] and not sp[t]],initial_FP64_response_only_pass=rp[0],final_FP64_response_only_pass=rp[-1],initial_response_budget_excess=max(drift[0]-.002,0.),final_response_budget_excess=max(drift[-1]-.002,0.),full_native_records_both_rounds=native_rows,intermediate_frames_are_FP64_response_only=True,FP64_full_acceptance_evaluated=False)
  rows.append(row)
  if outcomes[0]!='both_passed':focus.append(row)
 rescued=[r['state_id'] for r in rows if r['native_outcome_both_rounds']=='rescued'];destroyed=[r['state_id'] for r in rows if r['native_outcome_both_rounds']=='destroyed'];failed=[r['state_id'] for r in rows if r['native_outcome_both_rounds'] in ('both_failed','destroyed')]
 summary=dict(training_states_retained=84,FP64_frames=9,native_endpoint_records=336,native_outcomes_by_round=counts,focus_membership='Union of initial or final native complete failures; dynamic, not a preselected subset.',focus_state_ids=[r['state_id'] for r in focus],rescued_ids_both_rounds=rescued,destroyed_ids_both_rounds=destroyed,remaining_native_failed_ids_both_rounds=failed,rescued_and_final_FP64_response_pass_ids=[r['state_id'] for r in rows if r['state_id'] in rescued and r['final_FP64_response_only_pass']],rescued_but_final_FP64_response_fail_ids=[r['state_id'] for r in rows if r['state_id'] in rescued and not r['final_FP64_response_only_pass']],remaining_native_failed_but_FP64_response_pass_ids=[r['state_id'] for r in rows if r['state_id'] in failed and r['final_FP64_response_only_pass']],response_budget=.002,training_safety_budget=.0018,intermediate_FP64_response_crossing_is_not_native_full_qualification=True,native_gate_observables_not_independently_measured=True,no_future_steps_or_closure_time_extrapolation=True,new_model_forwards=0,FP64_full_acceptance_evaluated=False,LC2_closed=False)
 compare(counts['0'],counts['1'],'All native focus counts repeated')
 dump(out/'ALL84_ENDPOINT_AND_RESPONSE_TRAJECTORIES.json',rows);dump(out/'RESCUED_DESTROYED_AND_REMAINING_FAILURES.json',focus);dump(out/'FOCUS_SUMMARY.json',summary)
 import csv
 with (out/'ALL84_ENDPOINT_ACCOUNTING.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['state_id','native_outcome_both_rounds','initial_FP64_response_Linf','final_FP64_response_Linf','initial_FP64_response_slack','final_FP64_response_slack','final_FP64_response_only_pass'])
  for r in rows:w.writerow([r['state_id'],r['native_outcome_both_rounds'],r['FP64_response_Linf_by_frame'][0],r['FP64_response_Linf_by_frame'][-1],r['FP64_response_slack_by_frame'][0],r['FP64_response_slack_by_frame'][-1],r['final_FP64_response_only_pass']])
 return summary
