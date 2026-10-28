"""Independent cross-run replay arithmetic; same point is not equal observables."""
from core189 import *
NATIVE_OBS_TOL=2e-4
NATIVE_KL_TOL=5e-6
FP64_TOL=2e-9

def native_pair(old,new):
 x=farray(np.asarray(old['observed'],np.float64),(22,));y=farray(np.asarray(new['observed'],np.float64),(22,));gap=float(np.max(abs(y-x)));kl=abs(float(new['KL'])-float(old['KL']));need(math.isfinite(kl),'Finite replay KL');gates=old['gates']==new['gates'];point=old['theta_sha256']==new['theta_sha256']
 result=dict(observed_error_Linf=gap,KL_absolute_error=kl,gate_identity=gates,same_point_identity=point,within_replay_tolerance=bool(gap<=NATIVE_OBS_TOL and kl<=NATIVE_KL_TOL and gates))
 return result,dict(archived_parent_response_Linf=float(max(abs(x[:6]))),current_replay_response_Linf=float(max(abs(y[:6]))),signed_response_max_shift=float(max(abs(y[:6]))-max(abs(x[:6]))),response_coordinate_gap_Linf=float(max(abs(y[:6]-x[:6]))),all22_signed_observable_differences=(y-x).tolist(),observables_exactly_equal=bool(np.array_equal(x,y)))

def replay_report(a,ctx,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);nn=a['native'];old=ctx['C186_final_native'];initial=a['frames'][0];prev=ctx['C186_final_FP64'];compare([r['state_id'] for r in initial],IDS);compare([r['state_id'] for r in prev],IDS);rows=[];fp=[];maxima={}
 for rnd in (0,1):
  b=old[str(rnd)];e=nn['native_initial'+str(rnd)];compare([r['state_id'] for r in b],IDS);compare([r['state_id'] for r in e],IDS)
  final=nn['native_final'+str(rnd)];compare([r['state_id'] for r in final],IDS)
  for sid,x,y in zip(IDS,b,e):
   replay,diag=native_pair(x['metrics'],y['metrics']);compare(y['C186_final_native_replay'],replay,'Independent registered C186/native replay/'+sid);need(replay['within_replay_tolerance'] and replay['same_point_identity'],'Original native replay rule failed/'+sid)
   rows.append(dict(round=rnd,state_id=sid,**replay,**diag,parent_complete_pass=x['qualification']['complete_pass'],current_replay_complete_pass=y['qualification']['complete_pass'],complete_pass_same=x['qualification']['complete_pass']==y['qualification']['complete_pass']))
  om=max(max(abs(np.asarray(r['metrics']['observed'])[:6])) for r in b);im=max(max(abs(np.asarray(r['metrics']['observed'])[:6])) for r in e);fm=max(max(abs(np.asarray(r['metrics']['observed'])[:6])) for r in final);maxima[str(rnd)]=dict(parent_C186_final_maximum_response=float(om),C188_same_run_initial_maximum_response=float(im),C188_same_run_final_maximum_response=float(fm),cross_run_anchor_maximum_shift=float(im-om),same_run_training_maximum_change=float(fm-im),same_run_training_is_primary_comparison=True)
 for sid,x,y in zip(IDS,prev,initial):
  xo=farray(np.asarray(x['observed'],np.float64),(22,));yo=farray(np.asarray(y['observed'],np.float64),(22,));gap=float(max(abs(yo-xo)));ke=abs(float(y['KL'])-float(x['KL']));need(math.isfinite(ke),'Finite FP64 replay KL');r=dict(observed_Linf=gap,KL_absolute=ke,within_original_replay_tolerance=bool(gap<=FP64_TOL and ke<=FP64_TOL));compare(y['C186_final_FP64_replay'],r,'Independent registered C186/FP64 replay/'+sid);need(r['within_original_replay_tolerance'],'Original FP64 replay rule failed/'+sid)
  fp.append(dict(state_id=sid,**r,all22_signed_observable_differences=(yo-xo).tolist(),observables_exactly_equal=bool(np.array_equal(xo,yo)),same_archived_FP64_point_identity=(x['FP64_point_sha256']==y['FP64_point_sha256']) if 'FP64_point_sha256' in x and 'FP64_point_sha256' in y else None,point_identity_is_reported_without_adding_a_replay_gate=True))
 compare(len(rows),168);compare(len(fp),84);summary=dict(native_pairs=168,FP64_pairs=84,all_native_same_point_identity=all(r['same_point_identity'] for r in rows),all_native_observables_exactly_equal=all(r['observables_exactly_equal'] for r in rows),native_pairs_with_nonzero_observable_difference=sum(not r['observables_exactly_equal'] for r in rows),maximum_native_all22_observable_gap_Linf=max(r['observed_error_Linf'] for r in rows),maximum_native_response_coordinate_gap_Linf=max(r['response_coordinate_gap_Linf'] for r in rows),maximum_native_KL_gap=max(r['KL_absolute_error'] for r in rows),all_native_registered_replay_rules_passed=all(r['within_replay_tolerance'] and r['same_point_identity'] for r in rows),native_full_pass_disagreement_ids=[dict(round=r['round'],state_id=r['state_id']) for r in rows if not r['complete_pass_same']],maximum_FP64_all22_observable_gap_Linf=max(r['observed_Linf'] for r in fp),maximum_FP64_KL_gap=max(r['KL_absolute'] for r in fp),all_FP64_registered_replay_rules_passed=all(r['within_original_replay_tolerance'] for r in fp),maxima_by_round=maxima,registered_tolerances=dict(native_all22_observable_Linf=NATIVE_OBS_TOL,native_KL_absolute=NATIVE_KL_TOL,FP64_all22_observable_Linf=FP64_TOL,FP64_KL_absolute=FP64_TOL,native_metric_gates_equal_required=True,native_point_identity_required=True),acceptance_thresholds_changed=False,replay_rule_changed=False,no_recentring_or_calibration=True,new_model_forwards=0,measurement_accuracy_or_hardware_cause_established=False,no_accuracy_bound_claim=True,LC2_closed=False)
 dump(out/'ALL168_NATIVE_ANCHOR_REPLAY_PAIRS.json',rows);dump(out/'ALL84_FP64_ANCHOR_REPLAY_PAIRS.json',fp);dump(out/'REPLAY_SUMMARY.json',summary)
 import csv
 keys=['round','state_id','observed_error_Linf','KL_absolute_error','gate_identity','same_point_identity','within_replay_tolerance','observables_exactly_equal','archived_parent_response_Linf','current_replay_response_Linf','signed_response_max_shift','response_coordinate_gap_Linf','complete_pass_same']
 with (out/'ALL168_NATIVE_ANCHOR_REPLAY_PAIRS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(keys);w.writerows([[r[k] for k in keys] for r in rows])
 return summary
