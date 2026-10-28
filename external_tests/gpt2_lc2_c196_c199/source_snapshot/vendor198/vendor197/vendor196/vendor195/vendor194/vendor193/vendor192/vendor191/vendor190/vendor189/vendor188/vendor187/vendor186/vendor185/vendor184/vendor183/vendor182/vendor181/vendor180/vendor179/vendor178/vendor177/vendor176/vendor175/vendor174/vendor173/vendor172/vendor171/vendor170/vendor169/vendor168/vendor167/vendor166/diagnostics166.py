"""Full-state descriptive directional, finite-step and precision accounting."""
from core166 import *
def diagnostics(data,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);fd=data['fd'];hstats={};coord=[];precision=[];anchor=data['gradient_records'];probes=data['probe_records'];native=data['native_records']
 for h in H:
  group=[r for r in fd if r['h']==h];compare(len(group),84);hstats[str(h)]={}
  for name in ('response','guard'):
   rr=[r['checks'][name] for r in group];err=np.array([r['absolute_difference'] for r in rr]);ad=np.array([r['AD_directional'] for r in rr]);cd=np.array([r['central_difference'] for r in rr]);hstats[str(h)][name]=dict(n=84,maximum_absolute_AD_FD_difference=float(max(err)),mean_absolute_AD_FD_difference=float(np.mean(err)),maximum_absolute_AD_directional=float(max(abs(ad))),maximum_absolute_central_difference=float(max(abs(cd))),opposite_nonzero_sign_count=int(np.count_nonzero(ad*cd<0)),all_values_descriptive_no_derivative_accuracy_pass_gate=True)
 for label,h in zip(('1e-6','5e-6','1e-5'),H):
  for i,sid in enumerate(IDS):
   a=np.asarray(probes['minus_'+label][i]['observed']);b=np.asarray(probes['plus_'+label][i]['observed']);base=np.asarray(anchor[i]['observed']);central=(b[:6]-a[:6])/(2*h)
   for j in range(6):coord.append(dict(state_id=sid,h=h,coordinate=j,anchor_response=float(base[j]),minus_response=float(a[j]),plus_response=float(b[j]),response_coordinate_central_difference=float(central[j]),coordinate_AD_not_evaluated=True))
 all_frames=dict(anchor=anchor,**probes);frame_stats={}
 for arm,rr in all_frames.items():
  for t in (0,1):
   for a,b in zip(rr,native[(arm,t)]):
    fo=np.asarray(a['observed']);no=np.asarray(b['metrics']['observed']);diff=no[:6]-fo[:6];precision.append(dict(arm=arm,round=t,state_id=a['state_id'],FP64_response=fo[:6].tolist(),native_response=no[:6].tolist(),FP64_signed_response_slack=(.002-abs(fo[:6])).tolist(),native_signed_response_slack=(.002-abs(no[:6])).tolist(),native_minus_FP64_response_Linf=float(max(abs(diff))),all22_observable_gap_Linf=float(max(abs(no-fo))),native_failed_gates=b['qualification']['failed_gates'],FP64_full_acceptance_claim=False))
  frame_stats[arm]=dict(maximum_FP64_response=max(float(max(abs(np.asarray(a['observed'])[:6]))) for a in rr),FP64_response_only_passed=int(sum(max(abs(np.asarray(a['observed'])[:6]))<=.002 for a in rr)),native_by_round=data['diagnostics']['native_summaries'][arm],paired_vs_anchor=data['diagnostics']['paired_vs_anchor'].get(arm),FP64_full_acceptance_claim=False)
 summary=dict(direction=data['direction'],all3_h_AD_FD_descriptive_statistics=hstats,all7_frame_statistics=frame_stats,all84_contractions_retained=True,all252_scalar_difference_rows_retained=True,all1512_response_coordinate_FD_rows_retained=True,all1176_precision_comparisons_retained=True,no_new_threshold_or_probe_selection=True,recorded_gradient_truth_independently_verified=False,causal_explanation_established=False,convergence_or_infeasibility_certified=False)
 compare(len(coord),1512);compare(len(precision),1176);dump(out/'ALL1512_RESPONSE_COORDINATE_DIFFERENCES.json',coord);dump(out/'ALL1176_RESPONSE_PRECISION_COMPARISONS.json',precision);dump(out/'DIAGNOSTIC_SUMMARY.json',summary)
 import csv
 for name,rr,keys in [('ALL84_DIRECTION_CONTRACTIONS.csv',data['contractions'],['state_id','response_loss','active','projected_gradient_L2','normalized_response_gradient_dot_direction','response_gradient_dot_direction','guard_gradient_dot_direction','fails_original_strict_common_descent_threshold']),('ALL1512_RESPONSE_COORDINATE_DIFFERENCES.csv',coord,['state_id','h','coordinate','anchor_response','minus_response','plus_response','response_coordinate_central_difference'])]:
  with (out/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.writer(f);w.writerow(keys)
   for r in rr:w.writerow([r[k] for k in keys])
 return summary
