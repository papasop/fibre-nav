"""Descriptive precision margins from existing endpoints; no additional gate."""
from core188 import *
def precision_margins(native,initial_FP64,final_FP64,out):
 out=Path(out);rows=[];by_head={}
 for label,rr in [('initial',initial_FP64),('final',final_FP64)]:
  compare([r['state_id'] for r in rr],IDS,'All84 FP64 precision rows');per_round={}
  for rnd in (0,1):
   nn=native['native_'+label+str(rnd)];compare([r['state_id'] for r in nn],IDS,'All84 native precision rows');local=[]
   for i,sid in enumerate(IDS):
    fo=farray(np.asarray(rr[i]['observed'],np.float64),(22,));no=farray(np.asarray(nn[i]['metrics']['observed'],np.float64),(22,));fm=float(np.max(np.abs(fo[:6])));nm=float(np.max(np.abs(no[:6])));np_=nm<=.002;fp_=fm<=.002;qual=nn[i]['qualification'];need(type(qual['complete_pass']) is bool,'Boolean native full gate')
    r=dict(head=label,round=rnd,state_id=sid,native_response=no[:6].tolist(),FP64_response=fo[:6].tolist(),native_response_Linf=nm,FP64_response_Linf=fm,native_response_slack=.002-nm,FP64_response_slack=.002-fm,native_response_only_pass=np_,FP64_response_only_pass=fp_,native_full_pass=qual['complete_pass'],native_failed_gates=qual['failed_gates'],response_gate_disagrees=np_!=fp_,native_full_pass_and_FP64_response_pass=qual['complete_pass'] and fp_,response_coordinate_gap_Linf=float(np.max(abs(no[:6]-fo[:6]))),full22_observable_gap_Linf=float(np.max(abs(no-fo))),FP64_full_acceptance_evaluated=False,descriptive_only=True,thresholds_changed=False);rows.append(r);local.append(r)
   per_round[str(rnd)]=dict(n=84,native_full_passed=sum(r['native_full_pass'] for r in local),native_response_only_passed=sum(r['native_response_only_pass'] for r in local),FP64_response_only_passed=sum(r['FP64_response_only_pass'] for r in local),native_full_pass_and_FP64_response_pass=sum(r['native_full_pass_and_FP64_response_pass'] for r in local),response_gate_disagreements=sum(r['response_gate_disagrees'] for r in local),disagreement_state_ids=[r['state_id'] for r in local if r['response_gate_disagrees']],native_response_failed_ids=[r['state_id'] for r in local if not r['native_response_only_pass']],FP64_response_failed_ids=[r['state_id'] for r in local if not r['FP64_response_only_pass']],maximum_native_response=max(r['native_response_Linf'] for r in local),maximum_FP64_response=max(r['FP64_response_Linf'] for r in local),minimum_native_response_slack=min(r['native_response_slack'] for r in local),minimum_FP64_response_slack=min(r['FP64_response_slack'] for r in local))
  by_head[label]=per_round
 compare(len(rows),336);summary=dict(rows=336,by_head=by_head,maximum_observed_response_coordinate_gap_Linf=max(r['response_coordinate_gap_Linf'] for r in rows),response_budget=.002,native_full_acceptance_unchanged=True,FP64_full_acceptance_evaluated=False,FP64_response_only_not_an_added_gate=True,additional_model_forwards=0,no_accuracy_or_error_bound_claim=True,thresholds_changed=False)
 dump(out/'ALL336_ENDPOINT_PRECISION_MARGINS.json',rows);dump(out/'ENDPOINT_PRECISION_SUMMARY.json',summary)
 import csv
 keys=['head','round','state_id','native_response_Linf','FP64_response_Linf','native_response_slack','FP64_response_slack','native_full_pass','native_response_only_pass','FP64_response_only_pass','response_gate_disagrees','response_coordinate_gap_Linf']
 with (out/'ALL336_ENDPOINT_PRECISION_MARGINS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(keys);w.writerows([[r[k] for k in keys] for r in rows])
 return summary
