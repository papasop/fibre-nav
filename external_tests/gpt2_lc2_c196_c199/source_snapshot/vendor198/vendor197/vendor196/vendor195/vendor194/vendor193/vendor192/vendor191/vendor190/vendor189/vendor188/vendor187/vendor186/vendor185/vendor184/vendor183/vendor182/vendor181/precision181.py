"""Independent endpoint arithmetic, including original JSON and CSV binding."""
from core181 import *
import csv,io
CSV_KEYS=['head','round','state_id','native_response_Linf','FP64_response_Linf','native_response_slack','FP64_response_slack','native_full_pass','native_response_only_pass','FP64_response_only_pass','response_gate_disagrees','response_coordinate_gap_Linf']
def reconstruct(native,initial,final):
 rows=[];by_head={}
 for label,ff in [('initial',initial),('final',final)]:
  compare([r['state_id'] for r in ff],IDS,'Full ordered FP64 endpoint registry');per_round={}
  for rnd in (0,1):
   nn=native['native_'+label+str(rnd)];compare([r['state_id'] for r in nn],IDS,'Full ordered native endpoint registry');local=[]
   for f,n,sid in zip(ff,nn,IDS):
    x=farray(np.asarray(n['metrics']['observed'],np.float64),(22,));y=farray(np.asarray(f['observed'],np.float64),(22,));nx=max(abs(float(v)) for v in x[:6]);fy=max(abs(float(v)) for v in y[:6]);np_=nx<=.002;fp_=fy<=.002;q=n['qualification'];need(type(q['complete_pass']) is bool,'Strict native boolean qualification')
    r=dict(head=label,round=rnd,state_id=sid,native_response=x[:6].tolist(),FP64_response=y[:6].tolist(),native_response_Linf=nx,FP64_response_Linf=fy,native_response_slack=.002-nx,FP64_response_slack=.002-fy,native_response_only_pass=np_,FP64_response_only_pass=fp_,native_full_pass=q['complete_pass'],native_failed_gates=q['failed_gates'],response_gate_disagrees=np_!=fp_,native_full_pass_and_FP64_response_pass=q['complete_pass'] and fp_,response_coordinate_gap_Linf=max(abs(float(a)-float(b)) for a,b in zip(x[:6],y[:6])),full22_observable_gap_Linf=max(abs(float(a)-float(b)) for a,b in zip(x,y)),FP64_full_acceptance_evaluated=False,descriptive_only=True,thresholds_changed=False);rows.append(r);local.append(r)
   per_round[str(rnd)]=dict(n=84,native_full_passed=sum(r['native_full_pass'] for r in local),native_response_only_passed=sum(r['native_response_only_pass'] for r in local),FP64_response_only_passed=sum(r['FP64_response_only_pass'] for r in local),native_full_pass_and_FP64_response_pass=sum(r['native_full_pass_and_FP64_response_pass'] for r in local),response_gate_disagreements=sum(r['response_gate_disagrees'] for r in local),disagreement_state_ids=[r['state_id'] for r in local if r['response_gate_disagrees']],native_response_failed_ids=[r['state_id'] for r in local if not r['native_response_only_pass']],FP64_response_failed_ids=[r['state_id'] for r in local if not r['FP64_response_only_pass']],maximum_native_response=max(r['native_response_Linf'] for r in local),maximum_FP64_response=max(r['FP64_response_Linf'] for r in local),minimum_native_response_slack=min(r['native_response_slack'] for r in local),minimum_FP64_response_slack=min(r['FP64_response_slack'] for r in local))
  by_head[label]=per_round
 compare(len(rows),336)
 summary=dict(rows=336,by_head=by_head,maximum_observed_response_coordinate_gap_Linf=max(r['response_coordinate_gap_Linf'] for r in rows),response_budget=.002,native_full_acceptance_unchanged=True,FP64_full_acceptance_evaluated=False,FP64_response_only_not_an_added_gate=True,additional_model_forwards=0,no_accuracy_or_error_bound_claim=True,thresholds_changed=False)
 return rows,summary
def bind_csv(data,rows):
 old=list(csv.reader(io.StringIO(data.decode('utf-8'))));compare(old[0],CSV_KEYS,'Original precision CSV header');compare(len(old),337,'Original precision CSV complete rows')
 for cells,r in zip(old[1:],rows):
  compare(len(cells),len(CSV_KEYS),'Original precision CSV columns')
  for k,value in zip(CSV_KEYS,cells):
   want=r[k]
   if isinstance(want,bool):compare(value,str(want),'Original CSV boolean/'+k)
   elif isinstance(want,int):compare(value,str(want),'Original CSV integer/'+k)
   elif isinstance(want,float):
    got=float(value);need(math.isfinite(got),'Finite original CSV');compare(got,want,'Independent precision CSV/'+k,atol=1e-15,rtol=1e-12)
   else:compare(value,want,'Original CSV identity/'+k)
def audit_precision(z,parent_summary,native,initial,final,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);rows,summary=reconstruct(native,initial,final)
 compare(z.json('ALL336_ENDPOINT_PRECISION_MARGINS.json'),rows,'Independent complete336 precision rows',atol=1e-15,rtol=1e-12);compare(z.json('ENDPOINT_PRECISION_SUMMARY.json'),summary,'Independent original precision summary',atol=1e-15,rtol=1e-12);compare(parent_summary['endpoint_precision_margins'],summary,'Original main precision duplicate',atol=1e-15,rtol=1e-12);bind_csv(z.raw('ALL336_ENDPOINT_PRECISION_MARGINS.csv'),rows)
 transitions=[]
 for rnd in (0,1):
  before=[r for r in rows if r['head']=='initial' and r['round']==rnd];after=[r for r in rows if r['head']=='final' and r['round']==rnd]
  for x,y in zip(before,after):
   transitions.append(dict(round=rnd,state_id=x['state_id'],initial_response_disagreement=x['response_gate_disagrees'],final_response_disagreement=y['response_gate_disagrees'],disagreement_resolved_to_both_response_pass=bool(x['response_gate_disagrees'] and y['native_response_only_pass'] and y['FP64_response_only_pass']),disagreement_resolved_to_both_response_fail=bool(x['response_gate_disagrees'] and not y['native_response_only_pass'] and not y['FP64_response_only_pass']),new_response_disagreement=bool(not x['response_gate_disagrees'] and y['response_gate_disagrees']),initial_native_full_pass=x['native_full_pass'],final_native_full_pass=y['native_full_pass'],initial_both_native_full_and_FP64_response_pass=x['native_full_pass_and_FP64_response_pass'],final_both_native_full_and_FP64_response_pass=y['native_full_pass_and_FP64_response_pass'],native_rescued=bool(not x['native_full_pass'] and y['native_full_pass']),native_destroyed=bool(x['native_full_pass'] and not y['native_full_pass']),FP64_full_acceptance_evaluated=False))
 dump(out/'ALL336_PRECISION_ROWS_INDEPENDENT.json',rows);dump(out/'PRECISION_SUMMARY_INDEPENDENT.json',summary);dump(out/'ALL168_PRECISION_TRANSITIONS.json',transitions)
 with (out/'ALL336_PRECISION_ROWS_INDEPENDENT.csv').open('w',newline='',encoding='utf-8') as f:w=csv.writer(f);w.writerow(CSV_KEYS);w.writerows([[r[k] for k in CSV_KEYS] for r in rows])
 report=dict(original_precision_rows_independently_reconstructed=336,original_JSON_CSV_and_both_summary_copies_bound=True,original_precision_summary=summary,transition_rows=168,by_round={str(t):dict(resolved_to_both_response_pass_ids=[r['state_id'] for r in transitions if r['round']==t and r['disagreement_resolved_to_both_response_pass']],resolved_to_both_response_fail_ids=[r['state_id'] for r in transitions if r['round']==t and r['disagreement_resolved_to_both_response_fail']],new_response_disagreement_ids=[r['state_id'] for r in transitions if r['round']==t and r['new_response_disagreement']]) for t in (0,1)},FP64_full_acceptance_evaluated=False,precision_agreement_is_not_an_accuracy_bound=True,no_new_model_measurements=True,thresholds_changed=False)
 dump(out/'PRECISION_AUDIT_REPORT.json',report);return report
