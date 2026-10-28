"""Sixteen-original read-only C170 update and trajectory audit."""
import argparse,resource,traceback
from core171 import *
from audit171 import audit
from diagnostics171 import diagnostics
def run(paths,out,c=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';s=dict(protocol=PROTOCOL,complete=False,decision=INCOMPLETE,synthetic_fixture=not c['production'],C170_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=SCOPE,**no_compute())
 def budget():need(time.monotonic()-started<TIMEOUT,'C171 fixed30-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly sixteen original ZIPs')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
  for n,h in c['source170'].items():compare(sha(ROOT/'vendor170'/n),h,'Trusted delivered C170 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage='fifteen_original_binding';checkpoint()
  with archive155.prior.no_models(),safe.no_solvers():
   if input_loader is None:
    from inputs170 import load
    ctx=load({k:paths[k] for k in prior.INPUT_KEYS},out/'fifteen_original_binding',c['parent'],budget)
   else:ctx=input_loader(paths,out,c,budget)
   stage='eight_updates_recorded_certificates_and_all_native_gates';checkpoint();a=audit(paths['C170'],ctx,out/'C170_arithmetic',c,budget);budget();stage='all84_nine_frame_trajectory_diagnostics';d=diagnostics(a,out/'diagnostics');budget()
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub','scipy')),'C171 must not import model or solver libraries');s.update(complete=True,decision=DECISION,parent_protocol=prior.PROTOCOL,parent_decision=a['summary']['decision'],C170_zip_sha256=c['pins']['C170']['sha256'],members_hashed=a['members_hashed'],all_sixteen_originals_independently_bound=input_loader is None,training_states_audited=84,training_updates_audited=8,gradient_records_audited=672,ordered_records_audited=1092,recorded_LP_calls_reconstructed=a['LPcalls_reconstructed'],exact_recorded_direction_checks_reconstructed=8,actual_FP64_point_arrays_rehashed=756,native_FP32_point_arrays_rehashed=168,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],initial_head_digest=parameter_hash(a['heads'][0]),fixed_final_step8_head_digest=parameter_hash(a['heads'][-1]),native_summaries=a['native_summaries'],paired_by_round=a['paired'],initial_FP64_objective=frame_objective(a['frames'][0]),final_FP64_objective=frame_objective(a['frames'][-1]),diagnostics=d);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C171：C170八步LP训练只读审计\n\n'+json.dumps(dict(parent_decision=s['parent_decision'],native_summaries=s['native_summaries'],frame_statistics=d['all9_frame_statistics'],step_statistics=d['all8_step_statistics'],initial_failure_trajectories=d['initial_failure_state_trajectories']),ensure_ascii=False,indent=2)+'\n\n保留全部84态、8步、固定第8步头、失败态与两轮原生门槛。不选择中间头、不新求解或训练。不独立复核导数/测量/KL真值，也不宣称最优性、可行/不可行性、曲率界、收敛、数值因果或泛化。局部趋势不外推闭合所需步数。LC2仍开放。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  checks={}
  for k,p in paths.items():
   try:checks[k]=sha(p)==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):s['complete']=False;errors.append('Original input missing or changed')
  if not s['complete']:s['decision']=INCOMPLETE
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
