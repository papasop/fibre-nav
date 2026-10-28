"""Twenty-original NumPy-only C174 arithmetic and coordinate audit."""
import argparse,resource,traceback
from core175 import *
from audit175 import audit
from recompute175 import supplemental
def run(paths,out,c=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';s=dict(protocol=PROTOCOL,complete=False,decision=INCOMPLETE,synthetic_fixture=not c['production'],C174_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=SCOPE,**no_compute())
 def budget():need(time.monotonic()-started<TIMEOUT,'C175 fixed60-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly20 original ZIPs')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
  for n,h in c['source174'].items():compare(sha(ROOT/'vendor174'/n),h,'Trusted delivered C174 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage='nineteen_original_binding';checkpoint()
  with archive155.prior.no_models(),safe.no_solvers():
   if input_loader is None:
    from inputs174 import load
    ctx=load({k:paths[k] for k in prior.INPUT_KEYS},out/'nineteen_original_binding',c['parent'],budget)
   else:ctx=input_loader(paths,out,c,budget)
   stage='C174_full_coordinate_certificates_points_and_gates';checkpoint();a=audit(paths['C174'],ctx,out/'C174_arithmetic',c,budget);budget();stage='all84_all6_coordinate_AD_and_native_diagnostics';d=supplemental(a,out/'diagnostics');budget()
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub','scipy')),'C175 must not import model or solver libraries');s.update(complete=True,decision=DECISION,parent_protocol=prior.PROTOCOL,parent_decision=a['summary']['decision'],C174_zip_sha256=c['pins']['C174']['sha256'],members_hashed=a['members_hashed'],all_twenty_originals_independently_bound=input_loader is None,training_states_audited=84,temporary_probe_heads_audited=6,ordered_records_audited=1848,gradient_records_audited=84,recorded_LP_calls_reconstructed=a['solver']['recorded_LP_calls'],original_solver_internal_linear_system_count=None,exact_recorded_direction_check_reconstructed=True,exact_actual_head_certificates_reconstructed=6,exact_affine_certificate_signed_rows_reconstructed=7056,archived_response_Jacobian_blocks_checked=84,archived_response_coordinate_gradient_rows_checked=504,LP_signed_rows_reconstructed=1008,response_coordinate_affine_rows_recomputed=3024,scalar_gradient_chain_checks_recomputed=84,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_differences_recomputed=252,signed_scalar_loss_changes_recomputed=504,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],original_C170_fixed_final_head_digest=parameter_hash(a['head']),original_positive_steps=list(H),parent_diagnostics=a['diagnostics'],coordinate_diagnostics=d);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C175：C174全响应坐标方向与有限步只读审计\n\n'+json.dumps(dict(parent_decision=s['parent_decision'],native_summaries=a['diagnostics']['native_summaries'],coordinates=d),ensure_ascii=False,indent=2)+'\n\n所有状态、六臂、失败与两轮原生门槛保留；索引从0开始。有理数证书仅证明归档二进制浮点数的仿射算术；未独立验证模型输出或真实导数。训练上限不等于原生验收预算。无新模型计算、求解、选头或部署；LC2仍开放。\n')
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
