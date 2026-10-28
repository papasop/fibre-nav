"""Pure NumPy nine-original read-only audit; no model or training."""
import argparse,resource,traceback
from core164 import *
from audit164 import audit
from diagnostics164 import diagnostics
def run(paths,out,c=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();errors=[];stage='registered';s=dict(protocol=PROTOCOL,complete=False,decision='C164_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED',synthetic_fixture=not c['production'],C163_zip_sha256_known_before_runtime=True,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,gradients_independently_recomputed=False,endpoint_KL_independently_recomputed=False,Torch_FP64_forward_point_bytes_independently_rehashed=False,reported_map_errors_independently_recomputed=False,chronological_seal_order_independently_proved=False,chronology_or_nonaccess_proved=False,causal_failure_explanation_established=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
 def budget():need(time.monotonic()-start<TIMEOUT,'C164 fixed30-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not (out/'SUMMARY.json').exists(),'Fresh output directory');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly nine original scientific ZIPs')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
  for n,h in c['source163'].items():compare(sha(ROOT/'vendor163'/n),h,'Trusted C163 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage='eight_parent_readonly_binding';checkpoint()
  with archive155.prior.no_models(),safe.no_solvers():
   if input_loader is None:
    from inputs163 import load
    ctx=load({k:paths[k] for k in parent.INPUT_KEYS},out/'original_parent_binding',c['parent'],budget)
   else:ctx=input_loader(paths,out,c,budget)
   budget();stage='C163_full84_updates_and_actual_points';checkpoint();a=audit(paths['C163'],ctx,out/'C163_arithmetic',c,budget);budget();stage='fixed_retrospective_diagnostics';d=diagnostics(a,out/'diagnostics');budget()
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub')),'C164 must not import model libraries');s.update(complete=True,decision='C164_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',parent_protocol=parent.PROTOCOL,parent_decision=a['summary']['decision'],members_hashed=a['members_hashed'],training_states_audited=84,training_records_audited=672,training_updates_audited=8,final_FP64_records_audited=84,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,all1092_records_reconstructed=True,actual_initial_and_fixed_final_head_bound=True,actual_FP32_proposals_rehashed=168,NumPy_continuous_map_bindings_reconstructed=840,actual_Torch_FP64_points_rehashed=756,Torch_FP64_forward_point_bytes_independently_rehashed=True,reported_map_errors_independently_recomputed=True,nine_actual_FP64_batch_seals_verified=True,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_actual_FP64_vs_NumPy_map_error_Linf'],all_original_parent_archives_independently_bound=input_loader is None,native_summaries=a['native_summaries'],paired_by_round=a['paired'],maximum_update_arithmetic_error_Linf=a['maximum_update_arithmetic_error_Linf'],diagnostics=d,model_libraries_imported=False);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C164：C163只读平滑训练审计\n\n'+json.dumps(dict(parent_decision=s['parent_decision'],native_summaries=s['native_summaries'],diagnostics=d),ensure_ascii=False,indent=2)+'\n\n重哈希756个保存的实际FP64点，重算它们与NumPy映射的差异和行绑定；归档接收单与计数不独立证明真实先后顺序或模型确实消费这些点。记录梯度没有重新反传，KL没有重新前向计算。平滑目标改善与原始最坏态改善分别报告；诊断不证明因果、收敛、不可行性或泛化。原C163判定不改写，LC2仍开放。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  checks={}
  for k in INPUT_KEYS:
   try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if not all(checks.values()):s['complete']=False;errors.append('Original input missing or changed')
  if not s['complete']:s['decision']='C164_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
