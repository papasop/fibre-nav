"""Pure NumPy eleven-original read-only C165 direction/probe audit."""
import argparse,resource,traceback
from core166 import *
from audit166 import audit
from diagnostics166 import diagnostics

def run(paths,out,c=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();errors=[];stage='registered';s=dict(protocol=PROTOCOL,complete=False,decision='C166_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED',synthetic_fixture=not c['production'],C165_zip_sha256_known_before_runtime=True,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,gradients_independently_recomputed=False,KL_independently_recomputed=False,chronology_or_nonaccess_proved=False,model_consumption_of_sealed_points_independently_proved=False,optimality_or_infeasibility_certified=False,causal_explanation_established=False,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,scope=design()['scope'])
 def budget():need(time.monotonic()-start<TIMEOUT,'C166 fixed30-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not (out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly eleven originals')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
  for n,h in c['source165'].items():compare(sha(ROOT/'vendor165'/n),h,'Delivered trusted C165 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage='ten_original_parent_binding';checkpoint()
  with archive155.prior.no_models(),safe.no_solvers():
   if input_loader is None:
    from inputs165 import load
    ctx=load({k:paths[k] for k in parent.INPUT_KEYS},out/'original_parent_binding',c['parent'],budget)
   else:ctx=input_loader(paths,out,c,budget)
   stage='C165_complete_direction_and_probe_arithmetic';checkpoint();a=audit(paths['C165'],ctx,out/'C165_arithmetic',c,budget);budget();stage='full84_fixed_diagnostics';d=diagnostics(a,out/'diagnostics');budget()
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub')),'C166 must not import model libraries');s.update(complete=True,decision='C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',parent_protocol=parent.PROTOCOL,parent_decision=a['summary']['decision'],members_hashed=a['members_hashed'],all_eleven_originals_independently_bound=input_loader is None,all1848_records_reconstructed=True,training_states_audited=84,recorded_gradient_pairs_audited=84,recorded_direction_iterations_replayed=len(a['trace']),temporary_probe_heads_audited=6,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_difference_rows_audited=252,actual_FP64_map_error_recomputed=True,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],computed_strict_common_response_descent=a['direction']['computed_strict_common_response_descent'],direction_zero=a['direction']['direction_zero'],direction_diagnostics=a['direction'],native_summaries=a['diagnostics']['native_summaries'],paired_vs_anchor=a['diagnostics']['paired_vs_anchor'],diagnostics=d,model_libraries_imported=False);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C166：C165方向与固定探测只读审计\n\n'+json.dumps(dict(parent_decision=s['parent_decision'],direction=a['direction'],native_summaries=s['native_summaries'],AD_FD_statistics=d['all3_h_AD_FD_descriptive_statistics']),ensure_ascii=False,indent=2)+'\n\n所有1848记录、84态、6臂和原判定保留。重放原4096迭代只核验记录，不优化新方向；最终组合范数、FW间隙和符号仅为浮点诊断，不是最优性或不可行性证书。梯度和KL未通过新模型重算；中心差分分歧不单独证明AD错误。封存接收单与字节一致性不证明真实时间顺序或模型消费点的事实。LC2仍开放。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  checks={}
  for k,p in paths.items():
   try:checks[k]=sha(p)==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):s['complete']=False;errors.append('Original input missing or changed')
  if not s['complete']:s['decision']='C166_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
