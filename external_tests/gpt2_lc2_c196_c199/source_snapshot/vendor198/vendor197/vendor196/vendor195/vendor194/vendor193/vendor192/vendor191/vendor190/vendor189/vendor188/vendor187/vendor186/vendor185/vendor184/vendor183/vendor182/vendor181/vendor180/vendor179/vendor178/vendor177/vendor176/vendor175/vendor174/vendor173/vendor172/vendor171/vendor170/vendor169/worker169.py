"""Pure CPU NumPy audit, no model or solver execution."""
import argparse,resource,traceback
from core169 import *
from audit169 import audit
from diagnostics169 import diagnostics
def run(paths,out,c=None,input_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();errors=[];stage='registered';s=dict(protocol=PROTOCOL,complete=False,decision=INCOMPLETE,synthetic_fixture=not c['production'],C168_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_bound_certified=False,scope=SCOPE,**no_compute())
 def budget():need(time.monotonic()-started<TIMEOUT,'C169 fixed30-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh audit output only');dump(out/'DESIGN_FROZEN_BEFORE_AUDIT.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly fourteen originals')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
  for n,h in c['source168'].items():compare(sha(ROOT/'vendor168'/n),h,'Trusted delivered C168 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage='thirteen_original_bindings_and_exact_C167_certificate';checkpoint()
  with archive155.prior.no_models(),safe.no_solvers():
   if input_loader is None:
    from inputs168 import load
    ctx=load({k:paths[k] for k in prior.INPUT_KEYS},out/'thirteen_original_binding',c['parent'],budget)
   else:ctx=input_loader(paths,out,c,budget)
   stage='C168_all_records_points_gates_and_secants';checkpoint();a=audit(paths['C168'],ctx,out/'C168_arithmetic',c,budget);budget();stage='all84_all6_observed_change_diagnostics';d=diagnostics(a,out/'diagnostics');budget()
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub','scipy')),'C169 must not import model or solver libraries')
  s.update(complete=True,decision=DECISION,parent_protocol=prior.PROTOCOL,parent_decision=a['summary']['decision'],C168_zip_sha256=c['pins']['C168']['sha256'],members_hashed=a['members_hashed'],all_fourteen_originals_independently_bound=input_loader is None,exact_C167_recorded_gradient_certificate_reverified=input_loader is None,training_states_audited=84,temporary_probe_heads_audited=6,ordered_records_audited=1848,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_difference_rows_recomputed=252,signed_loss_change_rows_recomputed=504,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],positive_steps=a['panel']['positive_steps'],native_summaries=a['diagnostics']['native_summaries'],all_positive_arm_active_loss_changes=a['diagnostics']['all3_positive_arms_active_loss_changes'],FP64_response_statistics=a['diagnostics']['FP64_response_statistics'],diagnostics=d);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C169：C168有限步探针只读审计\n\n'+json.dumps(dict(parent_decision=s['parent_decision'],positive_steps=s['positive_steps'],native_summaries=s['native_summaries'],positive_active_loss_changes=s['all_positive_arm_active_loss_changes'],diagnostics=d),ensure_ascii=False,indent=2)+'\n\n保留全部84态、6臂、原生两轮和原判定。不选择头、步长或部署。小步下降、有限差分与归档梯度的一致性、较大步长变化及FP32/FP64差异均为描述性诊断；不证明导数真值、曲率界、舍入因果、可行/不可行性、收敛或泛化。本轮无模型重放，LC2仍开放。\n')
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
