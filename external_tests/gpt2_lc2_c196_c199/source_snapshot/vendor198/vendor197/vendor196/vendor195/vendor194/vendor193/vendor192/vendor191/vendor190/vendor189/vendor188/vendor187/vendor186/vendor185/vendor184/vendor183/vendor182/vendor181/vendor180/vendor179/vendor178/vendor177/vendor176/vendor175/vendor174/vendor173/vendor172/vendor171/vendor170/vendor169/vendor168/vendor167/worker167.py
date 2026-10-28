"""Bind twelve originals, rerun trusted C166 arithmetic, solve and verify exact witnesses."""
import argparse,resource,traceback
from core167 import *
from certificate167 import verify,verify_saved

def bind166(path,fresh,out,c):
 from audit147 import source_check
 with contextlib.closing(Archive(path)) as z:
  count=z.verify_manifest();source_check(z,c['source166'],'C166');s=z.json('SUMMARY.json')
  compare(s['protocol'],prior.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(s['errors'],[]);compare(s['synthetic_fixture'],not c['production']);compare(s['scope'],prior.design()['scope']);compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design())
  for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_direction_optimization_iterations','new_Jacobians','new_SVD_calls','linear_systems_solved','confirmation_states_evaluated','excluded_input_states_evaluated'):compare(s[k],0,'Original C166 count/'+k)
  for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decision_reclassified','gradients_independently_recomputed','KL_independently_recomputed','archived_source_executed','model_libraries_imported','optimality_or_infeasibility_certified'):compare(s[k],False)
  for k in ('all1848_records_reconstructed','C165_zip_sha256_known_before_runtime','actual_FP64_map_error_recomputed'):compare(s[k],True)
  for k in ('parent_protocol','parent_decision','members_hashed','training_states_audited','recorded_gradient_pairs_audited','recorded_direction_iterations_replayed','temporary_probe_heads_audited','actual_FP64_point_arrays_rehashed','native_FP32_point_arrays_rehashed','native_endpoint_records_audited','native_repeat_pairs_audited','FP64_anchor_repeat_pairs_audited','scalar_central_difference_rows_audited','maximum_actual_FP64_vs_NumPy_map_error_Linf','computed_strict_common_response_descent','direction_zero','direction_diagnostics','native_summaries','paired_vs_anchor','diagnostics','all_eleven_originals_independently_bound'):compare(s[k],fresh[k],'Independent C166 reconstruction/'+k)
  if c['production']:compare(s['all_eleven_originals_independently_bound'],True)
  compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C166 original link/'+k)
  # Every reconstructed arithmetic/diagnostic JSON and CSV is compared with the original C166 archive.
  checked=[]
  for folder in ('C165_arithmetic','diagnostics'):
   for f in sorted((out/folder).rglob('*')):
    if not f.is_file():continue
    name=f.relative_to(out).as_posix()
    if f.suffix=='.json':compare(z.json(name),clean_json(f.read_bytes()),'C166 full reconstructed artifact/'+name);checked.append(name)
    elif f.suffix=='.csv':compare(z.raw(name),f.read_bytes(),'C166 exact CSV/'+name);checked.append(name)
  need(checked,'C166 reconstructed artifacts present')
  return dict(protocol=s['protocol'],decision=s['decision'],members_hashed=count,reconstructed_artifacts_checked=checked,original_native_summaries=s['native_summaries'])

def run(paths,out,c=None,parent_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';errors=[]
 s=dict(protocol=PROTOCOL,complete=False,decision='C167_INCOMPLETE_NO_FEASIBILITY_CLAIM',synthetic_fixture=not c['production'],scope=SCOPE,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_program_calls_attempted=0,new_offline_direction_optimization=False,solver_internal_linear_system_count=None,archived_source_executed=False,gradients_independently_recomputed=False,KL_independently_recomputed=False,true_model_derivative_feasibility_certified=False,finite_endpoint_acceptance_evaluated=False,guard_preservation_certified=False,global_writer_infeasibility_certified=False,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,best_checkpoint_selected=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
 def budget():need(time.monotonic()-start<TIMEOUT,'C167 fixed30-minute worker budget')
 def checkpoint():s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output directory');dump(out/'DESIGN_FROZEN_BEFORE_SOLVE.json',design());checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly twelve pinned original ZIPs')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole input hash before payload/'+k)
  for n,h in c['source166'].items():compare(sha(ROOT/'vendor166'/n),h,'Trusted delivered C166 source/'+n)
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});stage='independent_original_C166_reconstruction';checkpoint()
  from worker166 import run as audit166_run
  with archive155.prior.no_models():
   fresh=audit166_run({k:paths[k] for k in prior.INPUT_KEYS},out/'C166_reconstruction',c['parent'],parent_loader);need(fresh['complete'],'C166 independent original reconstruction incomplete: '+str(fresh['errors']));budget();receipt=bind166(paths['C166'],fresh,out/'C166_reconstruction',c);dump(out/'C166_BINDING_RECEIPT.json',receipt)
   with contextlib.closing(Archive(paths['C165'])) as z:
    rr=[clean_json(line) for line in z.raw('ALL1848_RECORDS.jsonl').splitlines()];rows=[r for r in rr if r['stage']=='anchor_gradient' and r['arm']=='anchor'];p=z.arrays('ANCHOR_C163_FIXED_FINAL_HEAD.npz')['P'];old=z.arrays('ONE_FIXED_DIRECTION.npz')['direction']
   pr=problem(rows,p);budget();stage='two_fixed_offline_LPs';checkpoint()
   # Persist exact problem and hashes before importing/calling the new solver.
   np.savez_compressed(out/'RECORDED_LINEAR_PROBLEM.npz',**{k:pr[k] for k in ('anchor','raw','guards','free','active','norms','A','loss')},old_C165_direction=old)
   dump(out/'RECORDED_PROBLEM_BINDING.json',dict(problem_sha256=sha(out/'RECORDED_LINEAR_PROBLEM.npz'),ordered_state_ids=pr['ids'],active_state_ids=[IDS[i] for i in pr['active']],active_count=len(pr['active']),free_coordinates=int(pr['free'].sum()),raw_archive_sha256=c['pins']['C165']['sha256'],C166_audit_sha256=c['pins']['C166']['sha256'],old_direction_only_descriptive=True))
   from solver167 import solve
   import scipy
   need(scipy.__version__=='1.17.0','Pinned SciPy1.17.0 solver required')
   s.update(new_offline_direction_optimization=bool(len(pr['active'])),scipy_version=scipy.__version__,numpy_version=np.__version__);checkpoint()
   def attempt(i):s['linear_program_calls_attempted']=i;checkpoint();budget()
   solution=solve(pr['A'],pr['free'],on_attempt=attempt,on_result=lambda label,result:dump(out/('LP_'+label.upper()+'_RESULT.json'),result));budget();dump(out/'BOTH_LP_RESULTS.json',{k:v for k,v in solution.items() if k not in ('direction','dual')})
   s.update(active_state_ids=[IDS[i] for i in pr['active']],active_response_loss_rows=len(pr['active']),training_states_audited=N,free_coordinates=int(pr['free'].sum()),all_twelve_originals_independently_bound=parent_loader is None,parent_C165_decision=fresh['parent_decision'],parent_C166_decision=receipt['decision'],parent_anchor_native_round0_passed=fresh['native_summaries']['anchor']['0']['passed'],parent_anchor_native_round1_passed=fresh['native_summaries']['anchor']['1']['passed'])
   if solution['direction'] is None:
    s.update(recorded_linear_problem_classification='RECORDED_GRADIENT_FEASIBILITY_UNRESOLVED',exact_certificate_available=False,strict_recorded_raw_common_descent=False,no_strict_recorded_raw_common_descent_proved=False)
   else:
    stage='independent_exact_rational_verification';checkpoint();arrays={k:pr[k] for k in ('raw','guards','free','active','norms')};arrays.update(direction=solution['direction'],dual=solution['dual']);cert=verify(**arrays,ids=pr['ids']);budget();np.savez_compressed(out/'DIRECTION_CERTIFICATE_ARRAYS.npz',**arrays);dump(out/'EXACT_RATIONAL_CERTIFICATE.json',cert)
    # Serialize and independently reopen/recompute the final certificate; no solver imported by verifier.
    with np.load(out/'DIRECTION_CERTIFICATE_ARRAYS.npz',allow_pickle=False) as z:check={k:z[k] for k in z.files}
    verify_saved(check,clean_json((out/'EXACT_RATIONAL_CERTIFICATE.json').read_bytes()),pr['ids']);budget()
    olddots=pr['raw']@old.ravel();newdots=pr['raw']@solution['direction'];dump(out/'ALL84_OLD_AND_NEW_RECORDED_CONTRACTIONS.json',[dict(state_id=sid,active=i in pr['active'],old_raw_response_dot=float(olddots[i]),new_raw_response_dot=float(newdots[i]),exact_new_raw_response_dot=cert['rows'][i]['raw_response_dot'],exact_new_guard_dot=cert['rows'][i]['guard_dot']) for i,sid in enumerate(IDS)])
    s.update(recorded_linear_problem_classification=cert['classification'],exact_certificate_available=True,strict_recorded_raw_common_descent=cert['strict_recorded_raw_common_descent'],no_strict_recorded_raw_common_descent_proved=cert['no_strict_recorded_raw_common_descent_proved'],optimal_margin_lower=cert['optimal_margin_lower'],optimal_margin_upper=cert['optimal_margin_upper'],optimal_margin_interval_width=cert['optimal_margin_interval_width'],all84_exact_dot_products_verified=True,direction_witness_sha256=sha(out/'DIRECTION_CERTIFICATE_ARRAYS.npz'))
  need(not any(n in sys.modules for n in ('torch','transformers','peft','huggingface_hub')),'No model libraries');s.update(complete=True,decision='C167_RECORDED_LINEAR_PROBLEM_DIAGNOSTIC_COMPLETE_NO_MODEL_ACCEPTANCE',model_libraries_imported=False);stage='complete'
  (out/'REPORT_ZH.md').write_text('# C167 归档梯度共同下降可行性\n\n'+json.dumps({k:s[k] for k in ('recorded_linear_problem_classification','active_response_loss_rows','strict_recorded_raw_common_descent','no_strict_recorded_raw_common_descent_proved','parent_anchor_native_round0_passed')},ensure_ascii=False,indent=2)+'\n\n精确证书仅关于归档二进制浮点梯度定义的线性问题。没有重算真实梯度，也没有在GPT-2上执行新方向；保护项的点积已保留，但未作为本次共同下降约束。不能据此宣称有限写入通过、全局不可行或LC2闭合。原始门槛、训练分母与历史判定保持。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True)
 finally:
  s['original_inputs_unchanged']={}
  for k,p in paths.items():
   try:s['original_inputs_unchanged'][k]=sha(p)==c['pins'][k]['sha256']
   except Exception:s['original_inputs_unchanged'][k]=False
  if set(s['original_inputs_unchanged'])!=set(INPUT_KEYS) or not all(s['original_inputs_unchanged'].values()):s['complete']=False;errors.append('Original missing or changed')
  if not s['complete']:s['decision']='C167_INCOMPLETE_NO_FEASIBILITY_CLAIM'
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for k in INPUT_KEYS:ap.add_argument('--'+k.lower(),required=True)
 ap.add_argument('--out',required=True);a=ap.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
