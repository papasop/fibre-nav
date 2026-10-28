"""Bind all21 fixed originals and reconstruct original C175 audit before model."""
from core176 import *
from inputs174 import load as load174
from audit175 import audit as audit174
from recompute175 import supplemental
from audit147 import source_check

def bind175(path,a,d,out,c):
 expected=dict(protocol=prior.PROTOCOL,complete=True,decision=prior.DECISION,synthetic_fixture=not c['production'],C174_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=prior.SCOPE,**prior.no_compute(),parent_protocol=source174.PROTOCOL,parent_decision=a['summary']['decision'],C174_zip_sha256=c['pins']['C174']['sha256'],members_hashed=a['members_hashed'],all_twenty_originals_independently_bound=c['production'],training_states_audited=84,temporary_probe_heads_audited=6,ordered_records_audited=1848,gradient_records_audited=84,recorded_LP_calls_reconstructed=a['solver']['recorded_LP_calls'],original_solver_internal_linear_system_count=None,exact_recorded_direction_check_reconstructed=True,exact_actual_head_certificates_reconstructed=6,exact_affine_certificate_signed_rows_reconstructed=7056,archived_response_Jacobian_blocks_checked=84,archived_response_coordinate_gradient_rows_checked=504,LP_signed_rows_reconstructed=1008,response_coordinate_affine_rows_recomputed=3024,scalar_gradient_chain_checks_recomputed=84,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_differences_recomputed=252,signed_scalar_loss_changes_recomputed=504,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],original_C170_fixed_final_head_digest=parameter_hash(a['head']),original_positive_steps=list(source174.H),parent_diagnostics=a['diagnostics'],coordinate_diagnostics=d)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source175'],'C175');s=z.json('SUMMARY.json');compare(s['errors'],[]);compare(s['stage'],'complete')
  for k,v in expected.items():compare(s[k],v,'Independent C175 summary/'+k)
  compare(set(s),set(expected)|{'errors','stage','elapsed_worker_seconds','process_peak_RSS_KiB','original_inputs_unchanged'},'Original complete C175 summary schema');need(math.isfinite(s['elapsed_worker_seconds']) and s['elapsed_worker_seconds']>=0,'Original elapsed declaration');need(type(s['process_peak_RSS_KiB']) is int and s['process_peak_RSS_KiB']>=0,'Original RSS declaration');compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design());links=z.json('ORIGINAL_INPUTS.json')
  compare(set(links),set(prior.INPUT_KEYS))
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'Original C175 ancestry/'+k)
  compare(z.json('diagnostics/SUPPLEMENTAL_SUMMARY.json'),d,'C175 coordinate diagnostic duplicate');compare(z.json('C174_arithmetic/PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json'),a['diagnostics'])
 r=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=members,C174_original_independently_reaudited=True,C175_scientific_summary_reconstructed=True,new_model_forwards=0,new_LP_calls=0);dump(Path(out)/'C175_ORIGINAL_REBOUND.json',r);return r

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original before payload/'+k)
 for n,h in c['source175'].items():compare(sha(ROOT/'vendor175'/n),h,'Delivered trusted C175 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load174({k:paths[k] for k in source174.INPUT_KEYS},out/'nineteen_originals',c['parent']['parent'],budget);a=audit174(paths['C174'],ctx,out/'C174_reconstruction',c['parent'],budget);d=supplemental(a,out/'C175_reconstruction');receipt=bind175(paths['C175'],a,d,out/'C175_binding',c)
 ctx.update(C175_receipt=receipt,C170_final_head=a['head'].copy(),C174_anchor_FP64=a['anchor'],C174_anchor_native={str(i):a['native'][('anchor',i)] for i in (0,1)});dump(out/'TWENTY_ONE_ORIGINALS_BOUND_RECEIPT.json',dict(all_twenty_one_originals_independently_bound=True,C175_decision=receipt['decision'],new_model_forwards=0,new_LP_calls=0));return ctx
