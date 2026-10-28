"""Nineteen fixed originals; reconstruct C172 and bind original C173 audit."""
from core174 import *
from inputs172 import load as load172
from audit173 import audit as audit172
from diagnostics173 import diagnostics as coordinate172
from audit147 import source_check
def bind173(path,a,d,out,c):
 expected=dict(protocol=prior.PROTOCOL,complete=True,decision=prior.DECISION,synthetic_fixture=not c['production'],C172_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=prior.SCOPE,**prior.no_compute(),parent_protocol=source172.PROTOCOL,parent_decision=a['summary']['decision'],C172_zip_sha256=c['pins']['C172']['sha256'],members_hashed=a['members_hashed'],all_eighteen_originals_independently_bound=c['production'],training_states_audited=84,temporary_probe_heads_audited=6,ordered_records_audited=1848,gradient_records_audited=84,recorded_LP_calls_reconstructed=a['solver']['recorded_LP_calls'],original_solver_internal_linear_system_count=None,exact_recorded_direction_check_reconstructed=True,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_differences_recomputed=252,signed_scalar_loss_changes_recomputed=504,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],original_C170_fixed_final_head_digest=parameter_hash(a['head']),original_positive_steps=list(H),parent_diagnostics=a['diagnostics'],coordinate_diagnostics=d)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source173'],'C173');s=z.json('SUMMARY.json');compare(s['errors'],[]);compare(s['stage'],'complete')
  for k,v in expected.items():compare(s[k],v,'Independent C173 summary/'+k)
  compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design());links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C173 original ancestry/'+k)
  compare(z.json('diagnostics/DIAGNOSTIC_SUMMARY.json'),d,'C173 coordinate diagnostic duplicate');compare(z.json('C172_arithmetic/PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json'),a['diagnostics'])
 r=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=members,C172_original_independently_reaudited=True,C173_scientific_summary_reconstructed=True,new_model_forwards=0,new_LP_calls=0);dump(Path(out)/'C173_ORIGINAL_REBOUND.json',r);return r
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original before payload/'+k)
 for n,h in c['source173'].items():compare(sha(ROOT/'vendor173'/n),h,'Delivered trusted C173 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load172({k:paths[k] for k in source172.INPUT_KEYS},out/'seventeen_originals',c['parent']['parent'],budget);a=audit172(paths['C172'],ctx,out/'C172_reconstruction',c['parent'],budget);d=coordinate172(a,out/'C173_reconstruction');r=bind173(paths['C173'],a,d,out/'C173_binding',c)
 ctx.update(C173_receipt=r,C170_final_head=a['head'].copy(),C172_anchor_FP64=a['anchor'],C172_anchor_native={str(i):a['native'][('anchor',i)] for i in (0,1)});dump(out/'NINETEEN_ORIGINALS_BOUND_RECEIPT.json',dict(all_nineteen_originals_independently_bound=True,C173_decision=r['decision'],new_model_forwards=0,new_LP_calls=0));return ctx
