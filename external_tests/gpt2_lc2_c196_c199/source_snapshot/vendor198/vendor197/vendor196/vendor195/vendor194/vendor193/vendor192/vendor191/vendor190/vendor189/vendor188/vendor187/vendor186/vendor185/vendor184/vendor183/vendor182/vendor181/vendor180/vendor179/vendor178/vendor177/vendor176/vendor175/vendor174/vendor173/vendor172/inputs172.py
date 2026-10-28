"""Reopen17 originals, reconstruct C170 and bind original C171 audit."""
from core172 import *
from inputs170 import load as load170
from audit171 import audit as audit170
from diagnostics171 import diagnostics as diagnostic170
from audit147 import source_check
def bind171(path,a,d,out,c):
 expected=dict(protocol=prior.PROTOCOL,complete=True,decision=prior.DECISION,synthetic_fixture=not c['production'],C170_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=prior.SCOPE,**prior.no_compute(),parent_protocol=control.PROTOCOL,parent_decision=a['summary']['decision'],C170_zip_sha256=c['pins']['C170']['sha256'],members_hashed=a['members_hashed'],all_sixteen_originals_independently_bound=c['production'],training_states_audited=84,training_updates_audited=8,gradient_records_audited=672,ordered_records_audited=1092,recorded_LP_calls_reconstructed=a['LPcalls_reconstructed'],exact_recorded_direction_checks_reconstructed=8,actual_FP64_point_arrays_rehashed=756,native_FP32_point_arrays_rehashed=168,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],initial_head_digest=parameter_hash(a['heads'][0]),fixed_final_step8_head_digest=parameter_hash(a['heads'][-1]),native_summaries=a['native_summaries'],paired_by_round=a['paired'],initial_FP64_objective=prior.frame_objective(a['frames'][0]),final_FP64_objective=prior.frame_objective(a['frames'][-1]),diagnostics=d)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source171'],'C171');s=z.json('SUMMARY.json');compare(s['errors'],[]);compare(s['stage'],'complete')
  for k,v in expected.items():compare(s[k],v,'Independent C171 original/'+k)
  compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design());links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C171 original ancestry/'+k)
  compare(z.json('diagnostics/DIAGNOSTIC_SUMMARY.json'),d,'Original C171 diagnostic duplicate');compare(z.json('C170_arithmetic/ALL8_EXACT_DIRECTION_CHECKS_RECOMPUTED.json'),a['certificates'])
 r=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=members,C170_original_independently_reaudited=True,C171_scientific_summary_reconstructed=True,new_model_forwards=0,new_LP_calls=0);dump(Path(out)/'C171_ORIGINAL_REBOUND.json',r);return r
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
 for n,h in c['source171'].items():compare(sha(ROOT/'vendor171'/n),h,'Delivered trusted C171 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load170({k:paths[k] for k in control.INPUT_KEYS},out/'fifteen_originals',c['parent']['parent'],budget);a=audit170(paths['C170'],ctx,out/'C170_reconstruction',c['parent'],budget);d=diagnostic170(a,out/'C171_reconstruction');r=bind171(paths['C171'],a,d,out/'C171_binding',c)
 p=a['heads'][-1].copy()
 if c['production']:compare(parameter_hash(p),design()['initial_head_digest'],'Original fixed final digest')
 ctx.update(C171_receipt=r,C170_final_head=p,C170_final_FP64=a['frames'][-1],C170_final_native={str(i):a['native']['native_final'+str(i)] for i in (0,1)});dump(out/'SEVENTEEN_ORIGINALS_BOUND_RECEIPT.json',dict(all_seventeen_originals_independently_bound=True,C171_decision=r['decision'],new_model_forwards=0,new_LP_calls=0));return ctx
