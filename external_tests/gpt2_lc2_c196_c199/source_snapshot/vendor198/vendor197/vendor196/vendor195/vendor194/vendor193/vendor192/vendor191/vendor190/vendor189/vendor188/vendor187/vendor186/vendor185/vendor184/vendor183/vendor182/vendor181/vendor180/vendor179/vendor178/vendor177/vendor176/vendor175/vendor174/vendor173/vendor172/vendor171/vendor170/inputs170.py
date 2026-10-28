"""Fifteen fixed original ZIPs, one parent re-audit and C169 summary binding."""
from core170 import *
from inputs168 import load as load168
from audit169 import audit as audit168
from diagnostics169 import diagnostics as diag168
from audit147 import source_check
def reconstructed169(a,d,c):
 return dict(protocol=prior.PROTOCOL,complete=True,decision=prior.DECISION,synthetic_fixture=not c['production'],C168_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_bound_certified=False,scope=prior.SCOPE,**prior.no_compute(),parent_protocol=source168.PROTOCOL,parent_decision=a['summary']['decision'],C168_zip_sha256=c['pins']['C168']['sha256'],members_hashed=a['members_hashed'],all_fourteen_originals_independently_bound=c['production'],exact_C167_recorded_gradient_certificate_reverified=c['production'],training_states_audited=84,temporary_probe_heads_audited=6,ordered_records_audited=1848,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_difference_rows_recomputed=252,signed_loss_change_rows_recomputed=504,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],positive_steps=a['panel']['positive_steps'],native_summaries=a['diagnostics']['native_summaries'],all_positive_arm_active_loss_changes=a['diagnostics']['all3_positive_arms_active_loss_changes'],FP64_response_statistics=a['diagnostics']['FP64_response_statistics'],diagnostics=d)
def bind169(path,a,d,out,c):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);expected=reconstructed169(a,d,c)
 with contextlib.closing(Archive(path)) as z:
  n=z.verify_manifest();source_check(z,c['source169'],'C169');s=z.json('SUMMARY.json');compare(s['errors'],[]);compare(s['stage'],'complete')
  for k,v in expected.items():compare(s[k],v,'C169 independent summary/'+k)
  compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design());links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C169 original ancestry/'+k)
  compare(z.json('diagnostics/DIAGNOSTIC_SUMMARY.json'),d,'C169 actual diagnostic duplicate');compare(z.json('C168_arithmetic/PARENT_DIAGNOSTIC_SUMMARY_RECOMPUTED.json'),a['diagnostics'])
 r=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=n,C168_original_independently_reaudited=True,all_C169_scientific_summary_fields_reconstructed=True,new_model_forwards=0,new_LP_calls=0);dump(out/'C169_ORIGINAL_REBOUND.json',r);return r
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
 for n,h in c['source169'].items():compare(sha(ROOT/'vendor169'/n),h,'Delivered trusted C169 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load168({k:paths[k] for k in source168.INPUT_KEYS},out/'thirteen_originals',c['parent']['parent'],budget);a=audit168(paths['C168'],ctx,out/'C169_reconstruction/C168_arithmetic',c['parent'],budget);d=diag168(a,out/'C169_reconstruction/diagnostics');receipt=bind169(paths['C169'],a,d,out/'C169_binding',c)
 ctx.update(C169_receipt=receipt,C168_native_anchor_records={str(t):a['native'][('anchor',t)] for t in (0,1)},C168_FP64_anchor_records=a['anchor']);dump(out/'FIFTEEN_ORIGINALS_BOUND_RECEIPT.json',dict(all_fifteen_originals_independently_bound=True,C169_decision=receipt['decision'],new_model_forwards=0,new_LP_calls=0));return ctx
