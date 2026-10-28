"""Thirteen pinned originals; rerun arithmetic and reverify exact direction, no LP/model."""
from core168 import *
from inputs165 import load as load165
from audit166 import audit as audit165
from diagnostics166 import diagnostics as diagnostics166
from worker167 import bind166
from audit147 import source_check
from certificate167 import verify_saved

def reconstructed166(a,d):
 return dict(parent_protocol=old.PROTOCOL,parent_decision=a['summary']['decision'],members_hashed=a['members_hashed'],training_states_audited=84,recorded_gradient_pairs_audited=84,recorded_direction_iterations_replayed=len(a['trace']),temporary_probe_heads_audited=6,actual_FP64_point_arrays_rehashed=672,native_FP32_point_arrays_rehashed=588,native_endpoint_records_audited=1176,native_repeat_pairs_audited=588,FP64_anchor_repeat_pairs_audited=84,scalar_central_difference_rows_audited=252,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],computed_strict_common_response_descent=a['direction']['computed_strict_common_response_descent'],direction_zero=a['direction']['direction_zero'],direction_diagnostics=a['direction'],native_summaries=a['diagnostics']['native_summaries'],paired_vs_anchor=a['diagnostics']['paired_vs_anchor'],diagnostics=d,all_eleven_originals_independently_bound=True)

def bind167(path,ctx,a,out,c,budget=lambda:None):
 from core167 import problem
 pr=problem(a['gradient_records'],ctx['initial_head'])
 with contextlib.closing(Archive(path)) as z:
  count=z.verify_manifest();source_check(z,c['source167'],'C167');s=z.json('SUMMARY.json');compare(s['protocol'],prior.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C167_RECORDED_LINEAR_PROBLEM_DIAGNOSTIC_COMPLETE_NO_MODEL_ACCEPTANCE');compare(s['errors'],[]);compare(s['synthetic_fixture'],not c['production']);compare(s['scope'],prior.SCOPE);compare(z.json('DESIGN_FROZEN_BEFORE_SOLVE.json'),prior.design())
  for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','confirmation_states_evaluated','excluded_input_states_evaluated'):compare(s[k],0,'C167 original count/'+k)
  for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','best_checkpoint_selected','thresholds_changed','parent_decision_reclassified','archived_source_executed','gradients_independently_recomputed','KL_independently_recomputed','true_model_derivative_feasibility_certified','finite_endpoint_acceptance_evaluated','guard_preservation_certified','global_writer_infeasibility_certified','model_libraries_imported'):compare(s[k],False)
  compare(s['linear_program_calls_attempted'],2);compare(s['new_offline_direction_optimization'],True);compare(s['solver_internal_linear_system_count'],None);compare(s['scipy_version'],'1.17.0');compare(s['training_states_audited'],84);compare(s['active_response_loss_rows'],len(pr['active']));compare(s['active_state_ids'],[IDS[i] for i in pr['active']]);compare(s['free_coordinates'],int(pr['free'].sum()));compare(s['all_twelve_originals_independently_bound'],True)
  compare(s['parent_C165_decision'],a['summary']['decision']);compare(s['parent_C166_decision'],'C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED')
  for t in (0,1):compare(s['parent_anchor_native_round'+str(t)+'_passed'],a['diagnostics']['native_summaries']['anchor'][str(t)]['passed'])
  compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C167 original link/'+k)
  saved=z.arrays('RECORDED_LINEAR_PROBLEM.npz');compare(set(saved),set(('anchor','raw','guards','free','active','norms','A','loss','old_C165_direction')))
  for k in ('anchor','raw','guards','free','active','norms','A','loss'):exact(saved[k],pr[k],'C167 exact problem/'+k)
  with contextlib.closing(Archive(c['pins']['C165']['path'])) as original:
   original_direction=original.arrays('ONE_FIXED_DIRECTION.npz')['direction']
  exact(saved['old_C165_direction'],original_direction,'Original C165 direction remains distinct')
  pb=z.json('RECORDED_PROBLEM_BINDING.json');compare(pb['problem_sha256'],hashlib.sha256(z.raw('RECORDED_LINEAR_PROBLEM.npz')).hexdigest());compare(pb['ordered_state_ids'],IDS);compare(pb['active_state_ids'],[IDS[i] for i in pr['active']]);compare(pb['active_count'],len(pr['active']));compare(pb['free_coordinates'],int(pr['free'].sum()));compare(pb['raw_archive_sha256'],c['pins']['C165']['sha256']);compare(pb['C166_audit_sha256'],c['pins']['C166']['sha256']);compare(pb['old_direction_only_descriptive'],True)
  arrays=z.arrays('DIRECTION_CERTIFICATE_ARRAYS.npz');cert=z.json('EXACT_RATIONAL_CERTIFICATE.json')
  for k in ('raw','guards','free','active','norms'):exact(arrays[k],pr[k],'C167 certificate input/'+k)
  verify_saved(arrays,cert,IDS);budget();compare(cert['classification'],'RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED','Require positive exact common-descent witness');need(int(cert['optimal_margin_lower']['numerator'])>0,'Strictly positive exact lower bound')
  for k in ('optimal_margin_lower','optimal_margin_upper','optimal_margin_interval_width','strict_recorded_raw_common_descent','no_strict_recorded_raw_common_descent_proved'):compare(s[k],cert[k],'C167 independently reverified exact statement/'+k)
  compare(s['recorded_linear_problem_classification'],cert['classification']);compare(s['exact_certificate_available'],True);compare(s['all84_exact_dot_products_verified'],True);compare(s['direction_witness_sha256'],hashlib.sha256(z.raw('DIRECTION_CERTIFICATE_ARRAYS.npz')).hexdigest())
  d=arrays['direction'].reshape(ctx['initial_head'].shape).copy();compare(d.dtype,np.dtype(np.float64));need(np.all(d.ravel()[~pr['free']]==0) and 0<np.max(abs(d))<=1,'Same fixed bounded coefficient subspace')
  receipt=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=count,classification=cert['classification'],direction_witness_sha256=s['direction_witness_sha256'],exact_lower_bound=cert['optimal_margin_lower'],exact_upper_bound=cert['optimal_margin_upper'],active_state_ids=[IDS[i] for i in pr['active']],all84_recorded_gradients_rebound=True,new_LP_calls=0,archived_source_executed=False)
  out=Path(out);out.mkdir(parents=True,exist_ok=True);dump(out/'C167_EXACT_DIRECTION_REVERIFIED.json',receipt);dump(out/'EXACT_RATIONAL_CERTIFICATE_REVERIFIED.json',cert);np.savez_compressed(out/'DIRECTION_CERTIFICATE_ARRAYS_REBOUND.npz',**arrays)
 return d,receipt

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly thirteen original ZIPs')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole input before payload/'+k)
 for n,h in c['source167'].items():compare(sha(ROOT/'vendor167'/n),h,'Delivered trusted C167 source/'+n)
 # Bind explicit actual input locations; archived paths are only descriptive and may have moved.
 cc=dict(c);cc['pins']={k:dict(v,path=str(paths[k])) for k,v in c['pins'].items()}
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load165({k:paths[k] for k in old.INPUT_KEYS},out/'original10',c['parent']['parent']['parent'],budget)
  a=audit165(paths['C165'],ctx,out/'C166_reconstruction/C165_arithmetic',c['parent']['parent'],budget);d=diagnostics166(a,out/'C166_reconstruction/diagnostics');fresh=reconstructed166(a,d);receipt166=bind166(paths['C166'],fresh,out/'C166_reconstruction',c['parent']);direction,receipt167=bind167(paths['C167'],ctx,a,out/'C167_binding',cc,budget)
 ctx.update(C165_gradient_records=a['gradient_records'],C165_FP64_anchor_records=a['gradient_records'],C165_native_anchor_records={str(t):a['native_records'][('anchor',t)] for t in (0,1)},C166_receipt=receipt166,C167_receipt=receipt167,fixed_C167_direction=direction)
 dump(out/'THIRTEEN_ORIGINALS_BOUND_RECEIPT.json',dict(all_thirteen_originals_independently_bound=True,C166=receipt166['decision'],C167=receipt167['classification'],new_model_forwards=0,new_LP_calls=0));return ctx
