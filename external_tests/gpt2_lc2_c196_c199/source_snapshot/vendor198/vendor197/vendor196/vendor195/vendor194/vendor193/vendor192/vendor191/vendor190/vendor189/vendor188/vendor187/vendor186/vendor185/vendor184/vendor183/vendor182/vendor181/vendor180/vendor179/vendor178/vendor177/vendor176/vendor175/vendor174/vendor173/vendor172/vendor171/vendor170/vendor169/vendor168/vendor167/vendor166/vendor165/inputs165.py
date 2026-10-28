"""Ten originals; independently reconstruct C163/C164 before any new model call."""
from core165 import *
from inputs163 import load as load163
from audit164 import audit
from diagnostics164 import diagnostics
from audit147 import source_check

def check164(path,a,d,out,c,budget=lambda:None):
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source164'],'C164');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['synthetic_fixture'],not c['production']);compare(s['decision'],'C164_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),previous.design())
  for n,v in dict(training_states_audited=84,training_records_audited=672,training_updates_audited=8,final_FP64_records_audited=84,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,actual_FP32_proposals_rehashed=168,NumPy_continuous_map_bindings_reconstructed=840,actual_Torch_FP64_points_rehashed=756,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0).items():compare(s[n],v,'C164 denominator/'+n)
  for n in ('LC2_closed','LC2_single_fold_passed','candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decision_reclassified','gradients_independently_recomputed','endpoint_KL_independently_recomputed','chronological_seal_order_independently_proved','chronology_or_nonaccess_proved','causal_failure_explanation_established','model_libraries_imported','archived_source_executed'):compare(s[n],False)
  for n in ('C163_zip_sha256_known_before_runtime','Torch_FP64_forward_point_bytes_independently_rehashed','reported_map_errors_independently_recomputed','all1092_records_reconstructed','actual_initial_and_fixed_final_head_bound','nine_actual_FP64_batch_seals_verified','all_original_parent_archives_independently_bound'):compare(s[n],True)
  compare(s['scope'],previous.design()['scope']);compare(s['parent_decision'],a['summary']['decision']);compare(s['members_hashed'],a['members_hashed']);compare(s['maximum_update_arithmetic_error_Linf'],a['maximum_update_arithmetic_error_Linf']);compare(s['maximum_actual_FP64_vs_NumPy_map_error_Linf'],a['maximum_actual_FP64_vs_NumPy_map_error_Linf']);compare(s['native_summaries'],a['native_summaries']);compare(s['paired_by_round'],a['paired']);compare(s['diagnostics'],d);compare(s['original_inputs_unchanged'],{k:True for k in previous.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in previous.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C164 original link/'+k)
  for folder,prefix in [('C163_arithmetic','C163_arithmetic'),('diagnostics','diagnostics')]:
   for f in sorted((out/folder).rglob('*')):
    if not f.is_file():continue
    budget();name=prefix+'/'+f.relative_to(out/folder).as_posix()
    if f.suffix=='.json':compare(z.json(name),clean_json(f.read_bytes()),'C164 reconstructed diagnostic/'+name,atol=1e-10,rtol=1e-9)
    elif f.suffix=='.npz':
     old=z.arrays(name)
     with np.load(f,allow_pickle=False) as new:
      compare(set(old),set(new.files))
      for key in old:
       if old[key].dtype.kind=='b':exact(old[key],new[key],'C164 masks')
       else:close(old[key],new[key],'C164 reconstructed array',1e-12)
  return dict(decision=s['decision'],members_hashed=members,C163_original_decision=s['parent_decision'],actual_FP64_point_arrays_rehashed=756)

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly ten original ZIPs')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
 for n,h in c['source164'].items():compare(sha(ROOT/'vendor164'/n),h,'Trusted C164 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load163({k:paths[k] for k in parent.INPUT_KEYS},out/'eight_parent_binding',c['parent']['parent'],budget);a=audit(paths['C163'],ctx,out/'C163_arithmetic',c['parent'],budget);d=diagnostics(a,out/'diagnostics');receipt=check164(paths['C164'],a,d,out,c,budget)
 ctx.update(initial_head=a['heads'][-1].copy(),C163_audit=a,C164_receipt=receipt,parent_final_FP64_records=a['batches'][-1],parent_final_native_records={str(t):a['natives']['native_final'+str(t)] for t in (0,1)})
 dump(out/'C163_C164_BINDING_RECEIPT.json',receipt);return ctx
