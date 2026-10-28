"""Bind eight original ZIPs and independently reproduce C161/C162 arithmetic."""
from core163 import *
from inputs161 import load as load161
from audit162 import audit as audit161
from diagnostics162 import diagnostics
from audit147 import source_check
def audit162(path,a,d,out,c,budget=lambda:None):
 out=Path(out)
 with archive155.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source162'],'C162');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C162_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(s['synthetic_fixture'],not c['production']);compare(s['errors'],[]);compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),previous.design())
  for n,v in dict(training_states_audited=84,training_records_audited=672,training_updates_audited=8,final_FP64_records_audited=84,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,actual_FP32_proposals_rehashed=168,NumPy_continuous_map_bindings_reconstructed=840,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0).items():compare(s[n],v,'C162 denominator/'+n)
  for n in ('LC2_closed','LC2_single_fold_passed','candidate_selected','best_checkpoint_selected','writer_deployed','thresholds_changed','parent_decision_reclassified','gradients_independently_recomputed','endpoint_KL_independently_recomputed','Torch_FP64_forward_point_bytes_independently_rehashed','causal_failure_explanation_established','model_libraries_imported','archived_source_executed'):compare(s[n],False)
  for n in ('all1092_records_reconstructed','actual_initial_and_fixed_final_head_bound','all_original_parent_archives_independently_bound'):compare(s[n],True)
  compare(s['scope'],previous.design()['scope']);compare(s['members_hashed'],a['members_hashed'],'C162 actual C161 member count');compare(s['parent_decision'],a['summary']['decision']);compare(s['native_summaries'],a['native_summaries']);compare(s['paired_by_round'],a['paired']);compare(s['maximum_update_arithmetic_error_Linf'],a['maximum_update_arithmetic_error_Linf']);compare(s['diagnostics'],d);compare(s['original_inputs_unchanged'],{k:True for k in previous.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json')
  for k in previous.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C162 original link/'+k)
  for file in sorted(out.rglob('*')):
   if not file.is_file():continue
   budget();relative=file.relative_to(out).as_posix()
   if file.suffix=='.json':compare(z.json('diagnostics/'+relative),clean_json(file.read_bytes()),'C162 full diagnostic/'+relative,atol=1e-10,rtol=1e-9)
   elif file.suffix=='.npz':
    old=z.arrays('diagnostics/'+relative)
    with np.load(file,allow_pickle=False) as new:
     compare(set(old),set(new.files))
     for n in old:
      if old[n].dtype.kind=='b':exact(old[n],new[n],'C162 zero gradient masks')
      else:close(old[n],new[n],'C162 complete gradient cosine matrix',1e-12)
  return dict(decision=s['decision'],members_hashed=members,all672_transitions_and336_precision_rows_recomputed=True,parent_decision_retained=s['parent_decision'])
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly eight originals')
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
 for n,h in c['source162'].items():compare(sha(ROOT/'vendor162'/n),h,'Trusted C162 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load161({k:paths[k] for k in parent.INPUT_KEYS},out/'C155_to_C160',c['parent']['parent'],budget);budget();a=audit161(paths['C161'],ctx,out/'C161',c['parent'],budget);budget();d=diagnostics(a,out/'C162_recomputed_diagnostics');receipt=audit162(paths['C162'],a,d,out/'C162_recomputed_diagnostics',c,budget)
 ctx['C161_audit']=a;ctx['C162_receipt']=receipt;ctx['initial_head']=a['heads'][-1].copy();ctx['parent_final_FP64_records']=a['batches'][-1];ctx['parent_final_native_records']={str(t):a['natives']['native_final'+str(t)] for t in (0,1)};dump(out/'C161_C162_BINDING_RECEIPT.json',receipt);return ctx
