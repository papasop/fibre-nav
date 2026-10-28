"""Positive C167 schema fixture; synthetic upstream, never real history evidence."""
import io,copy,shutil
from core168 import *
from core167 import problem
from certificate167 import verify

def pack(root,path):
 import zipfile
 root=Path(root);dump(root/'RESULT_MANIFEST.json',{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json'})
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(root).as_posix())
 return path

def fixture(root):
 root=Path(root);root.mkdir(parents=True);p=np.zeros((3,2,36));d=np.zeros_like(p);d.ravel()[:9]=-1.;rows=[]
 for i,sid in enumerate(IDS):
  r=np.zeros_like(p)
  if i<9:r.ravel()[i]=1.
  rows.append(dict(state_id=sid,objective_parts=dict(response_loss=1. if i<9 else 0.),response_gradient=r.tolist(),guard_gradient=np.zeros_like(p).tolist()))
 ctx=dict(initial_head=p);a=dict(gradient_records=rows,summary=dict(decision='C165_FIXED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED'),diagnostics=dict(native_summaries={'anchor':{'0':dict(passed=79),'1':dict(passed=79)}}));paths={}
 for k in prior.INPUT_KEYS:
  path=root/(k+'.zip');path.write_bytes(b'Synthetic upstream marker only');paths[k]=path
 import zipfile
 bio=io.BytesIO();np.savez(bio,direction=np.zeros_like(p))
 with zipfile.ZipFile(paths['C165'],'w') as z:z.writestr('ONE_FIXED_DIRECTION.npz',bio.getvalue())
 c=contract();c.update(production=False,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});out=root/'C167';out.mkdir();pr=problem(rows,p);np.savez_compressed(out/'RECORDED_LINEAR_PROBLEM.npz',**{k:pr[k] for k in ('anchor','raw','guards','free','active','norms','A','loss')},old_C165_direction=np.zeros_like(p));arrays={k:pr[k] for k in ('raw','guards','free','active','norms')};arrays.update(direction=d.ravel(),dual=np.full(9,1/9,dtype=np.float64));cert=verify(**arrays,ids=IDS);np.savez_compressed(out/'DIRECTION_CERTIFICATE_ARRAYS.npz',**arrays);dump(out/'EXACT_RATIONAL_CERTIFICATE.json',cert);dump(out/'DESIGN_FROZEN_BEFORE_SOLVE.json',prior.design());dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});dump(out/'RECORDED_PROBLEM_BINDING.json',dict(problem_sha256=sha(out/'RECORDED_LINEAR_PROBLEM.npz'),ordered_state_ids=IDS,active_state_ids=IDS[:9],active_count=9,free_coordinates=216,raw_archive_sha256=c['pins']['C165']['sha256'],C166_audit_sha256=c['pins']['C166']['sha256'],old_direction_only_descriptive=True))
 s=dict(protocol=prior.PROTOCOL,complete=True,decision='C167_RECORDED_LINEAR_PROBLEM_DIAGNOSTIC_COMPLETE_NO_MODEL_ACCEPTANCE',errors=[],synthetic_fixture=True,scope=prior.SCOPE,linear_program_calls_attempted=2,new_offline_direction_optimization=True,solver_internal_linear_system_count=None,scipy_version='1.17.0',training_states_audited=84,active_response_loss_rows=9,active_state_ids=IDS[:9],free_coordinates=216,all_twelve_originals_independently_bound=True,parent_C165_decision=a['summary']['decision'],parent_C166_decision='C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED',parent_anchor_native_round0_passed=79,parent_anchor_native_round1_passed=79,original_inputs_unchanged={k:True for k in prior.INPUT_KEYS},recorded_linear_problem_classification=cert['classification'],exact_certificate_available=True,all84_exact_dot_products_verified=True,direction_witness_sha256=sha(out/'DIRECTION_CERTIFICATE_ARRAYS.npz'),synthetic_schema_fixture_not_real_archive_evidence=True)
 for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','confirmation_states_evaluated','excluded_input_states_evaluated'):s[k]=0
 for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','best_checkpoint_selected','thresholds_changed','parent_decision_reclassified','archived_source_executed','gradients_independently_recomputed','KL_independently_recomputed','true_model_derivative_feasibility_certified','finite_endpoint_acceptance_evaluated','guard_preservation_certified','global_writer_infeasibility_certified','model_libraries_imported'):s[k]=False
 for k in ('optimal_margin_lower','optimal_margin_upper','optimal_margin_interval_width','strict_recorded_raw_common_descent','no_strict_recorded_raw_common_descent_proved'):s[k]=cert[k]
 dump(out/'SUMMARY.json',s);shutil.copytree(ROOT/'vendor167',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z=pack(out,root/'C167.zip');paths['C167']=z;c['pins']['C167']=dict(path=str(z),sha256=sha(z));return paths,c,ctx,a,out
