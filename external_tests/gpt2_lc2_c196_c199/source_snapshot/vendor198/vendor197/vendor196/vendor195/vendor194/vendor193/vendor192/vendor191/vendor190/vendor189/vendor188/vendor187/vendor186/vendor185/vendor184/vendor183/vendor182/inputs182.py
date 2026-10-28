"""Independently re-audit C180 and bind original C181 before any model call."""
from core182 import *
from inputs180 import load as load180
from audit181 import audit
from diagnostics181 import diagnostics
from audit147 import source_check

def expected181(a,d,c):
 return dict(protocol=prior.PROTOCOL,complete=True,decision=prior.DECISION,synthetic_fixture=not c['production'],C180_zip_sha256_known_before_runtime=True,archived_source_executed=False,model_libraries_imported=False,gradients_independently_recomputed=False,model_outputs_independently_recomputed=False,KL_independently_recomputed=False,solver_optimality_or_infeasibility_certified=False,chronology_or_nonaccess_proved=False,model_consumption_of_saved_points_proved=False,causal_explanation_established=False,curvature_or_convergence_certified=False,scope=prior.SCOPE,**prior.no_compute(),parent_protocol=previous_training.PROTOCOL,parent_decision=a['summary']['decision'],C180_zip_sha256=c['pins']['C180']['sha256'],members_hashed=a['members_hashed'],all_twenty_six_originals_independently_bound=c['production'],training_states_audited=84,training_updates_audited=8,gradient_records_audited=672,ordered_records_audited=1176,recorded_LP_calls_reconstructed=a['LPcalls_reconstructed'],exact_recorded_direction_checks_reconstructed=8,exact_actual_head_checks_reconstructed=8,exact_signed_certificate_rows_reconstructed=16128,LP_signed_constraint_rows_reconstructed=8064,archived_response_Jacobian_blocks_checked=672,archived_response_coordinate_gradient_rows_checked=4032,scalar_gradient_chain_checks_recomputed=672,FP64_final_repeat_pairs_audited=84,response_coordinate_affine_rows_recomputed=4032,original_solver_internal_linear_system_count=None,actual_FP64_point_arrays_rehashed=840,native_FP32_point_arrays_rehashed=168,native_endpoint_records_audited=336,native_repeat_pairs_audited=168,maximum_actual_FP64_vs_NumPy_map_error_Linf=a['maximum_map_error'],initial_head_digest=parameter_hash(a['heads'][0]),fixed_final_step8_head_digest=parameter_hash(a['heads'][-1]),native_summaries=a['native_summaries'],paired_by_round=a['paired'],initial_FP64_objective=objective(a['frames'][0]),final_FP64_objective=objective(a['frames'][-1]),diagnostics=d,endpoint_precision_reconstruction=a['precision'])

def bind181(path,a,d,out,c,reconstructed=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source181'],'C181');s=z.json('SUMMARY.json');expected=expected181(a,d,c)
  for k,v in expected.items():compare(s[k],v,'Independent C181 summary/'+k)
  compare(set(s),set(expected)|{'errors','stage','elapsed_worker_seconds','process_peak_RSS_KiB','original_inputs_unchanged'},'Original complete C181 summary schema');compare(s['errors'],[]);compare(s['stage'],'complete');need(math.isfinite(s['elapsed_worker_seconds']) and s['elapsed_worker_seconds']>=0,'Original elapsed declaration');need(type(s['process_peak_RSS_KiB']) is int and s['process_peak_RSS_KiB']>=0,'Original RSS declaration');compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS});compare(z.json('DESIGN_FROZEN_BEFORE_AUDIT.json'),prior.design());links=z.json('ORIGINAL_INPUTS.json');compare(set(links),set(prior.INPUT_KEYS))
  for k in prior.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'Original C181 ancestry/'+k)
  compare(z.json('diagnostics/DIAGNOSTIC_SUMMARY.json'),d,'Original C181 diagnostic duplicate');checked=[]
  if reconstructed is not None:
   reconstructed=Path(reconstructed)
   for folder in ('C180_arithmetic','diagnostics'):
    for p in sorted((reconstructed/folder).rglob('*')):
     if not p.is_file():continue
     n=p.relative_to(reconstructed).as_posix()
     if p.suffix=='.json':compare(z.json(n),clean_json(p.read_bytes()),'Reconstructed original C181 artifact/'+n)
     elif p.suffix=='.csv':
      import csv,io
      old=list(csv.reader(io.StringIO(z.raw(n).decode('utf-8'))));new=list(csv.reader(io.StringIO(p.read_text())))
      compare(len(old),len(new),'CSV row count/'+n)
      for x,y in zip(old,new):
       compare(len(x),len(y),'CSV column count/'+n)
       for u,v in zip(x,y):
        try:uf=float(u);vf=float(v)
        except ValueError:compare(u,v,'CSV literal/'+n)
        else:need(math.isfinite(uf) and math.isfinite(vf),'Finite reconstructed CSV');compare(uf,vf,'CSV numeric/'+n)
     else:compare(z.raw(n),p.read_bytes(),'Reconstructed original C181 artifact/'+n)
     checked.append(n)
   need(len(checked)>=29,'Full reconstructed C181 arithmetic and diagnostic artifact inventory');compare(sum(n.startswith('C180_arithmetic/precision/') for n in checked),5,'All original C181 reconstructed precision artifacts')
 r=dict(protocol=prior.PROTOCOL,decision=s['decision'],members_hashed=members,C180_original_independently_reaudited=True,C181_scientific_summary_reconstructed=True,reconstructed_artifacts_bound=checked,new_model_forwards=0,new_LP_calls=0)
 dump(out/'C181_ORIGINAL_REBOUND.json',r);return r

def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original before payload/'+k)
 for n,h in c['source181'].items():compare(sha(ROOT/'vendor181'/n),h,'Delivered trusted C181 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load180({k:paths[k] for k in previous_training.INPUT_KEYS},out/'twenty_five_originals',c['parent']['parent'],budget)
  recon=out/'C181_independent_reconstruction';a=audit(paths['C180'],ctx,recon/'C180_arithmetic',c['parent'],budget);d=diagnostics(a,recon/'diagnostics');budget();receipt=bind181(paths['C181'],a,d,out/'C181_binding',c,recon)
 p=a['heads'][-1].copy()
 if c['production']:compare(p.shape,(3,2,36));compare(parameter_hash(p),ANCHOR_SHA256,'Predetermined C180 step8 anchor')
 ctx.update(C181_receipt=receipt,C180_final_head=p,C180_final_FP64=a['frames'][-1],C180_final_native={str(t):a['native']['native_final'+str(t)] for t in (0,1)})
 dump(out/'TWENTY_SEVEN_ORIGINALS_BOUND_RECEIPT.json',dict(all_twenty_seven_originals_independently_bound=c['production'],C181_decision=receipt['decision'],initial_C180_final_head_digest=parameter_hash(p),new_model_forwards=0,new_LP_calls=0));return ctx
