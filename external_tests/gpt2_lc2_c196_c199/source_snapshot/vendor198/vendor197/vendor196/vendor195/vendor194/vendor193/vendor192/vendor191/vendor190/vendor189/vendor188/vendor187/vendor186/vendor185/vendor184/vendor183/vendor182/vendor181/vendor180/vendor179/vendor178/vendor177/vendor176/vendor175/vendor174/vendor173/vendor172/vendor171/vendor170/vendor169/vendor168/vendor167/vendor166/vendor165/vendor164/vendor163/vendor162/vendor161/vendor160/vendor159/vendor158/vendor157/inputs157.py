"""Rehash both originals, rerun trusted C156 arithmetic; never execute ZIP source."""
from core157 import *
from audit156 import bind,baseline,recorded_gradients,analyse as analyse156
from audit147 import source_check
def load(paths,out,c=None,budget=lambda:None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    compare(set(paths),set(INPUT_KEYS),'Exactly C155 and C156 originals')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Original whole ZIP before payload/'+k)
    for n,h in c['source156'].items():compare(sha(ROOT/'vendor156'/n),h,'Trusted delivered C156 source/'+n)
    with parent.parent.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(paths['C155'])) as z:
        data=bind(z,c['parent'],budget);base=baseline(z,c['parent'],data);gradient=recorded_gradients(data);traj=analyse156(z,data,out/'RECOMPUTED_C156');pins=z.json('model_runtime/MODEL_PINS.json');baseline_logits=z.arrays('model_runtime/FP64_BASELINE_LOGITS.npz')
    with parent.parent.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(paths['C156'])) as z:
        count=z.verify_manifest();source_check(z,c['source156'],'C156');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['errors'],[]);compare(s['synthetic_fixture'],not c['production']);compare(s['decision'],'C156_READONLY_TRAJECTORY_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');compare(s['parent_decision'],data['parent']['decision']);compare(s['parent_decision_reclassified'],False)
        for k,v in dict(training_states_audited=84,records_audited=1530,actual_frozen_point_entries_verified=588,actual_frozen_coefficient_tensors_verified=7,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0).items():compare(s[k],v,'C156 denominator/count '+k)
        for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','thresholds_changed','writer_deployed','AD_accuracy_certified'):compare(s[k],False,'C156 boundary/'+k)
        compare(s['baseline_reconstruction'],base,'Recomputed C156 baseline');compare(s['archived_gradient_arithmetic'],gradient,'C156 archived arithmetic only');compare(s['trajectory_diagnostics'],traj,'Recomputed C156 entire trajectory summary')
        compare(s['original_C155_input_unchanged'],True,'C156 declared original unchanged');compare(z.json('BASELINE_REFERENCE_RECONSTRUCTION.json'),base);compare(z.json('ARCHIVED_GRADIENT_ARITHMETIC_ONLY.json'),gradient);compare(z.json('PRECISION_TRAJECTORY_SUMMARY.json'),traj);compare(z.json('INPUT_FROZEN.json')['sha256'],c['pins']['C155']['sha256'],'C156 parent whole-archive binding')
        for name in ['ALL84_RECOMPUTED_COORDINATE_TRAJECTORIES.json','TWO_FOCUS_COORDINATE_PRECISION_ANALYSIS.json','ALL84_SEALED_POINT_DISPLACEMENT_AND_ROUNDING.json']:
            compare(z.json(name),clean_json((out/'RECOMPUTED_C156'/name).read_bytes()),'C156 full saved diagnostic '+name)
    dump(out/'INPUT_AUDIT_RECEIPT.json',dict(C155_members_hashed=data['members_hashed'],C156_members_hashed=count,C155_records_rebound=1530,actual_points_reconstructed=588,C156_summary_recomputed=True,ten_predecessor_original_ZIPs_rehashed=False,archived_source_executed=False,model_measurement_accuracy_established=False))
    refs={}
    for sid in FOCUS:
        for n in ('zero','plus_one'):
            row=data['index'][('functional_FP64',n,sid,'preflight' if n=='zero' else 'probe')]['precision_measurement'];refs[(sid,n)]=farray(np.array(row['observed_local_reference'][:6])+np.array(base['FP64_response_reference']),(6,))
    return dict(data=data,modelpins=pins,baseline=base,baseline_logits=baseline_logits,replay_absolute=refs,archived_gradient_arithmetic=gradient)
