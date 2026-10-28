"""Validate C155/C156 with trusted C157 reader and bind the exact failed C157 R2 archive."""
from core158 import *
from inputs157 import load as previous_load
from audit147 import source_check
from baseline157 import record_comparison
def audit_failed157(path,context,out,c):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with previous.parent.parent.prior.no_models(),safe.no_solvers(),contextlib.closing(Archive(path)) as z:
        count=z.verify_manifest();source_check(z,c['source157'],'C157_R2');s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],False);compare(s['decision'],'C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');compare(s['delivery_revision'],'R2_BASELINE_RECORDING_ONLY');compare(s['synthetic_fixture'],not c['production']);compare(s['stage'],'pinned_model_and_baseline_replay')
        for k,v in dict(new_model_forwards=1,new_backward_calls=0,focus_training_states_evaluated=0,new_training_steps=0,new_head_fits=0,excluded_input_states_evaluated=0).items():compare(s[k],v,'Original C157 stop/'+k)
        for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','thresholds_changed','parent_decision_reclassified'):compare(s[k],False)
        compare(s['original_inputs_unchanged'],dict(C155=True,C156=True));links=z.json('ORIGINAL_INPUTS.json')
        for k in ('C155','C156'):compare(links[k]['sha256'],c['pins'][k]['sha256'],'C157 original input link/'+k)
        dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',previous.design());points,dirs,requests,seals=previous.freeze(context['data'],out)
        saved=z.arrays('ALL14_FP64_CHORD_POINTS.npz');compare(set(saved),{sid+'__'+n for sid,n in points})
        for (sid,n),p in points.items():exact(saved[sid+'__'+n],p,'Original C157 fixed point')
        saved_dirs=z.arrays('TWO_FIXED_DIRECTIONS.npz');compare(set(saved_dirs),set(dirs))
        for sid,p in dirs.items():exact(saved_dirs[sid],p,'Original C157 actual direction')
        compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design());compare(z.json('REQUESTS_FROZEN.json'),requests);compare(z.json('POINT_BINDINGS_AND_PATH_DIFFERENCES.json'),clean_json((out/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json').read_bytes()))
        for n,h in z.json('SEALED_BEFORE_MODEL.json')['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C157 archived seal/'+n)
        comparisons={}
        for label,expected in [('NATIVE_FP32_BASELINE_LOGITS',context['baseline_logits']['FP32_logits_cast64']),('NATIVE_FP32_CACHED_RESPONSE',np.asarray(context['baseline']['FP32_cached_response_reference'],float))]:
            arrays=z.arrays('model_runtime/'+label+'.npz');compare(set(arrays),{'actual','archived'});exact(arrays['archived'],expected,'C157 stored original baseline/'+label);report=record_comparison(out,label,arrays['actual'],arrays['archived']);compare(z.json('model_runtime/'+label+'.json'),report,'C157 failed baseline arithmetic/'+label);comparisons[label]=report
        need(comparisons['NATIVE_FP32_BASELINE_LOGITS']['passed_original_replay_tolerance'] is False,'Exact failed C157 replay branch required');compare(s['baseline_replay_diagnostics']['NATIVE_FP32_BASELINE_LOGITS'],comparisons['NATIVE_FP32_BASELINE_LOGITS'])
        env=z.json('model_runtime/BASELINE_RUNTIME_PROVENANCE.json');compare(env['archived_C155_MODEL_PINS'],context['modelpins'],'C157 archived pin binding')
        dump(out/'FAILED_C157_AUDIT_RECEIPT.json',dict(members_hashed=count,whole_zip_sha256=sha(path),C157_original_decision=s['decision'],C157_reclassified=False,archived_forward_count=1,archived_backward_count=0,actual14_points_verified=True,original_baseline_comparisons_recomputed=comparisons,original_runtime_provenance=env,archived_source_executed=False,measurement_accuracy_established=False))
    return dict(decision=s['decision'],comparisons=comparisons)
def load(paths,out,c=None,budget=lambda:None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS),'Exactly C155/C156/C157 originals')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed whole original before payload/'+k)
    for n,h in c['source157'].items():compare(sha(ROOT/'vendor157'/n),h,'Trusted C157 R2 source/'+n)
    context=previous_load({k:paths[k] for k in ('C155','C156')},out/'C155_C156',c['parent'],budget);budget();context['failed157']=audit_failed157(paths['C157'],context,out/'C157_R2',c);budget();return context
