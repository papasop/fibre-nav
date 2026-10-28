"""Bind the five originals and independently check archived C136 arithmetic."""
import contextlib
from core137 import *
import core136 as old
from inputs136 import load as load136
from worker136 import final_summaries as summaries136

def audit136(z,data,out,c):
    members=z.verify_manifest();s=z.json('SUMMARY.json')
    compare(s['protocol'],old.PROTOCOL);compare(s['complete'],True)
    if c['production']:
        compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
        compare(s['decision'],'C136_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
    for k in ('candidate_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified','best_checkpoint_selected','recovery_checkpoint_selected'):
        compare(s[k],False,'C136/'+k)
    for k in ('repeatable','raw_control_replay_consistent','step16_frozen_before_final_evaluation','functional_zero_parity_complete_before_gradients','all12_used_for_training'):compare(s[k],True,'C136/'+k)
    compare(s['completed_training_steps'],16);compare(s['new_backward_calls'],192);compare(s['final_endpoint_records'],48)
    source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source136']),'C136 delivered source inventory changed')
    for n,h in c['source136'].items():need(hashlib.sha256(z.raw('source/'+n)).hexdigest()==h,'C136 delivered source changed: '+n)
    compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),old.design(),'C136 design')
    for k in old.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C136 linked '+k)
    training=[dict(**r,C136_membership='training',independent_validation_eligible=False,prepared_sha256=safe.tensor_hash(data['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][r['state_id']]['raw'])) for r in registry()]
    compare(z.json('ALL12_TRAINING_REGISTRY_FROZEN.json'),training,'C136 all12 registry');compare(z.json('MODEL_PINS_REFERENCE.json'),data['modelpins'],'C136 model pins')
    exact(z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],data['V'],'C136 basis')
    rank=len(data['V']);a=np.zeros(rank);m=a.copy();v=a.copy();exact(z.arrays('INITIAL_ZERO_CONTROL.npz')['a'],a,'C136 initial zero')
    controls=z.json('PREFLIGHT_PROGRESS.json');need(len(controls)==36,'C136 native preflight denominator')
    archived={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'};fresh={}
    for r,(spec,arm) in zip(controls,[(spec,arm) for spec in registry() for arm in (None,*ARMS[:2])]):
        for k,x in spec.items():compare(r[k],x,'C136 preflight/'+k)
        sid=spec['state_id'];p=data['points'][sid];compare(r['C136_membership'],'training')
        if arm is None:
            compare(r['kind'],'prepared');safe.metric_gates(r['metrics'],spec['case'],p['prepared'],False);refs=[p['preparation']]
        else:
            compare(r['kind'],'endpoint');compare(r['arm'],arm);gate_record(r['metrics'],spec['case'],data['maps'][(sid,arm)]['point'],p['prepared'])
            refs=[archived[(sid,arm,k)] for k in (0,1)]+[data['C135_controls'][(sid,arm,k)] for k in (0,1)]
            if arm==ARMS[0]:fresh[sid]=r['metrics']
        checks=[safe.repeat_comparison(r['metrics'],ref) for ref in refs];compare(r['comparisons'],checks,'C136 native comparisons');need(all(q['within_replay_tolerance'] and q['same_point_identity'] for q in checks),'C136 archived native drift')
    zeros=z.json('ZERO_FUNCTIONAL_PROGRESS.json');need(len(zeros)==12,'C136 zero denominator')
    for spec,r in zip(registry(),zeros):
        sid=spec['state_id'];compare(r['state_id'],sid);compare(r['point_sha256'],safe.tensor_hash(data['points'][sid]['raw']))
        err=close(r['observed'],fresh[sid]['observed'],'C136 zero observable',2e-6);klerr=abs(r['KL']-fresh[sid]['KL']);need(klerr<=2e-7,'C136 zero KL');compare(r['observed_parity_Linf'],err);compare(r['KL_parity_absolute_error'],klerr)
    progress=z.json('TRAINING_PROGRESS.json');need(len(progress)==16,'C136 full16 updates required')
    for step,rec in enumerate(progress,1):
        name=f'training/step{step:02d}/UPDATE.npz';f=z.arrays(name);rows=z.json(f'training/step{step:02d}/OBSERVATIONS.json')
        need(set(f)=={'before_a','before_m','before_v','full_mean_gradient','after_a','after_m','after_v'},'C136 update fields');compare(z.json(f'training/step{step:02d}/GRADIENT_PROGRESS.json'),rows,'C136 gradient progress')
        need(len(rows)==12,'C136 gradient denominator');exact(f['before_a'],a,'C136 before a');exact(f['before_m'],m,'C136 before m');exact(f['before_v'],v,'C136 before v')
        for spec,r in zip(registry(),rows):
            sid=spec['state_id'];p=data['points'][sid];compare(r['state_id'],sid);compare(r['case'],spec['case']);compare(r['C136_membership'],'training');compare(r['coefficient_digest'],row_digest(a))
            point=old.proposal(p['prepared'],p['raw'],data['V'],a)[0]
            # A functional CPU BLAS map can differ from NumPy by <=2e-6; its
            # saved point digest is not independently reconstructible from a digest.
            need(type(r['functional_point_sha256']) is str and len(r['functional_point_sha256'])==64,'C136 functional point digest missing')
            need(0<=r['point_parity_error_Linf']<=2e-6,'C136 functional parity record')
            parts=old.objective(r['observed'],r['KL'],point,p['prepared'],a,spec['case']);compare(r['objective_parts'],parts,'C136 objective parts',atol=1e-10,rtol=1e-9);compare(r['objective'],sum(parts.values()),'C136 loss',atol=1e-10,rtol=1e-9);farray(np.asarray(r['gradient_contribution']),(rank,))
        reconstructed_g=np.sum([r['gradient_contribution'] for r in rows],axis=0);close(f['full_mean_gradient'],reconstructed_g,'C136 archived gradient sum',1e-12);g=f['full_mean_gradient']
        aa,mm,vv,info=old.adam_step(a,g,m,v,step)
        for key,x in (('after_a',aa),('after_m',mm),('after_v',vv)):close(f[key],x,'C136 update/'+key,2e-15)
        expected=dict(**info,before_digest=row_digest(a),after_digest=row_digest(f['after_a']),mean_objective_before_update=float(np.mean([r['objective'] for r in rows])),maximum_objective_before_update=float(max(r['objective'] for r in rows)),update_file=name,update_sha256=hashlib.sha256(z.raw(name)).hexdigest(),n_training_records=12)
        compare(rec,expected,'C136 update record');a,m,v=f['after_a'],f['after_m'],f['after_v']
    final=z.arrays('FINAL_SHARED_CONTROL_STEP16.npz');need(set(final)=={'a'},'C136 final vector fields');exact(final['a'],a,'C136 must be step16');need(np.max(abs(a))<=GAIN_BOUND,'C136 initial bounds')
    saved=z.arrays('ALL24_FROZEN_FINAL_POINTS.npz');outcomes=z.json('ALL24_FROZEN_FINAL_OUTCOMES.json');need(len(outcomes)==24,'C136 outcome count');points={};expected_outcomes=[]
    for spec in registry():
        sid=spec['state_id'];p=data['points'][sid];trained,diag=old.proposal(p['prepared'],p['raw'],data['V'],a)
        for arm,point in ((old.EVAL_ARMS[0],p['raw']),(old.EVAL_ARMS[1],trained)):
            key=sid+'__'+arm;need(key in saved,'C136 point missing');exact(saved[key],point,'C136 final committed point');points[(sid,arm)]=point
            expected_outcomes.append(dict(**spec,C136_membership='training',independent_validation_eligible=False,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=diag if arm==old.EVAL_ARMS[1] else {'original_raw_bytes_retained':True}))
    need(set(saved)=={sid+'__'+arm for sid,arm in points},'C136 point keys');compare(outcomes,expected_outcomes,'C136 final outcomes')
    freeze=z.json('STEP16_FREEZE_BEFORE_EVALUATION.json');compare(freeze['completed_steps'],16);compare(freeze['control_digest'],row_digest(a));compare(freeze['checkpoint_choice'],'Fixed step16 only');compare(freeze['no_best_or_recovery_selection'],True)
    for field,name in (('control_sha256','FINAL_SHARED_CONTROL_STEP16.npz'),('basis_sha256','FROZEN_SHARED_ROW_BASIS.npz'),('training_registry_sha256','ALL12_TRAINING_REGISTRY_FROZEN.json')):compare(freeze[field],hashlib.sha256(z.raw(name)).hexdigest())
    expected_seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL12_TRAINING_REGISTRY_FROZEN.json','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_ZERO_CONTROL.npz','FINAL_SHARED_CONTROL_STEP16.npz','ALL24_FROZEN_FINAL_POINTS.npz','ALL24_FROZEN_FINAL_OUTCOMES.json'}
    need(set(freeze['proposal_seals'])==expected_seals,'C136 freeze seal coverage')
    for n,h in freeze['proposal_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C136 freeze/'+n)
    records=z.json('ALL48_FINAL_TRAINING_ENDPOINTS.json');need(len(records)==48,'C136 final denominator');compare(z.json('FINAL_ENDPOINT_PROGRESS.json'),records,'C136 final progress');first={};controls136={}
    for i,(r,(rnd,spec,arm)) in enumerate(zip(records,[(rnd,spec,arm) for rnd in (0,1) for spec in registry() for arm in old.EVAL_ARMS])):
        sid=spec['state_id'];p=data['points'][sid];point=points[(sid,arm)]
        for k,x in dict(**spec,arm=arm,round=rnd,C136_membership='training',independent_validation_eligible=False).items():compare(r[k],x,'C136 endpoint identity/'+k)
        compare(z.json(f'final_measurements/record{i:02d}.json'),r,'C136 individual endpoint');compare(r['recomputed_gates'],gate_record(r['metrics'],spec['case'],point,p['prepared']),'C136 endpoint gates')
        aa=np.zeros_like(a) if arm==old.EVAL_ARMS[0] else a;parts=old.objective(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],aa,spec['case']);compare(r['training_objective_parts'],parts,'C136 final objective parts');compare(r['training_objective'],sum(parts.values()))
        if arm==old.EVAL_ARMS[0]:
            q=safe.repeat_comparison(r['metrics'],fresh[sid]);compare(r['raw_control_replay'],q);need(q['within_replay_tolerance'] and q['same_point_identity'],'C136 raw replay')
        if rnd==0:first[(sid,arm)]=r['metrics']
        else:
            q=safe.repeat_comparison(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q);need(q['within_replay_tolerance'] and q['same_point_identity'],'C136 repeat')
        controls136[(sid,arm,rnd)]=r
    sums,pairs=summaries136(records);compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(s['final_training_summaries'],sums);compare(z.json('PAIRED_VS_RAW.json'),pairs)
    for n in ('RAW_CONTROL_FAILURES.json','REPEATABILITY_FAILURES.json'):compare(z.json(n),[])
    passed=all(sums['all12'][str(rnd)][old.EVAL_ARMS[1]]['passed']==12 for rnd in (0,1));compare(s['fixed_final_training_qualification_passed'],passed)
    compare(s['decision'],'C136_FIXED_FINAL_ALL12_TRAINING_GATES_PASSED' if passed else 'C136_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
    dump(out/'C136_BOUND_RECEIPT.json',dict(parent_decision=s['decision'],parent_decision_reclassified=False,members_hashed=members,source_members_checked=len(source),training_updates_checked=16,training_rows_checked=192,final_records_checked=48,all24_points_reconstructed=True,archived_gradients_independently_recomputed=False,functional_point_hashes_independently_reconstructed=False,scope='Recorded arithmetic and identities; no independent model, derivative, chronology or non-access verification.'))
    data.update(initial_a=a.copy(),C136_controls=controls136,C136_summary=s)

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Five fixed originals required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    prior=out/'C132_C135_BINDINGS';prior.mkdir();data=load136({k:paths[k] for k in old.INPUT_KEYS},prior,dict(c,pins={k:c['pins'][k] for k in old.INPUT_KEYS}))
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C137=True) for k in INPUT_KEYS})
    with contextlib.closing(Archive(paths['C136'])) as z:audit136(z,data,out,c)
    training=[dict(**r,C137_membership='training',independent_validation_eligible=False,prepared_sha256=safe.tensor_hash(data['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][r['state_id']]['raw'])) for r in registry()]
    dump(out/'ALL12_TRAINING_REGISTRY_FROZEN.json',training);dump(out/'MODEL_PINS_REFERENCE.json',data['modelpins']);np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=data['V'])
    np.savez_compressed(out/'INITIAL_C136_STEP16_CONTROL.npz',a=data['initial_a'],m=np.zeros_like(data['initial_a']),v=np.zeros_like(data['initial_a']))
    print('C137_INPUT_BINDINGS_OK; retained C136 decision; new optimizer starts from its fixed step16 with zero moments.',flush=True)
    return data
