"""Trusted-source C140 arithmetic audit and immutable frozen candidate bindings."""
import contextlib,re
from core141 import *
from inputs140 import load as load140
from worker140 import final_summaries as summaries140

def audit140(z,d,out,c):
    count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C140_FIXED_FINAL_ALL36_TRAINING_GATES_PASSED')
    if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
    for k in ('all36_used_for_training','fixed_final_training_qualification_passed','repeatable','controls_replay_consistent','functional_zero_parity_complete_before_gradients','functional_warm_parity_complete_before_gradients','step12_frozen_before_final_evaluation'):compare(s[k],True,'C140/'+k)
    for k in ('candidate_selected','writer_deployed','best_checkpoint_selected','recovery_checkpoint_selected','thresholds_changed','parent_decision_reclassified','LC2_closed','LC2_single_fold_passed'):compare(s[k],False,'C140/'+k)
    source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source140']),'C140 source inventory')
    for n,h in c['source140'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'C140 source/'+n)
    compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design())
    for k in parent.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C140 linked '+k)
    expected_registry=[dict(r,prepared_sha256=safe.tensor_hash(d['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(d['points'][r['state_id']]['raw'])) for r in parent.registry()];compare(z.json('ALL36_TRAINING_REGISTRY_FROZEN.json'),expected_registry);compare(z.json('MODEL_PINS_REFERENCE.json'),d['modelpins']);compare(z.json('FROZEN_INPUT_IDENTITIES.json'),parent.frozen_digest(d))
    arrays=z.arrays('ALL36_TRAINING_INPUTS.npz');need(set(arrays)=={sid+'__'+k for sid in d['points'] for k in ('prepared','raw','features')},'C140 all36 input coverage')
    for sid,p in d['points'].items():
        for k in ('prepared','raw','features'):exact(arrays[sid+'__'+k],p[k],'C140 frozen training input')
    exact(z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],d['V'],'C140 frozen basis')
    for n,obj in (('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])):
        actual=z.arrays(n);need(set(actual)==set(obj),'C140 original head schema')
        for k,v in obj.items():exact(actual[k],v,'C140 original head/'+k)
    a=d['initial_a'].copy();m=np.zeros_like(a);v=np.zeros_like(a);ini=z.arrays('INITIAL_C137_STEP24_CONTROL.npz');need(set(ini)=={'a','m','v'},'C140 optimizer reset fields')
    for k,x in (('a',a),('m',m),('v',v)):exact(ini[k],x,'C140 initial/'+k)
    pre=z.json('PREFLIGHT_PROGRESS.json');need(len(pre)==108,'C140 all108 controls');raw={};warm={}
    for r,(sp,kind) in zip(pre,[(sp,k) for sp in parent.registry() for k in ('prepared','raw','warm')]):
        sid=sp['state_id'];p=d['points'][sid]
        for k,x in dict(**sp,kind=kind).items():compare(r[k],x)
        point=p['prepared'] if kind=='prepared' else (p['raw'] if kind=='raw' else parent.proposal(p['prepared'],p['raw'],d['V'],a)[0]);met=r['metrics']
        if kind=='prepared':need(safe.metric_gates(met,sp['case'],point,False),'C140 valid training preparation')
        else:parent.parent.gate_record(met,sp['case'],point,p['prepared'])
        checks=[safe.repeat_comparison(met,ref) for ref in d['native_refs'][(sid,kind)]];compare(r['comparisons'],checks);need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in checks),'C140 control replay drift')
        if kind=='raw':raw[sid]=met
        if kind=='warm':warm[sid]=met
    for n,refs,aa in (('ZERO_FUNCTIONAL_PROGRESS.json',raw,np.zeros_like(a)),('WARM_FUNCTIONAL_PROGRESS.json',warm,a)):
        rows=z.json(n);need(len(rows)==36,'C140 full36 functional parity')
        for sp,r in zip(parent.registry(),rows):
            sid=sp['state_id'];compare(r['state_id'],sid);err=close(r['observed'],refs[sid]['observed'],'C140 functional observation parity',2e-6);ke=abs(r['KL']-refs[sid]['KL']);need(ke<=2e-7,'C140 functional KL parity');compare(r['observed_parity_Linf'],err);compare(r['KL_parity_absolute_error'],ke);need(0<=r['point_parity_error_Linf']<=2e-6,'C140 functional point parity')
            pp=parent.proposal(d['points'][sid]['prepared'],d['points'][sid]['raw'],d['V'],aa)[0]
            if n.startswith('ZERO'):compare(r['point_sha256'],safe.tensor_hash(pp))
            else:compare(r['native_point_sha256'],safe.tensor_hash(pp))
    progress=z.json('TRAINING_PROGRESS.json');need(len(progress)==12,'C140 all12 updates')
    for step,rec in enumerate(progress,1):
        name=f'training/step{step:02d}/UPDATE.npz';f=z.arrays(name);rows=z.json(f'training/step{step:02d}/OBSERVATIONS.json');need(len(rows)==36 and set(f)=={'before_a','before_m','before_v','full_weighted_gradient','after_a','after_m','after_v'},'C140 update coverage')
        for k,x in (('before_a',a),('before_m',m),('before_v',v)):exact(f[k],x,'C140 update chain/'+k)
        for sp,r in zip(parent.registry(),rows):
            sid=sp['state_id'];p=d['points'][sid];compare(r['state_id'],sid);compare(r['case'],sp['case']);compare(r['C140_membership'],'training');compare(r['coefficient_digest'],row_digest(a));point=parent.proposal(p['prepared'],p['raw'],d['V'],a)[0];terms=parent.objective(r['observed'],r['KL'],point,p['prepared'],a,sp['case']);compare(r['objective_parts'],terms,atol=1e-10,rtol=1e-9);compare(r['objective'],sum(terms.values()),atol=1e-10,rtol=1e-9);farray(np.array(r['state_gradient']),(len(a),));need(0<=r['point_parity_error_Linf']<=2e-6,'C140 training point parity')
            if r['point_parity_error_Linf']==0:compare(r['functional_point_sha256'],safe.tensor_hash(point),'C140 exact functional point binding')
        g,agg=parent.aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows]);compare(z.json(f'training/step{step:02d}/AGGREGATION.json'),agg);compare(z.json(f'training/step{step:02d}/GRADIENT_PROGRESS.json'),rows)
        for r,w in zip(rows,agg['weights']):compare(r['aggregate_weight'],w);close(r['gradient_contribution'],w*np.array(r['state_gradient']),'C140 weighted row',1e-12)
        close(f['full_weighted_gradient'],g,'C140 full weighted gradient',1e-12);aa,mm,vv,info=parent.adam_step(a,f['full_weighted_gradient'],m,v,step)
        for k,x in (('after_a',aa),('after_m',mm),('after_v',vv)):close(f[k],x,'C140 Adam arithmetic/'+k,2e-15)
        compare(rec,dict(**info,aggregate_objective_before_update=agg['objective'],state_weights=agg['weights'],before_digest=row_digest(a),after_digest=row_digest(f['after_a']),mean_objective_before_update=agg['mean_state_loss'],maximum_objective_before_update=agg['maximum_state_loss'],update_file=name,update_sha256=hashlib.sha256(z.raw(name)).hexdigest(),n_training_records=36));a,m,v=f['after_a'],f['after_m'],f['after_v']
    final=z.arrays('FINAL_SHARED_CONTROL_STEP12.npz');need(set(final)=={'a'},'C140 final fields');exact(final['a'],a,'C140 final must equal fixed update12');saved=z.arrays('ALL108_FROZEN_FINAL_POINTS.npz');points={};outcomes=[]
    for sp in parent.registry():
        sid=sp['state_id'];p=d['points'][sid];trained,diag=parent.proposal(p['prepared'],p['raw'],d['V'],a);ww,wd=parent.proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])
        for arm,point,dd in ((parent.EVAL_ARMS[0],p['raw'],dict(original_raw_bytes_retained=True)),(parent.EVAL_ARMS[1],ww,wd),(parent.EVAL_ARMS[2],trained,diag)):
            exact(saved[sid+'__'+arm],point,'C140 final map');points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=dd))
    need(set(saved)=={sid+'__'+arm for sid,arm in points},'C140 all108 final points');compare(z.json('ALL108_FROZEN_FINAL_OUTCOMES.json'),outcomes)
    freeze=z.json('STEP12_FREEZE_BEFORE_EVALUATION.json');compare(freeze['completed_steps'],12);compare(freeze['control_sha256'],hashlib.sha256(z.raw('FINAL_SHARED_CONTROL_STEP12.npz')).hexdigest());compare(freeze['control_digest'],row_digest(a));compare(freeze['checkpoint_choice'],'Fixed C140 step12 only');compare(freeze['no_best_or_recovery_selection'],True)
    seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL36_TRAINING_REGISTRY_FROZEN.json','ALL36_TRAINING_INPUTS.npz','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C137_STEP24_CONTROL.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','FROZEN_INPUT_IDENTITIES.json','C139_BOUND_RECEIPT.json','FINAL_SHARED_CONTROL_STEP12.npz','ALL108_FROZEN_FINAL_POINTS.npz','ALL108_FROZEN_FINAL_OUTCOMES.json'};need(set(freeze['proposal_seals'])==seals,'C140 freeze coverage')
    for n,h in freeze['proposal_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C140 frozen seal/'+n)
    records=z.json('ALL216_FINAL_TRAINING_ENDPOINTS.json');compare(z.json('FINAL_ENDPOINT_PROGRESS.json'),records);need(len(records)==216,'C140 all216 endpoint records');first={};controls={}
    for i,(r,(rnd,sp,arm)) in enumerate(zip(records,[(rnd,sp,arm) for rnd in (0,1) for sp in parent.registry() for arm in parent.EVAL_ARMS])):
        sid=sp['state_id'];p=d['points'][sid];point=points[(sid,arm)]
        for k,x in dict(**sp,round=rnd,arm=arm).items():compare(r[k],x)
        compare(z.json(f'final_measurements/record{i:03d}.json'),r);compare(r['recomputed_gates'],parent.parent.gate_record(r['metrics'],sp['case'],point,p['prepared']));aaa=np.zeros_like(a) if arm==parent.EVAL_ARMS[0] else (d['initial_a'] if arm==parent.EVAL_ARMS[1] else a);terms=parent.objective(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],aaa,sp['case']);compare(r['training_objective_parts'],terms);compare(r['training_objective'],sum(terms.values()))
        if arm in parent.EVAL_ARMS[:2]:
            q=safe.repeat_comparison(r['metrics'],raw[sid] if arm==parent.EVAL_ARMS[0] else warm[sid]);compare(r['native_control_replay'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C140 raw/C137 control drift')
        if rnd==0:first[(sid,arm)]=r['metrics']
        else:
            q=safe.repeat_comparison(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C140 repeat drift')
        controls[(sid,arm,rnd)]=r
    sums,pairs=summaries140(records);compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(s['final_training_summaries'],sums);compare(z.json('PAIRED_VS_RAW_AND_C137.json'),pairs);compare(s['paired_vs_raw_and_C137'],pairs)
    for j in (0,1):need(sums['all36'][str(j)][parent.EVAL_ARMS[2]]['passed']==36,'C140 all36 qualification required')
    for n in ('CONTROL_REPLAY_FAILURES.json','REPEATABILITY_FAILURES.json'):compare(z.json(n),[])
    for k,x in (('completed_training_steps',12),('new_backward_calls',432),('final_endpoint_records',216),('new_model_forwards',1657)):compare(s[k],x)
    dump(out/'C140_BOUND_RECEIPT.json',dict(parent_decision=s['decision'],parent_decision_reclassified=False,members_hashed=count,source_members_checked=len(source),training_updates_checked=12,training_rows_checked=432,final_endpoints_checked=216,final_vector_and_all108_points_reconstructed=True,archived_gradients_independently_recomputed=False,chronology_or_measurement_accuracy_independently_certified=False))
    d.update(a140=a.copy(),C140_points=points,C140_controls=controls,C140_summary=s)

HASH_KEYS={'theta_sha256','point_sha256','prepared_sha256','raw_sha256','candidate_sha256','parent_sha256','source_full_point_sha256','native_point_sha256','functional_point_sha256'}
def explicit_coordinate_hashes(z):
    values=set()
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in HASH_KEYS and isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v):values.add(v)
                elif isinstance(v,(dict,list)):walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for n in z.z.namelist():
        if n.endswith('.json') and not n.startswith('source/') and n!='RESULT_MANIFEST.json':walk(z.json(n))
    return values

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Nine original scientific archives required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    prior=out/'C132_C139_BINDINGS';prior.mkdir();d=load140({k:paths[k] for k in parent.INPUT_KEYS},prior,dict(c,pins={k:c['pins'][k] for k in parent.INPUT_KEYS}))
    with contextlib.closing(Archive(paths['C140'])) as z:audit140(z,d,out,c)
    known=set(d['historical_hashes'])|{safe.tensor_hash(p[k]) for p in d['points'].values() for k in ('prepared','raw')}|{safe.tensor_hash(p) for p in d['C140_points'].values()};counts={}
    for k in INPUT_KEYS:
        with contextlib.closing(Archive(paths[k])) as z:hh=explicit_coordinate_hashes(z);known|=hh;counts[k]=len(hh)
    d['historical_hashes']=known;dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C141=True) for k in INPUT_KEYS});dump(out/'BOUND_HISTORICAL_COORDINATE_HASHES.json',dict(hashes=sorted(known),per_archive_explicit_coordinate_hash_counts=counts,exhaustive_programme_history_verified=False,scope='Bound36 training inputs, C140 final points and explicit coordinate-hash fields in nine originals; no global historical non-access certificate'));return d
