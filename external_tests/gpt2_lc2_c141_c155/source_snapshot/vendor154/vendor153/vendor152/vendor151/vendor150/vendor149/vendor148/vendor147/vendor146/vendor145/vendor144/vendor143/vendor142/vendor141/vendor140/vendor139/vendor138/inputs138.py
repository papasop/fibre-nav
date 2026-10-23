"""Trusted-source arithmetic audit of C137 and immutable proposal inputs."""
import contextlib
from core138 import *
from inputs137 import load as load137
from worker137 import final_summaries as summaries137

def validate_frozen_inputs(writer,control):
    required={'mean','scale','training_z','coefficients','delta_mean'}
    optional={'J_mean':(22,DIM),'B_all':(DIM,22),'B_response':(DIM,6),'response_offset':(6,)}
    actual=set(writer);missing=required-actual;unknown=actual-required-set(optional)
    need(not missing and not unknown,'Frozen raw writer schema: missing='+str(sorted(missing))+' unknown='+str(sorted(unknown))+' actual='+str(sorted(actual)))
    before=safe.head_digest(writer)
    for name,shape in (('mean',(3147,)),('scale',(3147,)),('delta_mean',(DIM,))):farray(writer[name],shape)
    need(np.asarray(writer['training_z']).ndim==2,'Frozen training_z rank');n=len(writer['training_z'])
    need(n>0 and np.all(writer['scale']>0),'Frozen writer rows/scale');farray(writer['training_z'],(n,3147));farray(writer['coefficients'],(n,DIM))
    for name,shape in optional.items():
        if name in writer:farray(writer[name],shape)
    need('delta_sketch' in control,'Frozen control delta_sketch missing');farray(control['delta_sketch'],(DIM,64));safe.head_digest(control)
    compare(safe.head_digest(writer),before,'Frozen schema validation must not mutate any field')
    return dict(required_raw_fields=sorted(required),retained_geometry_fields=sorted(actual-required),actual_fields=sorted(actual),
        all_fields_preserved=True,writer_digest=before,control_digest=safe.head_digest(control),
        geometry_extensions_used_in_raw_prediction=False,core_prediction_formula_changed=False,
        arrays={name:dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype)) for name,value in writer.items()})

def audit137(z,data,out,c):
    count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True)
    if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
    for k in ('candidate_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified','best_checkpoint_selected','recovery_checkpoint_selected'):compare(s[k],False,'C137/'+k)
    for k in ('repeatable','raw_control_replay_consistent','step24_frozen_before_final_evaluation','functional_zero_parity_complete_before_gradients','functional_warm_parity_complete_before_gradients','all12_used_for_training','fixed_final_training_qualification_passed'):compare(s[k],True,'C137/'+k)
    compare(s['decision'],'C137_FIXED_FINAL_ALL12_TRAINING_GATES_PASSED');compare(s['completed_training_steps'],24);compare(s['new_backward_calls'],288);compare(s['final_endpoint_records'],48)
    source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source137']),'C137 source inventory changed')
    for n,h in c['source137'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'C137 trusted source/'+n)
    compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design(),'C137 registered design')
    for k in parent.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C137 linked '+k)
    expected=[dict(**r,C137_membership='training',independent_validation_eligible=False,prepared_sha256=safe.tensor_hash(data['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][r['state_id']]['raw'])) for r in parent.registry()]
    compare(z.json('ALL12_TRAINING_REGISTRY_FROZEN.json'),expected);compare(z.json('MODEL_PINS_REFERENCE.json'),data['modelpins']);exact(z.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],data['V'],'C137 basis')
    rank=len(data['V']);a=data['initial_a'].copy();m=np.zeros(rank);v=np.zeros(rank);initial=z.arrays('INITIAL_C136_STEP16_CONTROL.npz')
    for k,x in (('a',a),('m',m),('v',v)):exact(initial[k],x,'C137 reset initial/'+k)
    controls=z.json('PREFLIGHT_PROGRESS.json');need(len(controls)==48,'C137 known controls denominator');raw={};warm={}
    oldrecords={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
    for r,(sp,kind) in zip(controls,[(sp,kind) for sp in parent.registry() for kind in ('prepared',*parent.ARMS[:2],'warm')]):
        sid=sp['state_id'];p=data['points'][sid]
        for k,x in sp.items():compare(r[k],x,'C137 preflight/'+k)
        compare(r['C137_membership'],'training')
        if kind=='prepared':point=p['prepared'];compare(r['kind'],'prepared');need(safe.metric_gates(r['metrics'],sp['case'],point,False),'C137 preparation');refs=[p['preparation']]
        elif kind=='warm':
            compare(r['kind'],'warm');compare(r['arm'],'C136_fixed_step16');point=parent.proposal(p['prepared'],p['raw'],data['V'],a)[0];parent.gate_record(r['metrics'],sp['case'],point,p['prepared']);refs=[data['C136_controls'][(sid,'shared_vector_step16',k)]['metrics'] for k in (0,1)];warm[sid]=r['metrics']
        else:
            compare(r['kind'],'endpoint');compare(r['arm'],kind);point=data['maps'][(sid,kind)]['point'];parent.gate_record(r['metrics'],sp['case'],point,p['prepared']);refs=[oldrecords[(sid,kind,k)] for k in (0,1)]+[data['C135_controls'][(sid,kind,k)] for k in (0,1)]
            if kind==parent.ARMS[0]:raw[sid]=r['metrics']
        checks=[safe.repeat_comparison(r['metrics'],ref) for ref in refs];compare(r['comparisons'],checks);need(all(q['within_replay_tolerance'] and q['same_point_identity'] for q in checks),'C137 control drift')
    for name,refs in (('ZERO_FUNCTIONAL_PROGRESS.json',raw),('WARM_FUNCTIONAL_PROGRESS.json',warm)):
        rows=z.json(name);need(len(rows)==12,'C137 functional parity denominator')
        for sp,r in zip(parent.registry(),rows):
            sid=sp['state_id'];compare(r['state_id'],sid);err=close(r['observed'],refs[sid]['observed'],'C137 functional observation parity',2e-6);klerr=abs(r['KL']-refs[sid]['KL']);need(klerr<=2e-7,'C137 functional KL parity');compare(r['observed_parity_Linf'],err);compare(r['KL_parity_absolute_error'],klerr);need(0<=r['point_parity_error_Linf']<=2e-6,'C137 functional point parity')
    progress=z.json('TRAINING_PROGRESS.json');need(len(progress)==24,'C137 all24 updates required')
    for step,rec in enumerate(progress,1):
        name=f'training/step{step:02d}/UPDATE.npz';f=z.arrays(name);rows=z.json(f'training/step{step:02d}/OBSERVATIONS.json');compare(z.json(f'training/step{step:02d}/GRADIENT_PROGRESS.json'),rows)
        need(set(f)=={'before_a','before_m','before_v','full_weighted_gradient','after_a','after_m','after_v'} and len(rows)==12,'C137 update coverage')
        for k,x in (('before_a',a),('before_m',m),('before_v',v)):exact(f[k],x,'C137 chain/'+k)
        for sp,r in zip(parent.registry(),rows):
            sid=sp['state_id'];p=data['points'][sid];compare(r['state_id'],sid);compare(r['case'],sp['case']);compare(r['C137_membership'],'training');compare(r['coefficient_digest'],row_digest(a));point=parent.proposal(p['prepared'],p['raw'],data['V'],a)[0]
            parts=parent.objective(r['observed'],r['KL'],point,p['prepared'],a,sp['case']);compare(r['objective_parts'],parts,atol=1e-10,rtol=1e-9);compare(r['objective'],sum(parts.values()),atol=1e-10,rtol=1e-9);farray(np.asarray(r['state_gradient']),(rank,));need(0<=r['point_parity_error_Linf']<=2e-6,'C137 proposal parity')
        g,agg=parent.aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows]);compare(z.json(f'training/step{step:02d}/AGGREGATION.json'),agg)
        for row,w in zip(rows,agg['weights']):compare(row['aggregate_weight'],w);close(row['gradient_contribution'],w*np.array(row['state_gradient']),'C137 weighted row',1e-12)
        close(f['full_weighted_gradient'],g,'C137 weighted gradient',1e-12);aa,mm,vv,info=parent.adam_step(a,f['full_weighted_gradient'],m,v,step)
        for k,x in (('after_a',aa),('after_m',mm),('after_v',vv)):close(f[k],x,'C137 update arithmetic/'+k,2e-15)
        compare(rec,dict(**info,aggregate_objective_before_update=agg['objective'],state_weights=agg['weights'],before_digest=row_digest(a),after_digest=row_digest(f['after_a']),mean_objective_before_update=float(np.mean([r['objective'] for r in rows])),maximum_objective_before_update=float(max(r['objective'] for r in rows)),update_file=name,update_sha256=hashlib.sha256(z.raw(name)).hexdigest(),n_training_records=12));a,m,v=f['after_a'],f['after_m'],f['after_v']
    final=z.arrays('FINAL_SHARED_CONTROL_STEP24.npz');need(set(final)=={'a'},'C137 final vector fields');exact(final['a'],a,'C137 fixed24 final');saved=z.arrays('ALL24_FROZEN_FINAL_POINTS.npz');outcomes=[];points={}
    for sp in parent.registry():
        sid=sp['state_id'];p=data['points'][sid];trained,diag=parent.proposal(p['prepared'],p['raw'],data['V'],a)
        for arm,point in ((parent.EVAL_ARMS[0],p['raw']),(parent.EVAL_ARMS[1],trained)):
            key=sid+'__'+arm;exact(saved[key],point,'C137 final coordinate');points[(sid,arm)]=point;outcomes.append(dict(**sp,C137_membership='training',independent_validation_eligible=False,arm=arm,point_sha256=safe.tensor_hash(point),proposal_diagnostics=diag if arm==parent.EVAL_ARMS[1] else {'original_raw_bytes_retained':True}))
    need(set(saved)=={sid+'__'+arm for sid,arm in points},'C137 point coverage');compare(z.json('ALL24_FROZEN_FINAL_OUTCOMES.json'),outcomes)
    freeze=z.json('STEP24_FREEZE_BEFORE_EVALUATION.json');compare(freeze['completed_steps'],24);compare(freeze['control_digest'],row_digest(a));compare(freeze['checkpoint_choice'],'Fixed step24 only');compare(freeze['no_best_or_recovery_selection'],True)
    for k,n in (('control_sha256','FINAL_SHARED_CONTROL_STEP24.npz'),('basis_sha256','FROZEN_SHARED_ROW_BASIS.npz'),('training_registry_sha256','ALL12_TRAINING_REGISTRY_FROZEN.json')):compare(freeze[k],hashlib.sha256(z.raw(n)).hexdigest())
    expected_seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL12_TRAINING_REGISTRY_FROZEN.json','MODEL_PINS_REFERENCE.json','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C136_STEP16_CONTROL.npz','FINAL_SHARED_CONTROL_STEP24.npz','ALL24_FROZEN_FINAL_POINTS.npz','ALL24_FROZEN_FINAL_OUTCOMES.json'}
    need(set(freeze['proposal_seals'])==expected_seals,'C137 freeze coverage')
    for n,h in freeze['proposal_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C137 freeze/'+n)
    records=z.json('ALL48_FINAL_TRAINING_ENDPOINTS.json');need(len(records)==48,'C137 all48 endpoints');compare(z.json('FINAL_ENDPOINT_PROGRESS.json'),records);first={};controls137={}
    for i,(r,(rnd,sp,arm)) in enumerate(zip(records,[(rnd,sp,arm) for rnd in (0,1) for sp in parent.registry() for arm in parent.EVAL_ARMS])):
        sid=sp['state_id'];p=data['points'][sid];point=points[(sid,arm)]
        for k,x in dict(**sp,arm=arm,round=rnd,C137_membership='training',independent_validation_eligible=False).items():compare(r[k],x)
        compare(z.json(f'final_measurements/record{i:02d}.json'),r);compare(r['recomputed_gates'],parent.gate_record(r['metrics'],sp['case'],point,p['prepared']))
        aa=np.zeros_like(a) if arm==parent.EVAL_ARMS[0] else a;parts=parent.objective(r['metrics']['observed'],r['metrics']['KL'],point,p['prepared'],aa,sp['case']);compare(r['training_objective_parts'],parts);compare(r['training_objective'],sum(parts.values()))
        if arm==parent.EVAL_ARMS[0]:
            q=safe.repeat_comparison(r['metrics'],raw[sid]);compare(r['raw_control_replay'],q);need(q['within_replay_tolerance'] and q['same_point_identity'],'C137 raw control replay')
        if rnd==0:first[(sid,arm)]=r['metrics']
        else:
            q=safe.repeat_comparison(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q);need(q['within_replay_tolerance'] and q['same_point_identity'],'C137 repeated endpoint')
        controls137[(sid,arm,rnd)]=r
    sums,pairs=summaries137(records);compare(z.json('FINAL_TRAINING_SUMMARIES.json'),sums);compare(s['final_training_summaries'],sums);compare(z.json('PAIRED_VS_RAW.json'),pairs)
    for rnd in (0,1):need(sums['all12'][str(rnd)][parent.EVAL_ARMS[1]]['passed']==12,'C137 training qualification required')
    paired={str(rnd):dict(rescued_ids=[sp['state_id'] for sp in parent.registry() if not data['C136_controls'][(sp['state_id'],'shared_vector_step16',rnd)]['recomputed_gates']['complete_pass']],destroyed_ids=[]) for rnd in (0,1)}
    compare(z.json('PAIRED_VS_C136_STEP16.json'),paired);compare(s['paired_vs_C136_step16'],paired)
    for n in ('RAW_CONTROL_FAILURES.json','REPEATABILITY_FAILURES.json'):compare(z.json(n),[])
    dump(out/'C137_BOUND_RECEIPT.json',dict(parent_decision=s['decision'],parent_decision_reclassified=False,members_hashed=count,source_members_checked=len(source),updates_checked=24,training_rows_checked=288,final_endpoints_checked=48,all24_points_reconstructed=True,archived_gradients_independently_recomputed=False,measurement_accuracy_or_chronology_independently_certified=False))
    data.update(a=a.copy(),C137_points={sp['state_id']:points[(sp['state_id'],parent.EVAL_ARMS[1])] for sp in parent.registry()},C137_controls=controls137,C137_summary=s)

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Six original scientific archives required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    prior=out/'C132_C136_BINDINGS';prior.mkdir();data=load137({k:paths[k] for k in parent.INPUT_KEYS},prior,dict(c,pins={k:c['pins'][k] for k in parent.INPUT_KEYS}))
    with contextlib.closing(Archive(paths['C137'])) as z:audit137(z,data,out,c)
    with contextlib.closing(Archive(paths['C132'])) as z:
        data['writer']=z.arrays('ORIGINAL_WRITER.npz');data['control']=z.arrays('ORIGINAL_CONTROL.npz')
    dump(out/'FROZEN_WRITER_SCHEMA_VALIDATION.json',validate_frozen_inputs(data['writer'],data['control']))
    known={safe.tensor_hash(p[k]) for p in data['points'].values() for k in ('prepared','raw')}
    known|={safe.tensor_hash(p['point']) for p in data['maps'].values() if p['point'] is not None}
    for kind in ('C136_controls','C137_controls'):known|={r['metrics']['theta_sha256'] for r in data[kind].values()}
    with contextlib.closing(Archive(paths['C135'])) as z:known|={r['point_sha256'] for r in z.json('ALL84_COEFFICIENT_OUTCOMES_FROZEN.json')}
    data['historical_hashes']=known
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C138=True) for k in INPUT_KEYS})
    dump(out/'BOUND_HISTORICAL_COORDINATE_HASHES.json',dict(hashes=sorted(known),scope='Available states and endpoints in six bound originals; not exhaustive programme history',exhaustive_historical_catalog_verified=False))
    return data
