"""Bind C139 exposed kernel states without executing archive code or refitting."""
import contextlib
from core140 import *
from inputs139 import load as load139

def audit139(z,data,out,c):
    count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True);compare(s['decision'],'C139_EXPOSED_DIRECTION_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED')
    if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
    for k in ('repeatable','known_controls_replay_consistent','kernel_preparations_all24_passed'):compare(s[k],True,'C139/'+k)
    for k in ('candidate_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified'):compare(s[k],False,'C139/'+k)
    source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source139']),'C139 source inventory')
    for n,h in c['source139'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'C139 source/'+n)
    compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design())
    for k in previous.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C139 linked '+k)
    compare(z.json('C138_PREPARATION_AND_ENDPOINT_AUDIT.json'),data['C138_audit'],'C139 retained C138 audit')
    for n,obj in (('FROZEN_RAW_WRITER.npz',data['writer']),('FROZEN_CONTROL.npz',data['control']),('FROZEN_C137_CONTROL.npz',dict(a=data['a'],V=data['V']))):
        arr=z.arrays(n);need(set(arr)==set(obj),'C139 head inventory')
        for k,v in obj.items():exact(arr[k],v,'C139 frozen '+k)
    compare(z.json('CANDIDATE_FROZEN_BEFORE_COORDINATES.json'),dict(heads=head_digests(data),checkpoint='C137 fixed step24',new_parameter_selection=False))
    kernel,rows=previous.freeze_kernel_coordinates(data);compare(z.json('ALL24_KERNEL_REGISTRY.json'),rows);saved=z.arrays('ALL24_KERNEL_COORDINATES_BEFORE_MODEL.npz');need(set(saved)==set(kernel),'C139 coordinate coverage')
    for sid,p in kernel.items():exact(saved[sid],p,'C139 kernel coordinate')
    known=z.json('KNOWN_CONTROL_PROGRESS.json');need(len(known)==12,'C139 known controls')
    for sp,r in zip(parent.registry(),known):
        sid=sp['state_id'];p=data['points'][sid]
        for k,v in sp.items():compare(r[k],v)
        for k in ('feature_error_Linf','full_feature_error_Linf','raw_map_error_Linf'):need(0<=r[k]<=2e-6,'C139 feature parity')
        need(len(r['measurements'])==3,'C139 known native controls')
        for rec,kind in zip(r['measurements'],('prepared','raw','candidate')):
            compare(rec['kind'],kind);point=p['prepared'] if kind=='prepared' else (p['raw'] if kind=='raw' else data['C137_points'][sid]);met=rec['metrics']
            if kind=='prepared':need(safe.metric_gates(met,sp['case'],point,False),'C139 parent preparation');refs=[p['preparation']]
            else:
                parent.gate_record(met,sp['case'],point,p['prepared']);arm=parent.EVAL_ARMS[0] if kind=='raw' else parent.EVAL_ARMS[1];refs=[data['C137_controls'][(sid,arm,j)]['metrics'] for j in (0,1)]
            checks=[safe.repeat_comparison(met,ref) for ref in refs];compare(rec['comparisons'],checks);need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in checks),'C139 parent replay drift')
    full=z.json('FULL_REPLAY_PROGRESS.json');need(len(full)==24,'C139 full replay denominator')
    for sp,r in zip(previous.registry(),full):
        for k,v in sp.items():compare(r[k],v)
        theta=data['C138_fresh'][sp['source_state_id']];compare(r['gates'],previous.preparation_gates(r['metrics'],sp['case'],theta));q=safe.repeat_comparison(r['metrics'],data['C138_prewrite'][sp['source_state_id']]['preparation']);compare(r['comparison'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C139 original C138 replay drift')
    preps=z.json('ALL48_KERNEL_PREPARATIONS.json');compare(z.json('KERNEL_PREPARATION_PROGRESS.json'),preps);need(len(preps)==48,'C139 all48 preparation records');by={};prepsummary={}
    for r,(rnd,sp) in zip(preps,[(rnd,sp) for rnd in (0,1) for sp in previous.registry()]):
        sid=sp['state_id']
        for k,v in dict(**sp,round=rnd).items():compare(r[k],v)
        g=previous.preparation_gates(r['metrics'],sp['case'],kernel[sid]);compare(r['gates'],g);need(g['complete_pass'],'C139 invalid preparation cannot be dropped')
        if rnd==1:
            q=safe.repeat_comparison(r['metrics'],by[(0,sid)]['metrics']);compare(r['repeat_comparison'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C139 preparation repeat drift')
        by[(rnd,sid)]=r
    prepsummary={str(rnd):previous.gate_table([r for r in preps if r['round']==rnd]) for rnd in (0,1)};compare(z.json('KERNEL_PREPARATION_SUMMARIES.json'),prepsummary);compare(s['preparation_summaries'],prepsummary)
    features=z.json('PREWRITE_PROGRESS.json');need(len(features)==24,'C139 all24 feature bindings');points={};outcomes=[];newpoints={}
    for sp,r in zip(previous.registry(),features):
        sid=sp['state_id'];theta=kernel[sid]
        for k,v in sp.items():compare(r[k],v)
        arr=z.arrays('prewrite/'+sid+'.npz');need(set(arr)=={'prepared','raw','candidate','capture3147','write_features3211','observed'},'C139 prewrite schema');exact(arr['prepared'],theta,'C139 prewrite coordinate');x=farray(arr['capture3147'],(3147,));farray(arr['observed'],(22,))
        compare(float(x[3076]),float(int(sp['case'][-2])^int(sp['case'][-1])));close(arr['observed'],by[(0,sid)]['metrics']['observed'],'C139 capture/native parity',.0002)
        raw,ff=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],theta,x);warm,diag=proposal(theta,raw,data['V'],data['a'])
        for k,v in (('raw',raw),('candidate',warm),('write_features3211',ff)):exact(arr[k],v,'C139 saved map/'+k)
        for k,v in (('prepared_sha256',safe.tensor_hash(theta)),('raw_sha256',safe.tensor_hash(raw)),('candidate_sha256',safe.tensor_hash(warm)),('full_feature_digest',row_digest(ff)),('preparation_valid_both_rounds',True)):compare(r[k],v)
        newpoints[sid]=dict(spec=sp,prepared=theta,raw=raw,features=ff,preparation=by[(0,sid)]['metrics'])
        for arm,point in ((previous.ARMS[0],raw),(previous.ARMS[1],warm)):
            points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid_both_rounds=True,proposal_diagnostics=diag if arm==previous.ARMS[1] else dict(original_raw_rule=True)))
    pp=z.arrays('ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz');need(set(pp)=={sid+'__'+arm for sid,arm in points},'C139 all48 proposal points')
    for (sid,arm),p in points.items():exact(pp[sid+'__'+arm],p,'C139 sealed proposal')
    compare(z.json('ALL48_PROPOSAL_OUTCOMES.json'),outcomes)
    freeze=z.json('PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');compare(freeze['heads'],head_digests(data));compare(freeze['proposal_points'],48);compare(freeze['endpoint_measurements_so_far'],0)
    expected={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','C138_PREPARATION_AND_ENDPOINT_AUDIT.json','FROZEN_RAW_WRITER.npz','FROZEN_CONTROL.npz','FROZEN_C137_CONTROL.npz','CANDIDATE_FROZEN_BEFORE_COORDINATES.json','ALL24_KERNEL_COORDINATES_BEFORE_MODEL.npz','ALL24_KERNEL_REGISTRY.json','ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL48_PROPOSAL_OUTCOMES.json'}
    need(set(freeze['artifact_seals'])==expected,'C139 freeze coverage')
    for n,h in freeze['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C139 seal/'+n)
    records=z.json('ALL96_KERNEL_ENDPOINTS.json');compare(z.json('ENDPOINT_PROGRESS.json'),records);need(len(records)==96,'C139 endpoint denominator');first={};controls={}
    for i,(r,(rnd,sp,arm)) in enumerate(zip(records,[(rnd,sp,arm) for rnd in (0,1) for sp in previous.registry() for arm in previous.ARMS])):
        sid=sp['state_id']
        for k,v in dict(**sp,round=rnd,arm=arm,status='EVALUATED',preparation_valid_both_rounds=True).items():compare(r[k],v)
        compare(z.json(f'endpoints/record{i:02d}.json'),r);compare(r['recomputed_gates'],previous.previous.gate_record(r['metrics'],sp['case'],points[(sid,arm)],kernel[sid],True))
        if rnd==0:first[(sid,arm)]=r['metrics']
        else:
            q=safe.repeat_comparison(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C139 endpoint repeat drift')
        controls[(sid,arm,rnd)]=r
    sums,pairs=previous.endpoint_summaries(records);compare(z.json('ENDPOINT_SUMMARIES.json'),sums);compare(s['endpoint_summaries'],sums);compare(z.json('PAIRED_VS_RAW.json'),pairs);compare(z.json('REPEATABILITY_FAILURES.json'),[])
    paired={str(rnd):dict(full_preparation_passed=sum(r['gates']['complete_pass'] for r in full),kernel_preparation_passed=24,rescued_preparation_ids=[sp['state_id'] for sp in previous.registry() if not data['C138_prewrite'][sp['source_state_id']]['preparation_valid']],destroyed_preparation_ids=[]) for rnd in (0,1)}
    compare(z.json('PAIRED_PREPARATIONS.json'),paired);compare(s['paired_preparations'],paired);compare(s['preparation_valid_both_rounds'],24)
    ok=all(sums['all24'][str(j)][previous.ARMS[1]]['passed']==24 for j in (0,1));compare(s['frozen_writer_all24_development_gates_passed'],ok)
    compare(s['next_branch'],'PREREGISTER_SEPARATE_UNSEEN_LOCAL_CONFIRMATION' if ok else 'RESOLVE_SEEN_CLASS_WRITER_FAILURES_BEFORE_CONFIRMATION')
    for k,v in (('new_model_forwards',517),('kernel_preparation_records',48),('kernel_feature_records',24),('final_endpoint_records',96)):compare(s[k],v)
    dump(out/'C139_BOUND_RECEIPT.json',dict(parent_decision=s['decision'],parent_decision_reclassified=False,members_hashed=count,source_members_checked=len(source),kernel_coordinates_reconstructed=24,preparations_checked=48,endpoints_checked=96,independent_native_measurement_accuracy_certified=False))
    data.update(C139_points=newpoints,C139_controls=controls,C139_preparations=by,C139_summary=s)

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Eight original scientific archives required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    prior=out/'C132_C138_BINDINGS';prior.mkdir();data=load139({k:paths[k] for k in previous.INPUT_KEYS},prior,dict(c,pins={k:c['pins'][k] for k in previous.INPUT_KEYS}))
    with contextlib.closing(Archive(paths['C139'])) as z:audit139(z,data,out,c)
    data['points']=dict(data['points'],**data['C139_points']);data['initial_a']=data['a'].copy();refs={}
    for sp in registry():
        sid=sp['state_id'];p=data['points'][sid];refs[(sid,'prepared')]=[p['preparation']]
        if sp['training_origin']=='historical12':
            for kind,arm in (('raw',parent.EVAL_ARMS[0]),('warm',parent.EVAL_ARMS[1])):refs[(sid,kind)]=[data['C137_controls'][(sid,arm,j)]['metrics'] for j in (0,1)]
        else:
            refs[(sid,'prepared')]=[data['C139_preparations'][(j,sid)]['metrics'] for j in (0,1)]
            for kind,arm in (('raw',previous.ARMS[0]),('warm',previous.ARMS[1])):refs[(sid,kind)]=[data['C139_controls'][(sid,arm,j)]['metrics'] for j in (0,1)]
    data['native_refs']=refs;dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C140=True) for k in INPUT_KEYS})
    dump(out/'ALL36_TRAINING_REGISTRY_FROZEN.json',[dict(r,prepared_sha256=safe.tensor_hash(data['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][r['state_id']]['raw'])) for r in registry()]);dump(out/'MODEL_PINS_REFERENCE.json',data['modelpins'])
    np.savez_compressed(out/'ALL36_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in data['points'].items() for k in ('prepared','raw','features')});np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=data['V']);np.savez_compressed(out/'INITIAL_C137_STEP24_CONTROL.npz',a=data['initial_a'],m=np.zeros_like(data['initial_a']),v=np.zeros_like(data['initial_a']))
    np.savez_compressed(out/'FROZEN_ORIGINAL_WRITER.npz',**data['writer']);np.savez_compressed(out/'FROZEN_ORIGINAL_CONTROL.npz',**data['control']);dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(data));return data
