"""Reconstruct bound C138 archive with delivered code, never archive execution."""
import contextlib
from core139 import *
from inputs138 import load as load138,validate_frozen_inputs

def audit138(z,data,out,c):
    count=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],previous.PROTOCOL);compare(s['complete'],True)
    if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
    compare(s['repeatable'],True);compare(s['known_controls_replay_consistent'],True)
    for k in ('candidate_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified'):compare(s[k],False,'C138/'+k)
    source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source138']),'C138 delivered source inventory')
    for n,h in c['source138'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'C138 source/'+n)
    compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),previous.design(),'C138 fixed design')
    for k in previous.INPUT_KEYS:compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C138 linked '+k)
    for n,obj in (('FROZEN_RAW_WRITER.npz',data['writer']),('FROZEN_CONTROL.npz',data['control']),('FROZEN_C137_CONTROL.npz',dict(a=data['a'],V=data['V']))):
        actual=z.arrays(n);need(set(actual)==set(obj),'C138 head inventory/'+n)
        for k,v in obj.items():exact(actual[k],v,'C138 frozen tensor/'+k)
    compare(z.json('FROZEN_WRITER_SCHEMA_VALIDATION.json'),validate_frozen_inputs(data['writer'],data['control']))
    compare(z.json('CANDIDATE_FROZEN_BEFORE_NEW_STATES.json'),dict(parent_protocol=parent.PROTOCOL,parent_decision=data['C137_summary']['decision'],parent_decision_reclassified=False,heads=head_digests(data),candidate_checkpoint='C137 fixed step24',new_parameter_selection=False))
    fresh,rows=previous.fresh_coordinates(data);compare(z.json('ALL24_NEW_STATE_REGISTRY.json'),rows);saved=z.arrays('ALL24_PREPARED_COORDINATES_BEFORE_MODEL.npz');need(set(saved)==set(fresh),'C138 all24 coordinates')
    for sid,p in fresh.items():exact(saved[sid],p,'C138 fresh coordinate')
    oldcontrols=z.json('KNOWN_CONTROL_PROGRESS.json');need(len(oldcontrols)==12,'C138 known12 controls')
    for sp,rec in zip(parent.registry(),oldcontrols):
        sid=sp['state_id'];p=data['points'][sid]
        for k,v in sp.items():compare(rec[k],v)
        for k in ('feature_error_Linf','full_feature_error_Linf','raw_map_error_Linf'):need(0<=rec[k]<=2e-6,'C138 recorded feature parity')
        need(len(rec['measurements'])==3,'C138 known native control coverage')
        for rr,kind in zip(rec['measurements'],('prepared','raw','candidate')):
            compare(rr['kind'],kind);point=p['prepared'] if kind=='prepared' else (p['raw'] if kind=='raw' else data['C137_points'][sid]);m=rr['metrics']
            if kind=='prepared':need(safe.metric_gates(m,sp['case'],point,False),'C138 known valid parent');refs=[p['preparation']]
            else:
                parent.gate_record(m,sp['case'],point,p['prepared']);arm=parent.EVAL_ARMS[0] if kind=='raw' else parent.EVAL_ARMS[1];refs=[data['C137_controls'][(sid,arm,rnd)]['metrics'] for rnd in (0,1)]
            checks=[safe.repeat_comparison(m,ref) for ref in refs];compare(rr['comparisons'],checks);need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in checks),'C138 known controls drift')
    pre=z.json('PREWRITE_PROGRESS.json');need(len(pre)==24,'C138 prewrite denominator');preby={};points={};outcomes=[];preaudit=[]
    for sp,r in zip(previous.registry(),pre):
        sid=sp['state_id'];theta=fresh[sid]
        for k,v in sp.items():compare(r[k],v)
        arr=z.arrays('prewrite/'+sid+'.npz');need(set(arr)=={'prepared','raw','candidate','capture3147','write_features3211','observed'},'C138 prewrite fields');exact(arr['prepared'],theta,'C138 prewrite prepared')
        x=farray(arr['capture3147'],(3147,));o=farray(arr['observed'],(22,));compare(float(x[3076]),float(int(sp['case'][-2])^int(sp['case'][-1])),'C138 desired feature')
        raw,full=safe.reconstruct_raw(data['writer'],data['control']['delta_sketch'],theta,x);cp,diag=parent.proposal(theta,raw,data['V'],data['a'])
        for k,v in (('raw',raw),('candidate',cp),('write_features3211',full)):exact(arr[k],v,'C138 saved map/'+k)
        close(o,r['preparation']['observed'],'C138 capture/native parity',.0002);g=preparation_gates(r['preparation'],sp['case'],theta);compare(r['preparation_valid'],g['complete_pass'])
        for k,v in (('prepared_sha256',safe.tensor_hash(theta)),('raw_sha256',safe.tensor_hash(raw)),('candidate_sha256',safe.tensor_hash(cp)),('full_feature_digest',row_digest(full))):compare(r[k],v,'C138 binding/'+k)
        preby[sid]=r;preaudit.append(dict(state_id=sid,case=sp['case'],metrics=r['preparation'],gates=g))
        for arm,point in ((ARMS[0],raw),(ARMS[1],cp)):
            points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid=g['complete_pass'],proposal_diagnostics=diag if arm==ARMS[1] else dict(original_raw_rule=True)))
    pp=z.arrays('ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz');need(set(pp)=={sid+'__'+arm for sid,arm in points},'C138 proposal coverage')
    for (sid,arm),point in points.items():exact(pp[sid+'__'+arm],point,'C138 frozen proposal')
    compare(z.json('ALL48_PROPOSAL_OUTCOMES_BEFORE_ENDPOINTS.json'),outcomes)
    freeze=z.json('ALL_PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json');compare(freeze['heads'],head_digests(data));compare(freeze['new_states'],24);compare(freeze['proposal_points'],48);compare(freeze['new_endpoint_measurements_so_far'],0)
    seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','BOUND_HISTORICAL_COORDINATE_HASHES.json','C137_BOUND_RECEIPT.json','FROZEN_WRITER_SCHEMA_VALIDATION.json','FROZEN_RAW_WRITER.npz','FROZEN_CONTROL.npz','FROZEN_C137_CONTROL.npz','CANDIDATE_FROZEN_BEFORE_NEW_STATES.json','ALL24_PREPARED_COORDINATES_BEFORE_MODEL.npz','ALL24_NEW_STATE_REGISTRY.json','ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL48_PROPOSAL_OUTCOMES_BEFORE_ENDPOINTS.json'}
    need(set(freeze['artifact_seals'])==seals,'C138 freeze coverage')
    for n,h in freeze['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C138 artifact seal/'+n)
    records=z.json('ALL96_CONFIRMATION_ENDPOINTS.json');compare(z.json('ENDPOINT_PROGRESS.json'),records);need(len(records)==96,'C138 all96 endpoints');first={}
    for i,(r,(rnd,sp,arm)) in enumerate(zip(records,[(rnd,sp,arm) for rnd in (0,1) for sp in previous.registry() for arm in ARMS])):
        sid=sp['state_id']
        for k,v in dict(**sp,round=rnd,arm=arm,status='EVALUATED',preparation_valid=preby[sid]['preparation_valid']).items():compare(r[k],v)
        compare(z.json(f'endpoints/record{i:02d}.json'),r);compare(r['recomputed_gates'],previous.gate_record(r['metrics'],sp['case'],points[(sid,arm)],fresh[sid],preby[sid]['preparation_valid']))
        if rnd==0:first[(sid,arm)]=r['metrics']
        else:
            q=safe.repeat_comparison(r['metrics'],first[(sid,arm)]);compare(r['repeat_comparison'],q);need(q['same_point_identity'] and q['within_replay_tolerance'],'C138 repeats')
    sums,pairs=previous.summaries(records);compare(z.json('CONFIRMATION_SUMMARIES.json'),sums);compare(s['confirmation_summaries'],sums);compare(z.json('PAIRED_VS_RAW.json'),pairs);compare(z.json('REPEATABILITY_FAILURES.json'),[])
    passed=sum(r['gates']['complete_pass'] for r in preaudit);compare(s['preparation_states_passed'],passed);compare(s['preparation_states_failed'],24-passed)
    success=passed==24 and all(sums['all24'][str(rnd)][ARMS[1]]['passed']==24 for rnd in (0,1));compare(s['local_confirmation_passed'],success);compare(s['decision'],'C138_FROZEN_LOCAL_CONFIRMATION_ALL24_PASSED' if success else 'C138_FROZEN_LOCAL_CONFIRMATION_FAILURE')
    for k,v in (('final_endpoint_records',96),('prewrite_records',24),('new_model_forwards',421)):compare(s[k],v)
    # Diagnostic strata keep the all24 denominator; no subset is relabelled as a pass.
    strata={}
    for flag in (True,False):
        ids={r['state_id'] for r in preaudit if r['gates']['complete_pass']==flag};sub=[r for r in records if r['state_id'] in ids];strata['valid_preparation' if flag else 'invalid_preparation']=dict(n_states=len(ids),state_ids=sorted(ids),endpoint_pass_counts={str(rnd):{arm:sum(r['recomputed_gates']['complete_pass'] for r in sub if r['round']==rnd and r['arm']==arm) for arm in ARMS} for rnd in (0,1)})
    report=dict(parent_decision=s['decision'],parent_decision_reclassified=False,all24_preparation_gates=gate_table(preaudit),preparation_records=preaudit,descriptive_strata=strata,all24_endpoint_summaries=sums,paired_vs_raw=pairs,subset_requalification_performed=False,members_hashed=count,source_members_checked=len(source),native_measurement_accuracy_independently_certified=False)
    dump(out/'C138_PREPARATION_AND_ENDPOINT_AUDIT.json',report)
    data.update(C138_fresh=fresh,C138_prewrite=preby,C138_summary=s,C138_audit=report)

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Seven original scientific archives required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    prior=out/'C132_C137_BINDINGS';prior.mkdir();data=load138({k:paths[k] for k in previous.INPUT_KEYS},prior,dict(c,pins={k:c['pins'][k] for k in previous.INPUT_KEYS}))
    with contextlib.closing(Archive(paths['C138'])) as z:audit138(z,data,out,c)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C139=True) for k in INPUT_KEYS})
    return data
