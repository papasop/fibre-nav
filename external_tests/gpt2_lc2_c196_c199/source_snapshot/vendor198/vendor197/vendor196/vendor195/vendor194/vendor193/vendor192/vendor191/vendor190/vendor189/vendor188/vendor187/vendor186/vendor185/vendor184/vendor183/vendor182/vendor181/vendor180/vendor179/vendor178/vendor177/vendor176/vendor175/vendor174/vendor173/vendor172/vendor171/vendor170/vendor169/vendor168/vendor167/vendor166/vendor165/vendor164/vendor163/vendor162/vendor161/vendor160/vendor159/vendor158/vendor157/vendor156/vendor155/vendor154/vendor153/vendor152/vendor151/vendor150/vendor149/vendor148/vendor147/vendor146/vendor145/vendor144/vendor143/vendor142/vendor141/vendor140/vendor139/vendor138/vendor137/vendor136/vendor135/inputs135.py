"""Bind three original capsules and replay C134's frozen path arithmetic."""
import contextlib
from core135 import *
from inputs134 import load as load132133
from core134 import design as design134,freeze_paths,registry as registry134,ALPHAS,path_gates,analyse

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Exactly C132/C133/C134 original ZIPs required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C135=True) for k in INPUT_KEYS})
    base=out/'C132_C133_BINDINGS';base.mkdir();cc=dict(c,pins={k:c['pins'][k] for k in ('C132','C133')})
    data=load132133({k:paths[k] for k in ('C132','C133')},base,cc)
    with contextlib.closing(Archive(paths['C132'])) as z:data['V']=z.arrays('FROZEN_SHARED_OPERATORS.npz')['V']
    with contextlib.closing(Archive(paths['C134'])) as z:
        members=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],'GPT2_LC2_C134_FIXED_WRITE_PATH_CPU_V1');compare(s['complete'],True);compare(s['decision'],'C134_FIXED_PATH_MEASUREMENT_COMPLETE_NO_CANDIDATE')
        if c['production']:compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True)
        for k in ('candidate_selected','step_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified'):compare(s[k],False,k)
        for k in ('repeatable','known_point_replay_consistent','path_interpretation_qualified','all_path_points_frozen_before_model','preflight_completed_before_paths'):compare(s[k],True,k)
        source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source134']),'C134 source inventory changed')
        for n,h in c['source134'].items():need(hashlib.sha256(z.raw('source/'+n)).hexdigest()==h,'C134 delivered source changed: '+n)
        compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),design134(),'C134 frozen design')
        for k in ('C132','C133'):compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C134 linked '+k)
        seals=z.json('POINT_SEALS_BEFORE_MODEL.json');need(set(seals)=={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','FROZEN_MODEL_PINS_REFERENCE.json','ALL_AVAILABLE_PATH_POINTS.npz','ALL126_PATH_OUTCOMES_FROZEN.json'},'C134 path seal coverage changed')
        for n,h in seals.items():need(hashlib.sha256(z.raw(n)).hexdigest()==h,'C134 path seal changed')
        compare(z.json('FROZEN_MODEL_PINS_REFERENCE.json'),data['modelpins'],'C134 pinned chart reference')
        archived_points=z.arrays('ALL_AVAILABLE_PATH_POINTS.npz');computed,rows,maps=freeze_paths(data);need(set(archived_points)==set(computed),'C134 point coverage differs')
        for k,v in computed.items():exact(archived_points[k],v,'C134 frozen path point differs')
        compare(z.json('ALL126_PATH_OUTCOMES_FROZEN.json'),rows,'C134 frozen point outcomes')
        controls=z.json('PREFLIGHT_PROGRESS.json');expected=[('prepared',r,None) if a is None else ('endpoint',r,a) for r in registry134() for a in (None,*ARMS) if a is None or data['maps'][(r['state_id'],a)]['point'] is not None]
        need(len(controls)==len(expected),'C134 preflight coverage differs');fresh={};archive_end={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
        for (kind,spec,arm),r in zip(expected,controls):
            sid=spec['state_id']
            for k,v in spec.items():compare(r[k],v,'C134 preflight identity/'+k)
            compare(r['kind'],kind);point=data['points'][sid]['prepared'] if kind=='prepared' else data['maps'][(sid,arm)]['point'];safe.metric_gates(r['metrics'],spec['case'],point,kind=='endpoint')
            if kind=='prepared':
                fresh[sid]=r['metrics'];comp=safe.repeat_comparison(r['metrics'],data['points'][sid]['preparation']);compare(r['comparison'],comp,'C134 preparation replay');need(comp['within_replay_tolerance'] and comp['same_point_identity'],'C134 prep replay inconsistent')
            else:
                compare(r['arm'],arm);checks=[safe.repeat_comparison(r['metrics'],archive_end[(sid,arm,rnd)]) for rnd in (0,1)];compare(r['comparisons'],checks,'C134 full endpoint replay');need(all(v['within_replay_tolerance'] and v['same_point_identity'] for v in checks),'C134 endpoint replay inconsistent')
        records=z.json('ALL252_PATH_MEASUREMENTS.json');need(len(records)==252,'C134 all252 records required');compare(z.json('PATH_PROGRESS.json'),records,'C134 path progress');stored_components=z.arrays('PATH_OBSERVABLE_COMPONENTS.npz');component_keys=set();first={};evaluated=0;endpoint134={}
        for i,((rnd,row),r) in enumerate(zip([(rnd,row) for rnd in (0,1) for row in rows],records)):
            for k,v in dict(**row,round=rnd).items():compare(r[k],v,'C134 measurement identity/'+k)
            compare(z.json(f'measurements/record{i:03d}.json'),r,'C134 individual measurement')
            sid=row['state_id'];arm=row['arm'];step=row['step'];point=maps[(sid,arm,step)];p=data['points'][sid]
            if point is None:
                compare(r['status'],'UNAVAILABLE_COMMON_READ_OPERATOR');compare(r['metrics'],None);continue
            evaluated+=1;compare(r['status'],'EVALUATED');g=path_gates(r['metrics'],row['case'],point,p['prepared'],row['alpha']);compare(r['recomputed_gates'],g,'C134 recomputed gates')
            values,stats=analyse(data['jacobians'][sid],p['prepared'],data['maps'][(sid,arm)]['point'],point,row['alpha'],fresh[sid]['observed'][:10],r['metrics']['observed'][:10]);compare(r['statistics'],stats,'C134 path statistics');compare(r['array_prefix'],f'record{i:03d}')
            for k,v in values.items():
                name=r['array_prefix']+'__'+k;component_keys.add(name);close(stored_components[name],v,'C134 component '+name,1e-9)
            if row['alpha'] in (0.,1.):
                reference=p['preparation'] if row['alpha']==0. else archive_end[(sid,arm,0)];comp=safe.repeat_comparison(r['metrics'],reference);compare(r['known_point_replay'],comp,'C134 known-point control');need(comp['within_replay_tolerance'] and comp['same_point_identity'],'C134 measured control drift')
            key=(sid,arm,step)
            if rnd==0:first[key]=r['metrics']
            else:
                comp=safe.repeat_comparison(r['metrics'],first[key]);compare(r['repeat_comparison'],comp,'C134 repeat');need(comp['within_replay_tolerance'] and comp['same_point_identity'],'C134 path repeat drift')
            if row['alpha']==1. and arm in ARMS[:2]:endpoint134[(sid,arm,rnd)]=r['metrics']
        need(set(stored_components)==component_keys,'C134 component inventory differs');compare(z.json('REPEATABILITY_FAILURES.json'),[]);compare(z.json('KNOWN_POINT_REPLAY_FAILURES.json'),[])
        summaries={}
        for rnd in (0,1):
            summaries[str(rnd)]={}
            for arm in ARMS:
                summaries[str(rnd)][arm]=[]
                for step,alpha in enumerate(ALPHAS):
                    ev=[r for r in records if r['round']==rnd and r['arm']==arm and r['step']==step and r['status']=='EVALUATED']
                    summaries[str(rnd)][arm].append(dict(step=step,alpha=alpha,evaluation_target='prepared_state' if alpha==0. else 'requested_write',n_requested=6,n_evaluated=len(ev),n_unavailable=6-len(ev),response_passed=sum(r['statistics']['finite_response_within_budget'] for r in ev),all_declared_gates_passed=sum(r['recomputed_gates']['complete_pass'] for r in ev),maximum_response_drift=max((r['statistics']['response_drift_Linf'] for r in ev),default=None),maximum_response_remainder=max((r['statistics']['response_remainder_Linf'] for r in ev),default=None)))
        compare(z.json('PATH_SUMMARIES.json'),summaries,'C134 path summaries');compare(s['path_summaries'],summaries,'C134 embedded summaries');compare(s['evaluated_path_measurements'],evaluated);compare(s['unavailable_path_measurements'],252-evaluated);compare(s['runtime_calls']['path_metrics'],evaluated)
        dump(out/'C134_BOUND_PATH_RECEIPT.json',dict(members_hashed=members,source_members_checked=len(source),path_coordinate_outcomes_checked=126,path_measurement_records_checked=252,component_arrays_checked=len(component_keys),frozen_formula_replay=True,new_model_forwards=0,archived_source_executed=False))
    data['C134_endpoints']=endpoint134;data['C134_preparations']=fresh;print('C135_INPUT_BINDINGS_OK C132/C133/C134; all prior points and path arithmetic checked; no C135 model yet.',flush=True);return data
