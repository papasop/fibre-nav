"""Read-only original bindings; all former probe states become explicit training."""
import contextlib
from core136 import *
from inputs135 import load as load_prior
from core135 import design as design135,freeze_coefficients,BETAS,local_diagnostic
from worker135 import summarise,historical_rule

def load(paths,out,c):
    need(set(paths)==set(INPUT_KEYS),'Four fixed original archives required')
    for k in INPUT_KEYS:need(sha(paths[k])==c['pins'][k]['sha256'],'Wrong fixed original '+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C136=True) for k in INPUT_KEYS})
    parent=out/'C132_C134_BINDINGS';parent.mkdir();data=load_prior({k:paths[k] for k in ('C132','C133','C134')},parent,dict(c,pins={k:c['pins'][k] for k in ('C132','C133','C134')}))
    with contextlib.closing(Archive(paths['C135'])) as z:
        members=z.verify_manifest();s=z.json('SUMMARY.json');compare(s['protocol'],'GPT2_LC2_C135_NORMAL_COEFFICIENT_CPU_V1');compare(s['complete'],True)
        if c['production']:
            compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True);compare(s['decision'],'C135_FIXED_NORMAL_COEFFICIENT_COMPLETE_NO_QUALIFIED_COEFFICIENT');need(len(data['V'])==36,'Pinned C132 numeric36 row basis required')
        for k in ('candidate_selected','coefficient_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified'):compare(s[k],False,k)
        for k in ('repeatable','known_control_replay_consistent','interpretation_qualified','all_coefficient_points_frozen_before_model','preflight_completed_before_coefficients'):compare(s[k],True,k)
        source={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(source==set(c['source135']),'C135 source inventory changed')
        for n,h in c['source135'].items():need(hashlib.sha256(z.raw('source/'+n)).hexdigest()==h,'C135 delivered source changed: '+n)
        compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),design135(),'C135 frozen design')
        for k in ('C132','C133','C134'):compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],c['pins'][k]['sha256'],'C135 linked '+k)
        seals=z.json('POINT_SEALS_BEFORE_MODEL.json');need(set(seals)=={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','ALL84_FROZEN_POINTS.npz','ALL84_COEFFICIENT_OUTCOMES_FROZEN.json'},'C135 seal coverage changed')
        for n,h in seals.items():need(hashlib.sha256(z.raw(n)).hexdigest()==h,'C135 premodel seal changed')
        arrays,rows,maps,parts=freeze_coefficients(data);stored=z.arrays('ALL84_FROZEN_POINTS.npz');need(set(stored)==set(arrays),'C135 point coverage differs')
        for k,v in arrays.items():exact(stored[k],v,'C135 committed point differs')
        compare(z.json('ALL84_COEFFICIENT_OUTCOMES_FROZEN.json'),rows,'C135 frozen outcomes')
        controls=z.json('PREFLIGHT_PROGRESS.json');expected=[('prepared',r,None) if a is None else ('endpoint',r,a) for r in registry() for a in (None,*ARMS[:2])];need(len(controls)==36,'C135 all36 controls required');fresh={};original={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in data['records'] if r['status']=='EVALUATED'}
        for (kind,spec,arm),r in zip(expected,controls):
            sid=spec['state_id']
            for k,v in spec.items():compare(r[k],v,'C135 control identity/'+k)
            compare(r['kind'],kind);point=data['points'][sid]['prepared'] if kind=='prepared' else data['maps'][(sid,arm)]['point'];safe.metric_gates(r['metrics'],spec['case'],point,kind=='endpoint')
            if kind=='prepared':
                fresh[sid]=r['metrics'];refs=[data['points'][sid]['preparation']]
                if sid in data['C134_preparations']:refs.append(data['C134_preparations'][sid])
            else:
                compare(r['arm'],arm);refs=[original[(sid,arm,k)] for k in (0,1)]
                if (sid,arm,0) in data['C134_endpoints']:refs += [data['C134_endpoints'][(sid,arm,k)] for k in (0,1)]
            checks=[safe.repeat_comparison(r['metrics'],ref) for ref in refs];compare(r['comparisons'],checks,'C135 preflight comparison');need(all(q['within_replay_tolerance'] and q['same_point_identity'] for q in checks),'C135 control drift')
        records=z.json('ALL168_COEFFICIENT_MEASUREMENTS.json');need(len(records)==168,'C135 all168 records required');compare(z.json('ENDPOINT_PROGRESS.json'),records,'C135 progress')
        saved=z.arrays('GEOMETRY_ONLY_LINEAR_COMPONENTS.npz');keys=set();first={};C135_controls={}
        for i,((rnd,row),r) in enumerate(zip([(rnd,row) for rnd in (0,1) for row in rows],records)):
            for k,v in dict(**row,round=rnd).items():compare(r[k],v,'C135 endpoint identity/'+k)
            compare(z.json(f'measurements/record{i:03d}.json'),r,'C135 individual record');compare(r['status'],'EVALUATED');sid=row['state_id'];ci=row['coefficient_index'];point=maps[(sid,ci)];p=data['points'][sid]
            g=gate_record(r['metrics'],row['case'],point,p['prepared']);compare(r['recomputed_gates'],g,'C135 gates')
            values,stats=local_diagnostic(data['jacobians'].get(sid),parts[(sid,ci)],row['beta'],fresh[sid]['observed'][:10],r['metrics']['observed'][:10]);compare(r['local_statistics'],stats,'C135 local statistics')
            if values is not None:
                compare(r['array_prefix'],f'record{i:03d}')
                for k,v in values.items():
                    name=r['array_prefix']+'__'+k;keys.add(name);close(saved[name],v,'C135 component '+name,1e-9)
            if row['beta'] in (0.,1.):
                arm=ARMS[1] if row['beta']==0. else ARMS[0];refs=[original[(sid,arm,k)] for k in (0,1)]
                if (sid,arm,0) in data['C134_endpoints']:refs += [data['C134_endpoints'][(sid,arm,k)] for k in (0,1)]
                checks=[safe.repeat_comparison(r['metrics'],ref) for ref in refs];compare(r['known_control_comparisons'],checks,'C135 known control');need(all(q['within_replay_tolerance'] and q['same_point_identity'] for q in checks),'C135 known endpoint drift');C135_controls[(sid,arm,rnd)]=r['metrics']
            key=(sid,ci)
            if rnd==0:first[key]=r['metrics']
            else:
                q=safe.repeat_comparison(r['metrics'],first[key]);compare(r['repeat_comparison'],q,'C135 repeat');need(q['within_replay_tolerance'] and q['same_point_identity'],'C135 repeat drift')
        need(set(saved)==keys,'C135 component coverage differs');compare(z.json('KNOWN_CONTROL_FAILURES.json'),[]);compare(z.json('REPEATABILITY_FAILURES.json'),[])
        summaries,pairs=summarise(records);compare(z.json('COEFFICIENT_SUMMARIES.json'),summaries,'C135 summary');compare(s['endpoint_summaries'],summaries,'C135 embedded summary');compare(z.json('PAIRED_VS_RAW.json'),pairs,'C135 pairs');compare(s['noncontrol_coefficients_meeting_historical_rule'],historical_rule(summaries,pairs,True),'C135 historical rule')
        dump(out/'C135_BOUND_RECEIPT.json',dict(members_hashed=members,source_members_checked=len(source),all168_records_checked=True,all84_proposal_coordinates_checked=True,coefficient_family_not_extended=True,archived_source_executed=False))
    data['C135_controls']=C135_controls
    training=[dict(**r,C136_membership='training',independent_validation_eligible=False,prepared_sha256=safe.tensor_hash(data['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(data['points'][r['state_id']]['raw'])) for r in registry()]
    dump(out/'ALL12_TRAINING_REGISTRY_FROZEN.json',training);dump(out/'MODEL_PINS_REFERENCE.json',data['modelpins']);np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=data['V'])
    print('C136_INPUT_BINDINGS_OK; all12 explicitly training; no C136 model or optimizer yet.',flush=True);return data
