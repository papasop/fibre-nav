"""C26: archival diagnosis, never a training, forward, or feasibility experiment."""
import collections
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

PROTOCOL = 'GPT2_C26_ARCHIVAL_STALL_DIAGNOSIS_V1'
PARENT_PROTOCOL = 'GPT2_C25_FOUR_ROUTE_COVERAGE_A100_V1'
RESULT = 'gpt2_c26_archival_stall_diagnosis_results'
HEAD_SHA = '4af097017d739e95cd143d69070028b3751be04f977c4f900394f6153574560f'
START_SHA = '390170a4a99d1f8728de3f01b5165838a0bc19f8c1f895c1e0d93a0d4eb254b0'
BUDGET = .002
GATES = ('bits_match', 'output_flip', 'unselected_preserved', 'response', 'KL', 'exact_read_margin', 'development_read_margin')
CASES = ('xor_00', 'xor_01', 'xor_10', 'xor_11')
STARTS = {'xor_00':[0,0,1,1], 'xor_01':[0,1,1,0], 'xor_10':[1,0,1,0], 'xor_11':[1,1,1,1]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(z, name):
    def bad_constant(x):
        raise ValueError('Nonfinite JSON constant: ' + x)
    return json.loads(z.read(name), parse_constant=bad_constant)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def verify_archive(z, heartbeat=print):
    names = z.namelist()
    require(len(names) == len(set(names)), 'Duplicate ZIP members')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names), 'Unsafe ZIP member path')
    require('MANIFEST.json' in names, 'Result MANIFEST.json missing: choose result ZIP, not source ZIP')
    manifest = read_json(z, 'MANIFEST.json')
    require(isinstance(manifest, dict), 'Manifest must be a mapping')
    require({n for n in names if not n.endswith('/')} == set(manifest) | {'MANIFEST.json'}, 'Manifest coverage mismatch')
    before = time.monotonic()
    for i, (name, expected) in enumerate(manifest.items(), 1):
        require(isinstance(expected, str) and len(expected) == 64, 'Invalid manifest digest')
        h = hashlib.sha256()
        with z.open(name) as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(b)  # Reading to EOF also checks each member CRC.
        require(h.hexdigest() == expected, 'Member SHA256 mismatch: ' + name)
        if i % 1000 == 0 or time.monotonic() - before > 30:
            heartbeat('HASHES_OK %d / %d' % (i, len(manifest)), flush=True)
            before = time.monotonic()
    return manifest


def complete(row):
    return bool(row['reference_valid'] and all(row['metrics']['gates'].values()))


def quality(rows, ids, original):
    require(set(rows) == set(ids), 'Training denominator changed')
    for sid in ids:
        m = rows[sid]['metrics']
        require(set(m['gates']) == set(GATES), 'Gate set changed: ' + sid)
        require(all(type(v) is bool for v in m['gates'].values()), 'Nonboolean gate')
        require(math.isfinite(m['drift']) and m['drift'] >= 0 and math.isfinite(m['kl']), 'Nonfinite/negative metric')
    return {
        'passed_ids': [sid for sid in ids if complete(rows[sid])],
        'maximum_drift': max(rows[s]['metrics']['drift'] for s in ids),
        'original60_maximum_drift': max(rows[s]['metrics']['drift'] for s in ids if s in original),
        'score': sum((rows[s]['metrics']['drift'] / BUDGET) ** 2 for s in ids) / len(ids),
        'violation_score': sum((max(rows[s]['metrics']['drift'] - BUDGET, 0.) / BUDGET) ** 2 for s in ids) / len(ids),
        'valid_references': sum(bool(rows[s]['reference_valid']) for s in ids),
    }


def guard(previous, candidate, ids, original):
    a, b = quality(previous, ids, original), quality(candidate, ids, original)
    reasons = []
    if a['valid_references'] != len(ids) or b['valid_references'] != len(ids):
        reasons.append('INVALID_TRAINING_REFERENCE')
    if not set(a['passed_ids']).issubset(b['passed_ids']):
        reasons.append('LOST_PREVIOUS_COMPLETE_PASS')
    if not set(original).issubset(b['passed_ids']):
        reasons.append('ORIGINAL60_NOT_ALL_ACCEPTED')
    for sid in ids:
        for g in GATES:
            if g != 'response' and previous[sid]['metrics']['gates'][g] and not candidate[sid]['metrics']['gates'][g]:
                reasons.append('LOST_NON_RESPONSE_GATE:' + sid + ':' + g)
    if b['maximum_drift'] > a['maximum_drift']:
        reasons.append('WORST76_RESPONSE_DRIFT_INCREASED')
    if len(b['passed_ids']) <= len(a['passed_ids']) and not b['violation_score'] < a['violation_score'] * (1 - 1e-6):
        reasons.append('NO_DECLARED_TRAINING_IMPROVEMENT')
    return {'accepted': not reasons, 'reasons': reasons, 'before': a, 'after': b}


def validate_metric(sid, m):
    start=STARTS[sid.split('__')[0]];target=start[:3]+[start[0]^start[1]]
    exact=m['evaluation']['exact_margins'];dev=m['evaluation']['heldout_margins']
    require(len(exact)==4 and len(dev)==12 and all(math.isfinite(x) for x in exact+dev+[m['drift'],m['kl']]), 'Invalid raw margin records')
    expected={'bits_match':m['bits']==target,'output_flip':m['bits'][3]!=start[3],
              'unselected_preserved':m['bits'][:3]==start[:3],'response':m['drift']<=BUDGET,'KL':m['kl']<=.005,
              'exact_read_margin':min(exact)>=.05,'development_read_margin':min(dev)>=.05}
    require(m['gates']==expected,'Raw metric/gate mismatch: '+sid)
    endpoint=all(expected[g] for g in ('bits_match','response','KL','exact_read_margin','development_read_margin'))
    require(m['pass']==endpoint and m['episode_pass']==all(expected.values()),'Endpoint/episode boolean mismatch')


def compact_state(sid, row):
    m = row['metrics']
    validate_metric(sid,m)
    residual = m.get('response_residual')
    require(isinstance(residual, list) and len(residual) == 6 and all(math.isfinite(x) for x in residual), 'Invalid response residual: ' + sid)
    require(abs(max(abs(x) for x in residual) - m['drift']) <= 1e-7, 'Residual/drift mismatch: ' + sid)
    require(m['gates']['response'] == (m['drift'] <= BUDGET), 'Response gate mismatch: ' + sid)
    require(m['gates']['KL'] == (m['kl'] <= .005), 'KL gate mismatch: ' + sid)
    avg64 = m.get('response_residual_float64_aggregation')
    common=sum(residual)/6.;energy=sum(x*x for x in residual)
    return {
        'id': sid, 'case': sid.split('__')[0], 'valid': row['reference_valid'], 'passed': complete(row),
        'R': m['drift'], 'KL': m['kl'], 'response_slack': BUDGET - m['drift'],
        'residual_common_component':common,'residual_centered_Linf':max(abs(x-common) for x in residual),
        'common_component_energy_fraction':6*common*common/energy if energy else 0.,
        'minimum_exact_read_margin':min(m['evaluation']['exact_margins']),
        'minimum_development_read_margin':min(m['evaluation']['heldout_margins']),
        'failed_gates': [k for k, v in m['gates'].items() if not v], 'response_residual': residual,
        'float64_aggregation_R': max(map(abs, avg64)) if avg64 is not None else None,
        'aggregation_max_difference': max(abs(a-b) for a,b in zip(avg64, residual)) if avg64 is not None else None,
        'float64_aggregation_changes_response_gate': ((max(map(abs, avg64)) <= BUDGET) != m['gates']['response']) if avg64 is not None else None,
        'per_anchor_diagnostics': m.get('per_anchor_diagnostics'),
        'note': 'Float64 aggregation of saved float32 logits is not a float64 model replay; per-anchor values are diagnostics, not new gates.',
    }


def cohort(rows, ids):
    details = [compact_state(s, rows[s]) for s in ids]
    return {'n': len(ids), 'valid': sum(x['valid'] for x in details), 'passed': sum(x['passed'] for x in details),
            'gate_counts': {g: sum(rows[s]['metrics']['gates'][g] for s in ids) for g in GATES},
            'maximum_R': max(x['R'] for x in details), 'failed_ids': [x['id'] for x in details if not x['passed']]}


def residual_comparison(before, after, ids):
    details = []
    for sid in ids:
        a, b = compact_state(sid, before[sid]), compact_state(sid, after[sid])
        details.append({'id': sid, 'case': b['case'], 'before': a, 'after': b,
                        'R_change': b['R']-a['R'], 'gained': b['passed'] and not a['passed'], 'lost': a['passed'] and not b['passed']})
    by_case = {}
    for c in CASES:
        d = [r for r in details if r['case'] == c]
        by_case[c] = {'n': len(d), 'before_passed': sum(r['before']['passed'] for r in d),
                      'after_passed': sum(r['after']['passed'] for r in d),
                      'gained_ids': [r['id'] for r in d if r['gained']], 'lost_ids': [r['id'] for r in d if r['lost']]}
    return {'before': cohort(before, ids), 'after': cohort(after, ids), 'by_case': by_case,
            'gained_ids': [r['id'] for r in details if r['gained']], 'lost_ids': [r['id'] for r in details if r['lost']], 'states': details}


def inspect_parent(z, pins):
    identity, summary, split = (read_json(z, n) for n in ('identity.json', 'SUMMARY.json', 'SPLIT.json'))
    require(identity['protocol'] == PARENT_PROTOCOL and summary['protocol'] == PARENT_PROTOCOL and summary['complete'] is True, 'Complete C25 result required')
    require(identity['source_hashes'] == pins, 'Different C25 source revision; refuse guessed implementation mapping')
    require(summary['writer_head_sha256'] == HEAD_SHA and summary['actual_start_head_sha256'] == START_SHA, 'Requires recorded final C25 head')
    require(summary['offline_calibration_rounds'] == 30 and summary['accepted_layer_updates'] == 29 and summary['offline_stop'] == 'ALL_REMAINING_ROUTE_GRIDS_STALLED', 'C25 outcome differs from supplied run')
    require(summary['decision'] == 'C25_RAW_EXPANDED_TRAINING_FUNCTION_FAILURE', 'Unexpected C25 decision')
    require(read_json(z,'source/MODEL_PINS.json')['starts']==STARTS,'Memory case/target identity changed')
    settings = identity['settings']
    require(settings['response_budget'] == .002 and settings['kl_budget'] == .005 and settings['margin_gate'] == .05, 'Threshold changed')
    ids = [t['id'] for t in split['training']]
    original = [t['id'] for t in split['original60']]
    new_ids = [t['id'] for t in split['prospective_new_states']]
    require(len(ids) == len(set(ids)) == 76 and len(original) == len(set(original)) == 60 and set(original).issubset(ids), '76/60 split invalid')
    require(len(new_ids) == len(set(new_ids)) == 16 and not set(new_ids).intersection(ids), 'New-state split invalid')
    for c in CASES:
        require(sum(t['case'] == c for t in split['training']) == 19, '19-state route required')
        require(sum(t['case'] == c for t in split['prospective_new_states']) == 4, 'Four new states per case required')
    for n,h in pins.items():
        require(hashlib.sha256(z.read('source/'+n)).hexdigest() == h, 'Pinned archived source mismatch: '+n)
    progress, frozen = (read_json(z,n) for n in ('heads/calibrated/PROGRESS.json', 'WRITER_FROZEN.json'))
    member = frozen['selected_checkpoint']
    require(member == 'heads/calibrated/'+progress['checkpoint'] and progress['done'] is True and progress['rounds'] == 30, 'Final checkpoint pointer differs')
    require(hashlib.sha256(z.read(member)).hexdigest() == progress['checkpoint_sha256'], 'Final checkpoint digest mismatch')
    require(frozen['head_sha256'] == HEAD_SHA, 'Frozen identity mismatch')
    initial, final = (read_json(z,n) for n in ('C24_EXPANDED76_BASELINE.json', 'FINAL_TRAINING.json'))
    require(len(quality(initial,ids,original)['passed_ids']) == 69 and len(quality(final,ids,original)['passed_ids']) == 74, 'Actual saved training records differ')
    require(set(original).issubset(quality(final,ids,original)['passed_ids']), 'Original60 lost')
    new_before, new_after = {}, {}
    for sid in new_ids:
        r = read_json(z,'evaluation/new/'+sid+'/COMPLETE.json')
        require(r['id'] == sid and r['final_head_sha256'] == HEAD_SHA and r['writer_frozen_before_generation'] and not r['replacement_used'] and r['generation_retries'] == 0, 'New-state identity/protocol mismatch')
        for arm, dest in (('c24', new_before), ('calibrated', new_after)):
            dest[sid] = {'reference_valid': r['reference_valid'], 'metrics': r['arms'][arm]}
    train_compare = residual_comparison(initial, final, ids)
    new_compare = residual_comparison(new_before, new_after, new_ids)
    for key,b,a in [('training',train_compare['before'],train_compare['after']), ('prospective_local_state_development',new_compare['before'],new_compare['after'])]:
        for arm,actual in [('c24',b),('calibrated',a)]:
            reported = summary[key][arm]
            require(actual['n'] == reported['n_requested'] and actual['valid'] == reported['valid_state_references'] and actual['passed'] == reported['passed'] and actual['gate_counts'] == reported['gate_counts'] and actual['maximum_R'] == reported['maximum_response_drift'], 'Summary/records recount mismatch: '+key+'/'+arm)
    return {'identity':identity,'summary':summary,'split':split,'ids':ids,'original':original,'new_ids':new_ids,'initial':initial,'final':final,
            'new_before':new_before,'new_after':new_after,'training_comparison':train_compare,'new_comparison':new_compare,'checkpoint_member':member,'progress':progress,'frozen':frozen}


def replay_equal(expected, actual, tolerance=2e-6):
    require(set(expected)==set(actual),'Replay denominator changed')
    errors={'maximum_R_difference':0.,'maximum_KL_difference':0.,'maximum_residual_difference':0.}
    for sid in expected:
        e,a=expected[sid],actual[sid]
        require(e['reference_valid']==a['reference_valid'] and e['metrics']['gates']==a['metrics']['gates'],'Replay gates/reference differ: '+sid)
        require(e['metrics']['bits']==a['metrics']['bits'] and e['metrics']['theta_sha256']==a['metrics']['theta_sha256'],'Replay parameter identity/content differ: '+sid)
        for k,out in [('drift','maximum_R_difference'),('kl','maximum_KL_difference')]:
            delta=abs(e['metrics'][k]-a['metrics'][k]);errors[out]=max(errors[out],delta)
            require(delta<=tolerance,'Replay metric differs: '+sid+'/'+k)
        re=max(abs(x-y) for x,y in zip(e['metrics']['response_residual'],a['metrics']['response_residual']))
        errors['maximum_residual_difference']=max(errors['maximum_residual_difference'],re)
        require(re<=tolerance,'Replay residual differs: '+sid)
    return {**errors,'tolerance':tolerance,'timing_fields_ignored':True}


def scan_history(z, parent, heartbeat=print):
    ids, original = parent['ids'], parent['original']
    current, head = parent['initial'], START_SHA
    round_reports, candidates, last_context = [], [], None
    all_hashes = set()
    for round_no in range(1,31):
        prefix = 'calibration/round_%06d/' % round_no
        result, start = read_json(z,prefix+'RESULT.json'), read_json(z,prefix+'START.json')
        q = quality(current,ids,original)
        require(start['round'] == round_no and start['head_sha256'] == head and start['quality'] == q, 'Round origin/quality chain mismatch')
        require(result['round'] == round_no and result['case'] == start['case'], 'Round identity mismatch')
        route_ids = [s for s in ids if s.split('__')[0] == start['case']]
        failed_route = [s for s in route_ids if not complete(current[s])]
        reasons, lost_counts, unique, eligible = collections.Counter(), collections.Counter(), set(), 0
        round_candidates=[]
        for index in range(90):
            cid = 'candidate_%03d'%index
            rec = read_json(z,prefix+'candidates/'+cid+'/COMPLETE.json')
            spec, route_rows = rec['spec'], rec['route_rows']
            require(spec['id'] == cid and spec['case'] == start['case'] and rec['start_head_sha256'] == head and set(route_rows) == set(route_ids), 'Candidate identity/19-state denominator mismatch')
            settings = parent['identity']['settings']
            rdi, ri, ai = index//30,(index%30)//10,index%10
            require(spec['radius_index']==rdi and spec['ridge_index']==ri and spec['alpha_index']==ai and spec['backtrack_index']==ai
                    and spec['increment_radius']==settings['radius_grid'][rdi] and spec['ridge']==settings['ridge_grid'][ri] and spec['alpha']==settings['alpha_grid'][ai], 'Candidate fixed grid differs')
            for sid,row in route_rows.items():
                compact_state(sid,row)
            merged = dict(current); merged.update(route_rows)
            g = guard(current,merged,ids,original)
            require(g == rec['route_guard'] and g['accepted'] == rec['route_guard_eligible'], 'Stored guard fails independent recomputation')
            unique.add(rec['head_sha256']);all_hashes.add(rec['head_sha256']);eligible+=int(g['accepted'])
            reasons.update(set(x.split(':')[0] for x in g['reasons']))
            lost = [s for s in q['passed_ids'] if not complete(merged[s])]
            lost_counts.update(lost)
            passed_failures=[s for s in failed_route if complete(merged[s])]
            pp=rec['prediction']; actual_R=max(route_rows[s]['metrics']['drift'] for s in route_ids)
            prediction_max=pp['maximum_drift'];prediction_error=pp['maximum_prediction_error']
            require(all(math.isfinite(x) and x>=0 for x in (prediction_max,prediction_error)), 'Invalid archived prediction')
            require(abs(actual_R-prediction_max)<=prediction_error+1e-7, 'Archived prediction/error triangle inequality violated')
            ideal=rec['direction_info']['predicted_full_step_residuals']
            require(len(ideal)==19 and all(len(e)==6 for e in ideal), '19x6 ideal prediction required')
            # This is PRE head-rounding interpolation. Exact archived physical prediction is kept separately.
            raw_ideal={s:[current[s]['metrics']['response_residual'][k]+spec['alpha']*(ideal[i][k]-current[s]['metrics']['response_residual'][k]) for k in range(6)] for i,s in enumerate(route_ids)}
            ideal_R=max(abs(x) for e in raw_ideal.values() for x in e)
            detail={'round':round_no,'case':start['case'],'id':cid,'head_sha256':rec['head_sha256'],'spec':spec,'alias_of':rec.get('byte_identical_alias_of'),
                    'eligible':g['accepted'],'reasons':g['reasons'],'passed_after':len(g['after']['passed_ids']),'maximum76_R':g['after']['maximum_drift'],
                    'route_actual_R':actual_R,'original60_R':g['after']['original60_maximum_drift'],'violation_score':g['after']['violation_score'],
                    'lost_pass_ids':lost,'lost_original60_ids':[s for s in lost if s in original],'rescued_previously_failing_ids':passed_failures,
                    'all_previous_route_failures_rescued':bool(failed_route) and len(passed_failures)==len(failed_route),
                    'archived_linear_prediction_R':prediction_max,'archived_maximum_prediction_error':prediction_error,
                    'linear_predicts_route_within_budget_but_actual_exceeds':prediction_max<=BUDGET and actual_R>BUDGET,
                    'ideal_pre_rounding_interpolated_R':ideal_R,'ideal_pre_rounding_prediction_excess':max(0.,ideal_R-BUDGET),
                    'trust_clip_factor':rec['direction_info']['trust_clip_factor'],'actual_layer_step_norm':pp['actual_layer_step_norm'],
                    'actual_maximum_memory_step_norm':pp['maximum_actual_memory_step_norm'],
                    'recomputed_direction_available':True,'all76_candidate_rows_physically_remeasured':False,
                    'scope':'Route19 physical measurements merged with unchanged other routes; full76 confirmation is a separate AUDIT.'}
            candidates.append(detail);round_candidates.append(detail)
        require(result['scan']['n_completed_candidate_specs']==90 and result['scan']['n_unique_candidate_heads']==len(unique) and result['scan']['n_route_guard_eligible_specs']==eligible, 'Round scan counters mismatch')
        if result['accepted']:
            chosen=result['chosen']['id'];a=read_json(z,prefix+'candidate_audits/'+chosen+'/AUDIT.json')
            require(a['start_head_sha256']==head and a['spec']==result['chosen'] and a['head_sha256']==result['head_sha256'], 'Committed audit identity mismatch')
            require(a['full76_verified'] and a['replay_error'] is None and a['guard']==guard(current,a['rows'],ids,original) and a['guard']['accepted'], 'Committed audit invalid')
            candidate=next(r for r in round_candidates if r['id']==chosen)
            require(candidate['eligible'] and candidate['head_sha256']==a['head_sha256'],'Chosen candidate not eligible')
            candidate_record=read_json(z,prefix+'candidates/'+chosen+'/COMPLETE.json')
            replay_equal({**current,**candidate_record['route_rows']},a['rows'])
            for sid,row in a['rows'].items():compact_state(sid,row)
            current=a['rows'];head=a['head_sha256']
        else:
            require(result['chosen'] is None and result['head_sha256']==head and eligible==0, 'Rejected round altered head or eligible candidate unexplained')
        require(result['quality']==quality(current,ids,original), 'Committed quality chain mismatch')
        round_reports.append({'round':round_no,'case':result['case'],'accepted':result['accepted'],'head_sha256':head,
                              'passed':len(result['quality']['passed_ids']),'maximum_R':result['quality']['maximum_drift'],
                              'violation_score':result['quality']['violation_score'],'n_specs':90,'n_unique_heads':len(unique),'eligible_specs':eligible,
                              'rejection_reason_counts_nonexclusive':dict(reasons),'lost_pass_frequency':dict(lost_counts)})
        if round_no==30:
            last_context={'prefix':prefix,'start':start,'result':result,'baseline':current,'route_ids':route_ids,'candidates':round_candidates}
        heartbeat('AUDITED ROUND %d /30 pass %d/76'%(round_no,len(result['quality']['passed_ids'])),flush=True)
    require(head==HEAD_SHA, 'History does not reconstruct final head')
    replay=replay_equal(current,parent['final'])
    return {'rounds':round_reports,'candidate_rows':candidates,'n_specs':len(candidates),'n_unique_candidate_head_hashes_across_rounds':len(all_hashes),'last':last_context,'final_replay_comparison':replay}


def last_round_diagnosis(history):
    last=history['last'];rows=last['candidates'];unique={}
    for r in rows:unique.setdefault(r['head_sha256'],r)
    def stats(items):
        items=list(items)
        reason_counts=collections.Counter(r.split(':')[0] for x in items for r in set(x['reasons']))
        blockers=collections.Counter(s for x in items for s in x['lost_pass_ids'])
        return {'n':len(items),'eligible':sum(r['eligible'] for r in items),'reasons_nonexclusive':dict(reason_counts),
                'lose_original60':sum(bool(r['lost_original60_ids']) for r in items),
                'rescue_all_remaining_route_failures':sum(r['all_previous_route_failures_rescued'] for r in items),
                'reduce_worst76':sum(r['maximum76_R']<last['start']['quality']['maximum_drift'] for r in items),
                'optimistic_linear_full_route':sum(r['linear_predicts_route_within_budget_but_actual_exceeds'] for r in items),
                'most_frequent_lost_passes':dict(blockers.most_common(12))}
    best=sorted(rows,key=lambda r:(r['maximum76_R'],r['violation_score'],r['id']))[:10]
    return {'round':30,'case':last['start']['case'],'specs':stats(rows),'unique_heads':stats(unique.values()),
            'smallest_worst_response_candidates_diagnostic_only':best,
            'guard_causes_are_nonexclusive':True,'no_candidate_selected_or_committed':True,
            'cannot_conclude_infeasibility_or_global_capacity_limit':True}


def is_writer_input_coordinate(name):
    """Parse expert and layer separately; expert 0 is not layer 0."""
    parts=name.split('.')
    require(len(parts)==4 and parts[0]=='experts' and parts[1] in ('0','1','2','3')
            and parts[2] in ('0','2') and parts[3] in ('weight','bias'), 'Unexpected writer tensor coordinate: '+name)
    return parts[2]=='0'


def tensor_identity(z,parent,torch):
    import numpy as np
    def load(name):return torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=True)
    cp=load(parent['checkpoint_member']);state=cp['head']
    require(list(state)==[f'experts.{i}.{l}.{t}' for i in range(4) for l in (0,2) for t in ('weight','bias')], 'Head tensor order differs')
    for i in range(4):
        for k,shape in [('0.weight',(64,3147)),('0.bias',(64,)),('2.weight',(36864,64)),('2.bias',(36864,))]:
            t=state[f'experts.{i}.'+k]
            require(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and tuple(t.shape)==shape and torch.isfinite(t).all().item(), 'Head shape/dtype/finite check failed')
    digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in state.values())).hexdigest()
    require(digest==HEAD_SHA and cp['head_sha256']==HEAD_SHA and cp['done'] and cp['rounds']==30 and cp['accepted_updates']==29 and cp['optimiser'] is None, 'Actual final tensors/checkpoint differ')
    require(cp['current_training']==parent['final'] and cp['train_ids']==parent['ids'], 'Tensor checkpoint records differ')
    with zipfile.ZipFile(io.BytesIO(z.read('parent_capsule.zip'))) as capsule:
        baseline=torch.load(io.BytesIO(capsule.read('start_checkpoint.pt')),map_location='cpu',weights_only=True)['head']
    require(hashlib.sha256(b''.join(v.numpy().tobytes() for v in baseline.values())).hexdigest()==START_SHA,'Actual C24 baseline tensor identity differs')
    changed={}
    for n,t in state.items():
        if is_writer_input_coordinate(n):require(torch.equal(t,baseline[n]),'Writer frozen input layer changed: '+n)
        changed[n]=float(torch.linalg.vector_norm((t.double()-baseline[n].double()).reshape(-1)))
    return cp,state,{'checkpoint':parent['checkpoint_member'],'checkpoint_sha256':parent['progress']['checkpoint_sha256'],
                    'head_sha256_recomputed':digest,'all_tensors_finite':True,'frozen_input_layers_byte_equal_to_C24':True,
                    'actual_parameter_change_norms':changed,'torch_load_weights_only':True}


def geometry_diagnosis(z,parent,history,state,torch):
    import numpy as np
    last=history['last'];prefix=last['prefix'];J=[];H=[];E=[];state_rows=[]
    for sid in last['route_ids']:
        r=torch.load(io.BytesIO(z.read(prefix+'states/'+sid+'/JACOBIAN.pt')),map_location='cpu',weights_only=True)
        require(r['id']==sid and r['head_sha256']==HEAD_SHA,'Last-round Jacobian not at final frozen head')
        require(tuple(r['J'].shape)==(6,36864) and tuple(r['H'].shape)==(65,) and tuple(r['residual'].shape)==(6,),'Geometry shape changed')
        require(all(torch.isfinite(r[k]).all().item() for k in ('J','H','residual','theta')),'Nonfinite saved geometry')
        require(hashlib.sha256(r['theta'].numpy().tobytes()).hexdigest()==parent['final'][sid]['metrics']['theta_sha256'],'Saved geometry theta digest differs from final record')
        require(np.max(np.abs(r['residual'].double().numpy()-np.array(parent['final'][sid]['metrics']['response_residual'])))<=2e-6,'Geometry not at recorded final response')
        J.append(r['J'].double().numpy());H.append(r['H'].double().numpy());E.append(r['residual'].double().numpy())
        state_rows.append({'id':sid,'J_Frobenius_norm':float(np.linalg.norm(J[-1])),'H_norm':float(np.linalg.norm(H[-1])),
                           'response_R':float(np.max(np.abs(E[-1]))),'functional_replay_error':r['functional_residual_error']})
    J,H,E=map(np.stack,(J,H,E));scale=.07025910168886185
    # Scale checked against the archived C9 normalisation, not silently inferred.
    chain=['parent_capsule.zip','c24_capsule.zip','c23_capsule.zip','training_capsule.zip','training_capsule.zip']
    payload=None
    for member in chain:
        if payload is None:payload=z.read(member)
        else:
            with zipfile.ZipFile(io.BytesIO(payload)) as nested:payload=nested.read(member)
    with zipfile.ZipFile(io.BytesIO(payload)) as nested:
        norm=torch.load(io.BytesIO(nested.read('c9_checkpoint.pt')),map_location='cpu',weights_only=True)['normalisation']
    require(float(norm['output_scale'])==scale,'Pinned output scale changed')
    j=J.reshape(114,36864);h=np.repeat(H,6,axis=0)
    den=np.maximum(scale*np.linalg.norm(j,axis=1)*np.linalg.norm(h,axis=1),1e-12)
    gram=scale**2*(j@j.T)*(h@h.T)/(den[:,None]*den[None,:]);gram=(gram+gram.T)/2
    vals=np.linalg.eigvalsh(gram);hvals=np.linalg.svd(H,compute_uv=False)
    numerical_rank={str(t):int(np.sum(vals>t*max(float(vals.max()),1.))) for t in (1e-6,1e-8,1e-10,1e-12)}
    correlations=(H@H.T)/np.maximum(np.linalg.norm(H,axis=1)[:,None]*np.linalg.norm(H,axis=1)[None,:],1e-300)
    collisions=[]
    failed=[i for i,s in enumerate(last['route_ids']) if not complete(parent['final'][s])]
    passing=[i for i in range(19) if i not in failed]
    for i in failed:
        nearest=sorted(passing,key=lambda k:-correlations[i,k])[:3]
        for k in nearest:collisions.append({'failing_id':last['route_ids'][i],'passing_id':last['route_ids'][k],
                                          'hidden_cosine':float(correlations[i,k]),'hidden_L2_distance':float(np.linalg.norm(H[i]-H[k])),
                                          'response_residual_difference_Linf':float(np.max(np.abs(E[i]-E[k])))})
    return {'n_states':19,'n_equations':114,'output_layer_variables':36864*65,'source_J_dtype':'float32','algebra_dtype':'float64',
            'scale_verified_from_archived_normalisation':scale,'normalised_Gram_eigenvalues':vals.tolist(),'rank_sensitivity_diagnostic_only':numerical_rank,
            'hidden_matrix_singular_values':hvals.tolist(),'failed_vs_passing_hidden_neighbours':collisions,'states':state_rows,
            'solver_records':last['result']['solver_infos'],
            'interpretation':'Numerical Gram spectra and hidden-state proximity describe conditioning and coupling. Neither certifies rank, nonlinear feasibility, architectural insufficiency, or a capacity limit.',
            'new_Jacobians':0,'new_SVDs_of_model_J':0,'SVD_of_saved_hidden_matrix_only':True}


def reconstruct_linear_grid(z, parent, history, geometry, torch):
    """Independent float64 reconstruction of ideal steps; no model calls."""
    import numpy as np
    last=history['last'];prefix=last['prefix'];J=[];H=[];E=[]
    for sid in last['route_ids']:
        r=torch.load(io.BytesIO(z.read(prefix+'states/'+sid+'/JACOBIAN.pt')),map_location='cpu',weights_only=True)
        J.append(r['J'].double().numpy());H.append(r['H'].double().numpy());E.append(r['residual'].double().numpy())
    J,H,E=map(np.stack,(J,H,E));n,p,d=J.shape;j=J.reshape(n*p,d);h=np.repeat(H,p,axis=0)
    scale=geometry['scale_verified_from_archived_normalisation'];settings=parent['identity']['settings'];out=[]
    for ri,ridge in enumerate(settings['ridge_grid']):
        cp=torch.load(io.BytesIO(z.read(prefix+'COEFFICIENTS_%02d.pt'%ri)),map_location='cpu',weights_only=True)
        require(cp['start_head_sha256']==HEAD_SHA and cp['ridge']==ridge and cp['case']==last['start']['case'] and cp['training_ids']==last['route_ids'],'Coefficient identity mismatch')
        c=cp['coefficients'].double().numpy();require(c.shape==(114,) and np.isfinite(c).all(),'Invalid coefficients')
        D=scale*(j.T@(c[:,None]*h));unclipped_layer=float(np.linalg.norm(D))
        increments=scale*(D@H.T).T;unclipped_memory=float(np.max(np.linalg.norm(increments,axis=1)))
        unscaled=E+np.einsum('npd,nd->np',J,increments)
        solver=history['last']['result']['solver_infos'][ri]
        require(abs(float(np.max(np.abs(unscaled)))-solver['unscaled_solver_linear_maximum_drift'])<1e-6,'Saved coefficient/solver linear prediction mismatch')
        for rec in (r for r in last['candidates'] if r['spec']['ridge_index']==ri):
            spec=rec['spec'];factor=min(1.,settings['layer_frobenius_radius']/max(unclipped_layer,1e-300),spec['increment_radius']/max(unclipped_memory,1e-300))
            # Same ideal pre-float32-head direction as C25, including alpha.
            pred=E+spec['alpha']*np.einsum('npd,nd->np',J,scale*((D*factor)@H.T).T)
            require(abs(rec['ideal_pre_rounding_interpolated_R']-float(np.max(np.abs(pred))))<1e-6,'Ideal prediction reconstruction differs')
            require(abs(rec['trust_clip_factor']-factor)<1e-7,'Trust scaling reconstruction differs')
            protected=[i for i,s in enumerate(last['route_ids']) if complete(parent['final'][s])]
            out.append({'id':rec['id'],'spec':spec,'unclipped_layer_norm':unclipped_layer,'unclipped_memory_norm_max':unclipped_memory,
                        'trust_clip_factor_recomputed':factor,'ideal_predicted_residuals':pred.tolist(),'ideal_predicted_R':float(np.max(np.abs(pred))),
                        'ideal_predicted_protected_state_violations':[last['route_ids'][i] for i in protected if np.max(np.abs(pred[i]))>BUDGET],
                        'physical_route_R':rec['route_actual_R'],'archived_physical_step_prediction_R':rec['archived_linear_prediction_R'],
                        'archived_prediction_error':rec['archived_maximum_prediction_error'],
                        'scope':'Ideal layer direction before float32 head and memory rounding. Does not replace archived actual-displacement predictions or fresh model forwards.'})
        print('RECONSTRUCTED SAVED RIDGE',ridge,flush=True)
    return {'n_reconstructed_specs':len(out),'specs':sorted(out,key=lambda x:x['id']),
            'new_model_calls':0,'new_Jacobians':0,'certified_curvature_bounds':False,'new_candidate_proposals_or_commits':0}
