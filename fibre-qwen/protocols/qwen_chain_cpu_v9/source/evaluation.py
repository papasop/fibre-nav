"""Post-inference scoring only; expected answers never enter the execution module."""
from protocol import TOTAL,TOL,ID,CONFIG,programs
from execution import decoded_bit,validate_record
from v8_scoring import score,constrained_score,compare_all

def truth(op,a,b):
    return int({'OR':bool(a or b),'AND':bool(a and b),'XOR':a!=b,'EQUAL':a==b}[op])
def scoring_record(record):
    f=record['fields'];o=record['observation']
    return dict(run_id=record['item']['id'],query=dict(pair=str(f['a'])+str(f['b']),task=f['op']),**o)
def summarize(records,plan,baseline,environment_match):
    if len(records)>TOTAL or len(plan)!=TOTAL:raise ValueError('Record count')
    prompts={(q['task'],int(q['pair'][0]),int(q['pair'][1])):q['prompt'] for q in baseline['queries']}
    direct={};audits=[];wires=[];scores=[];replays=[];legacy_fail=0
    for i,(r,item) in enumerate(zip(records,plan)):
        wire=validate_record(r,item,records[:i],prompts);wires.append(wire)
        o=r['observation'];f=r['fields']
        if f is None or o.get('accepted') is not True:
            scores.append(None);audits.append(False);continue
        s=scoring_record(r);a=constrained_score(s);primary=score(s)
        # Explicit no-tie gate even if a bounded decoder emits one bit.
        valid=a['constraint_audit_pass'] and primary['primary_prediction'] is not None
        audits.append(valid);scores.append(dict(primary=primary,bounded=a))
        legacy_fail+=int(not a['legacy_cross_path_audit_pass'])
        key=(f['op'],f['a'],f['b'])
        if item['kind']=='direct':direct[key]=s
        elif key in direct:replays.append(compare_all([direct[key]],[s])['pass_gate'])
    cases=[]
    for j,q in enumerate(programs()):
        start=16+3*j
        if len(records)<start+3:break
        first,second,flipped=records[start:start+3]
        observed=decoded_bit(first['observation'])
        y=decoded_bit(second['observation']);yf=decoded_bit(flipped['observation'])
        expected_mid=truth(q['op1'],q['a'],q['b']);expected_final=truth(q['op2'],expected_mid,q['c'])
        conditional=None if observed is None else truth(q['op2'],observed,q['c'])
        cf=None if observed is None else truth(q['op2'],1-observed,q['c'])
        sensitive=conditional is not None and conditional!=cf
        cases.append(dict(id=q['id'],observed_intermediate=observed,expected_intermediate=expected_mid,
            observed_final=y,expected_final=expected_final,intermediate_correct=observed==expected_mid,
            end_to_end_correct=y is not None and y==expected_final,
            follows_actual_intermediate=y is not None and y==conditional,
            flip_output=yf,flip_expected=cf,flip_correct=yf is not None and yf==cf,
            intervention_sensitive=sensitive,observed_intervention_effect=(y!=yf) if y is not None and yf is not None else None,
            upstream_error=observed!=expected_mid,
            error_propagated=observed!=expected_mid and y is not None and y!=expected_final))
    complete=len(records)==TOTAL
    raw_direct=[scoring_record(r) for r in records[:16] if r['fields'] is not None and r['observation'].get('accepted') is True]
    hist=compare_all(raw_direct,baseline['records']) if len(raw_direct)==16 else None
    valid_scores=[s for s in scores if s is not None]
    numeric=bool(complete and all(audits))
    integrity=bool(complete and environment_match and all(wires) and numeric and hist and hist['pass_gate']
                   and len(replays)==96 and all(replays))
    all_truth=all(s and s['primary']['primary_correct'] and s['bounded']['constrained_correct'] for s in scores)
    closed=bool(integrity and all_truth and len(cases)==32 and all(c['intermediate_correct'] and c['end_to_end_correct']
                and c['follows_actual_intermediate'] and c['flip_correct']
                and c['observed_intervention_effect']==c['intervention_sensitive'] for c in cases))
    return dict(protocol_id=ID,completed=len(records),expected=TOTAL,complete=complete,
        overall_pass=closed,decision='INCOMPLETE' if not complete else 'EXTERNAL_TWO_STEP_INTERFACE_PASS' if closed else 'FROZEN_GATE_FAILURE',
        wiring_matches=sum(wires),primary_correct=sum(s['primary']['primary_correct'] for s in valid_scores),
        bounded_correct=sum(s['bounded']['constrained_correct'] for s in valid_scores),
        per_call_numerical_passes=sum(audits),legacy_cross_path_diagnostic_failures=legacy_fail,
        direct_replays_checked=len(replays),direct_replays_passed=sum(replays),historical_replay=hist,
        environment_match=environment_match,integrity_pass=integrity,cases=cases,
        completed_programs=len(cases),end_to_end_correct=sum(c['end_to_end_correct'] for c in cases),
        intervention_sensitive_cases=sum(c['intervention_sensitive'] for c in cases),
        upstream_errors=sum(c['upstream_error'] for c in cases),
        error_propagation_test_observed=any(c['upstream_error'] for c in cases),
        scope=CONFIG['scope'],v8_final_pass_confirmed=False,training_steps=0,memory_write_operations=0,
        native_planning=False,deployment_ready=False,formal_l6_pass=False)
