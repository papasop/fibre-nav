"""Frozen grammar parsing and canonical compilation; no neural language-generalization claim."""
import hashlib,json,math
MODEL='Qwen/Qwen3-1.7B';REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
ID='QWEN_COMPILER_CPU_V8'
PAIRS=('00','01','10','11');OPS=('OR','AND','XOR','EQUAL');BLOCKS=('canonical','sentence','record','json')
TOTAL=128;TOL=1e-5;BIT_IDS=(15,16);EOS_ID=151645
from interface_compiler import RULES,OP_LABEL,SUFFIX,parse_input,compile_prompt,ParseError

POLICY=dict(decoder='bit_then_eos',allowed_token_ids=list(BIT_IDS),forced_eos_token_id=EOS_ID,max_output_tokens=2)
CONFIG=dict(id=ID,model=MODEL,revision=REVISION,device='cpu',dtype='float32',threads=4,attention='sdpa',seed=84917,
 primary_interface='declared_grammar_compiled_interface',blocks=list(BLOCKS),cases_per_block=16,checkpoints=TOTAL,
 free_generations=0,constrained_generations=128,total_generations=128,candidate_forward_calls=384,numerical_diagnostic_forward_calls=256,negative_parser_cases=24,unique_model_prompts=16,
 tie_tolerance_nats=TOL,replay_score_atol=TOL,aligned_projection_logit_atol=TOL,legacy_cross_path_gap_atol=TOL,full_repetition_per_block=16,
 constraint_policy=POLICY,training_steps=0,memory_write_operations=0,adapter_count=0,
 scope='External deterministic parsing of3 declared textual forms and JSON into(op,a,b), canonical compilation to exact V6 prompts, then frozen Qwen constrained generation. All64 accepted inputs independently evaluated and repeated;16 unique model prompts.24 invalid inputs must be rejected. No answer computation in parser/compiler, no per-case repair or result cache. This is software normalization, not native language generalization or parameter-memory computation. V7 raw archive is not audited; only its failure log motivated this test. No model updates, free-generation evaluation or deployment claim.')


def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def truth(pair,op):
    a,b=map(int,pair)
    if op=='OR':return str(a|b)
    if op=='AND':return str(a&b)
    if op=='XOR':return str(a^b)
    if op=='EQUAL':return str(int(a==b))
    raise ValueError('OP')

def make_queries(tok,baseline):
    old=baseline['queries'];expected=[(p,o) for p in PAIRS for o in OPS]
    if [(q['pair'],q['task']) for q in old]!=expected:raise ValueError('HISTORICAL_INVENTORY')
    qs=[]
    for block in BLOCKS:
        for i,(pair,op) in enumerate(expected):
            fields=dict(op=op,a=int(pair[0]),b=int(pair[1]));label=OP_LABEL[op]
            if block=='canonical':source=old[i]['content']
            elif block=='sentence':source=RULES[op]+'\n'+f'The input bits are A: {pair[0]}; B: {pair[1]}.\nApply {label} to these two bits.\n'+SUFFIX
            elif block=='record':source=RULES[op]+'\n'+f'Evaluate {label} on this record:\nB = {pair[1]}\nA = {pair[0]}\n'+SUFFIX
            else:source=json.dumps(dict(b=int(pair[1]),op=op,a=int(pair[0])),separators=(',',':'))
            # Expected target comes from the independent frozen baseline, not from parser output.
            qs.append(dict(id=f'{block}-{pair}-{op}',block=block,pair=pair,task=op,source_text=source,
                expected_fields=fields,compiled_content=old[i]['content'],prompt=old[i]['prompt']))
    # Attest the compiler/chat wrapper separately without repairing mismatches.
    for q in qs:
        c,p=compile_prompt(tok,q['expected_fields'])
        if c!=q['compiled_content'] or p!=q['prompt']:raise ValueError('CANONICAL_TEMPLATE_CHANGED')
    return qs

def parser_audit():
    from pathlib import Path
    cases=json.loads((Path(__file__).resolve().parent/'FROZEN_NEGATIVES.json').read_text());rows=[]
    if len(cases)!=24 or len({c['id'] for c in cases})!=24:raise ValueError('NEGATIVE_INVENTORY')
    for c in cases:
        try:parsed=parse_input(c['text']);error=None;rejected=False
        except ParseError as exc:parsed=None;error=str(exc);rejected=True
        rows.append(dict(**c,rejected=rejected,parsed=parsed,error=error))
    return dict(n=24,rejected=sum(r['rejected'] for r in rows),pass_gate=all(r['rejected'] for r in rows),rows=rows,model_invocations=0)

def compiler_audit(r):
    q=r['query']
    return bool(r.get('accepted') is True and r.get('model_invocations')==1 and r.get('parse_error') is None
        and r.get('parsed')==q['expected_fields'] and r.get('compiled_content')==q['compiled_content'] and r.get('compiled_prompt')==q['prompt'])

def schedule(queries):
    if [(q['block'],q['pair'],q['task']) for q in queries]!=[(b,p,o) for b in BLOCKS for p in PAIRS for o in OPS]:raise ValueError('QUERY_INVENTORY')
    plan=[dict(run_id=q['id'],repeat=False,query=q) for q in queries]+[dict(run_id=q['id']+'-repeat',repeat=True,query=q) for q in queries]
    assert len(plan)==TOTAL
    return plan

def score(record):
    q=record['query'];expected=truth(q['pair'],q['task'])
    if record.get('accepted') is False:return dict(pair=q['pair'],task=q['task'],expected=expected,primary_prediction=None,secondary_prediction=None,primary_correct=False,secondary_correct=False,signed_margin=None,gap=None)
    scores=record['scores'];a=scores['bit_logp'];b=scores['eos_given_bit_logp']
    if len(a)!=2 or len(b)!=2 or not all(math.isfinite(x) and x<=0 for x in a+b):raise ValueError('NONFINITE_OR_INVALID_SCORES')
    if sum(math.exp(x) for x in a)>1+1e-8:raise ValueError('INVALID_PROBABILITY_MASS')
    gap=a[1]-a[0];second=a[1]+b[1]-a[0]-b[0]
    pred=None if abs(gap)<=TOL else str(int(gap>0));pred2=None if abs(second)<=TOL else str(int(second>0))
    q=record['query'];expected=truth(q['pair'],q['task'])
    return dict(pair=q['pair'],task=q['task'],expected=expected,primary_prediction=pred,secondary_prediction=pred2,
       primary_correct=pred==expected,secondary_correct=pred2==expected,signed_margin=(1 if expected=='1' else -1)*gap,gap=gap)

def group(records):
    if [(r['query']['pair'],r['query']['task']) for r in records]!=[(p,o) for p in PAIRS for o in OPS]:raise ValueError('GROUP_INVENTORY')
    rs=[score(r) for r in records];lookup={(r['pair'],r['task']):r for r in rs};oe=[];ie=[]
    def ok(a,b):return a['primary_correct'] and b['primary_correct'] and a['primary_prediction']!=b['primary_prediction']
    for p in PAIRS:
        for i,o in enumerate(OPS):
            for other in OPS[i+1:]:
                if truth(p,o)!=truth(p,other):oe.append(ok(lookup[p,o],lookup[p,other]))
        for c in range(2):
            other=p[:c]+str(1-int(p[c]))+p[c+1:]
            if other>p:
                for o in OPS:
                    if truth(p,o)!=truth(other,o):ie.append(ok(lookup[p,o],lookup[other,o]))
    assert len(oe)==14 and len(ie)==12
    return dict(n=16,primary_correct=sum(r['primary_correct'] for r in rs),secondary_correct=sum(r['secondary_correct'] for r in rs),operation_contrasts_passed=sum(oe),input_contrasts_passed=sum(ie),pass_gate=all(r['primary_correct'] for r in rs) and all(oe+ie),rows=rs)

def compare(a,b):
    if len(a)!=len(b) or not a:raise ValueError('REPEAT_INVENTORY')
    if any(r.get('accepted') is False for r in a+b):return dict(pass_gate=False,reason='PARSER_REJECTION')
    delta=max(abs(x-y) for r,t in zip(a,b) for k in ('bit_logp','eos_given_bit_logp') for x,y in zip(r['scores'][k],t['scores'][k]))
    return dict(max_abs_score_delta=delta,pass_gate=delta<=TOL)

def constrained_score(record):
    s=score(record)
    if record.get('accepted') is False:return dict(constrained_prediction=None,constrained_correct=False,constrained_format_valid=False,gap_abs_delta=None,legacy_cross_path_audit_pass=False,aligned_projection_logit_abs_delta=None,same_forward_mask_pass=False,selection_matches_primary=None,constraint_audit_pass=False)
    c=record['constrained']
    if c['policy']!=POLICY:raise ValueError('CONSTRAINT_POLICY_CHANGED')
    trace=c['trace'];logits=trace['first_bit_logits']
    if len(logits)!=2 or not all(math.isfinite(x) for x in logits):raise ValueError('INVALID_CONSTRAINT_LOGITS')
    raw=c['raw'].strip();pred=raw if raw in ('0','1') else None
    valid=pred is not None and c['output_ids']==[BIT_IDS[int(pred)],EOS_ID] and not c['truncated'] and trace['steps']==[0,1]
    traced_choice=str(int(logits[1]>logits[0]))
    gap_delta=abs((logits[1]-logits[0])-s['gap'])
    agree=s['primary_prediction'] is None or pred==s['primary_prediction']
    after=trace.get('masked_bit_logits')
    same_forward=after==logits and trace.get('finite_bit_count')==2
    from numerical_diagnostic import audit_numeric
    audit_numeric(record)  # require both independently computed paths and their frozen settings
    aligned_delta=max(abs(x-y) for name in ('last_no_cache','last_cache')
                      for x,y in zip(logits,record['numeric'][name]['bit_logits']))
    legacy=bool(valid and pred==traced_choice and agree and gap_delta<=TOL and same_forward)
    audit=bool(valid and pred==traced_choice and agree and aligned_delta<=TOL and same_forward)
    return dict(constrained_prediction=pred,constrained_correct=bool(valid and pred==s['expected']),constrained_format_valid=valid,
        gap_abs_delta=gap_delta,legacy_cross_path_audit_pass=legacy,aligned_projection_logit_abs_delta=aligned_delta,same_forward_mask_pass=same_forward,selection_matches_primary=None if s['primary_prediction'] is None else agree,constraint_audit_pass=audit)

def constrained_group(records):
    g=group(records);rows=[dict(pair=r['query']['pair'],task=r['query']['task'],**constrained_score(r)) for r in records]
    semantic=g['primary_correct']==16 and g['operation_contrasts_passed']==14 and g['input_contrasts_passed']==12
    return dict(n=16,correct=sum(r['constrained_correct'] for r in rows),format_valid=sum(r['constrained_format_valid'] for r in rows),
        audit_pass=all(r['constraint_audit_pass'] for r in rows),pass_gate=semantic and all(r['constrained_correct'] and r['constraint_audit_pass'] for r in rows),rows=rows)

def compare_constrained(a,b):
    if len(a)!=len(b) or not a:raise ValueError('REPEAT_INVENTORY')
    if any(r.get('accepted') is False for r in a+b):return dict(pass_gate=False,reason='PARSER_REJECTION')
    same=all(all(r['constrained'][k]==t['constrained'][k] for k in ('raw','output_ids','truncated','effective_config','policy')) and r['constrained']['trace']['steps']==t['constrained']['trace']['steps'] for r,t in zip(a,b))
    delta=max(abs(x-y) for r,t in zip(a,b) for x,y in zip(r['constrained']['trace']['first_bit_logits'],t['constrained']['trace']['first_bit_logits']))
    return dict(output_config_pass=same,max_abs_logit_delta=delta,pass_gate=same and delta<=TOL)

def compare_numeric(a,b):
    if len(a)!=len(b) or not a:raise ValueError('NUMERIC_REPEAT_INVENTORY')
    if any(r.get('accepted') is False for r in a+b):return dict(pass_gate=False,reason='PARSER_REJECTION')
    settings=all(r['numeric'][name]['settings']==t['numeric'][name]['settings'] for r,t in zip(a,b) for name in ('last_no_cache','last_cache'))
    delta=max(abs(x-y) for r,t in zip(a,b) for name in ('last_no_cache','last_cache') for x,y in zip(r['numeric'][name]['bit_logits'],t['numeric'][name]['bit_logits']))
    raw_delta=max(abs(x-y) for r,t in zip(a,b) for x,y in zip(r['scores']['bit_logits'],t['scores']['bit_logits']))
    return dict(max_abs_logit_delta=delta,max_abs_base_logit_delta=raw_delta,settings_match=settings,pass_gate=settings and max(delta,raw_delta)<=TOL)

def compare_all(a,b):
    out=dict(scores=compare(a,b),constrained=compare_constrained(a,b),numeric=compare_numeric(a,b))
    out['pass_gate']=all(v['pass_gate'] for v in out.values())
    return out

def summarize(records,plan,baseline,environment_match=True):
    if len(plan)!=TOTAL:raise ValueError('SCHEDULE_INVENTORY')
    if len({r['run_id'] for r in records})!=len(records):raise ValueError('DUPLICATE_RECORD')
    if len(records)>len(plan):raise ValueError('EXTRA_RECORDS')
    from numerical_diagnostic import audit_numeric
    audits=[];numerics=[];legacy=[];compilers=[]
    for r,p in zip(records,plan):
        if any(r[k]!=p[k] for k in ('run_id','repeat','query')):raise ValueError('QUERY_OR_ORDER_CHANGED')
        if type(r.get('accepted')) is not bool:raise ValueError('MISSING_PARSER_STATUS')
        compilers.append(compiler_audit(r));score(r)
        if r['accepted'] and 'constrained' not in r:raise ValueError('MISSING_CONSTRAINED_OUTPUT')
        c=constrained_score(r);audits.append(c['constraint_audit_pass']);legacy.append(c['legacy_cross_path_audit_pass'])
        if r['accepted']:numerics.append(audit_numeric(r))
        elif r.get('model_invocations')!=0 or any(k in r for k in ('scores','constrained','numeric')):raise ValueError('REJECTED_INPUT_REACHED_MODEL')
    gs={};cgs={};reps={};first={};cross={}
    for b in BLOCKS:
        a=[r for r in records if r['query']['block']==b and not r['repeat']]
        repeated=[r for r in records if r['query']['block']==b and r['repeat']]
        if len(a)==16:gs[b]=group(a);cgs[b]=constrained_group(a);first[b]=a
        if len(repeated)==16:reps[b]=dict(n=16,**compare_all(a,repeated))
    hist=compare_all(first['canonical'],baseline['records']) if 'canonical' in first else None
    if 'canonical' in first:
        for b in BLOCKS[1:]:
            if b in first:cross[b]=compare_all(first['canonical'],first[b])
    neg=parser_audit();complete=len(records)==TOTAL
    compiler_pass=complete and all(compilers) and neg['pass_gate']
    constraint_audit=complete and all(audits)
    integrity=bool(complete and environment_match and hist and hist['pass_gate'] and compiler_pass and constraint_audit
        and len(reps)==4 and all(v['pass_gate'] for v in reps.values()) and len(cross)==3 and all(v['pass_gate'] for v in cross.values()))
    semantic=bool(integrity and all(gs[b]['pass_gate'] for b in BLOCKS))
    bounded=bool(semantic and all(cgs[b]['pass_gate'] for b in BLOCKS))
    decision='INCOMPLETE'
    if complete:decision='PARSER_REPLAY_ENVIRONMENT_OR_AUDIT_FAILURE' if not integrity else 'DECLARED_GRAMMAR_COMPILED_INTERFACE_PASS' if bounded else 'CANONICAL_MODEL_COMPUTATION_FAILURE'
    return dict(protocol_id=ID,completed=len(records),expected=TOTAL,complete=complete,groups=gs,constrained_groups=cgs,
        parser_negative_audit=neg,positive_compiler_matches=sum(compilers),compiler_pass=compiler_pass,
        repeats=reps,historical_replay=hist,cross_format_equivalence=cross,environment_match=environment_match,constraint_audit_pass=constraint_audit,integrity_pass=integrity,
        numerical_diagnostic=dict(n=len(numerics),rows=numerics,legacy_cross_path_atol=TOL,legacy_cross_path_failures=sum(not x for x in legacy),legacy_cross_path_pass=complete and all(legacy),aligned_projection_logit_atol=TOL),
        semantic_pass=semantic,compiled_interface_pass=bounded,constrained_interface_pass=bounded,overall_pass=bounded,
        overall_pass_definition='V8 declared grammar software interface:64 supported inputs compiled correctly and128 independent bounded evaluations including repeats;24 rejected inputs;16 underlying canonical model prompts; all original scores, truth contrasts, constrained outputs, V6 replay, cross-format equality and aligned audits pass. No native language-generalization claim.',
        decision=decision,deployment_ready=False,unseen_expression_confirmation=False,native_free_pass=None,free_generation_evaluated=False,
        formal_l6_pass=False,native_implicit_closure=False,v6_overall_pass=False,v7_logged_overall_pass=False,v7_raw_archive_audited=False,
        training_steps=0,memory_write_operations=0,adapter_count=0,unique_model_prompts=16,scope=CONFIG['scope'])
