"""Prompt-only numerical records for the prospective aligned audit and legacy diagnostic.
Two controlled full-prefix forwards: last-logit projection with cache off/on.
The baseline full projection is recorded from the existing primary scorer.
Differences localize path sensitivity, not proof of a unique root cause.
"""
import math
VARIANTS=('last_no_cache','last_cache')
SETTINGS={'last_no_cache':dict(use_cache=False,logits_to_keep=1),
          'last_cache':dict(use_cache=True,logits_to_keep=1)}
def evaluate_numeric(model,tok,prompt):
    import torch
    x=torch.tensor([tok.encode(prompt,add_special_tokens=False)],device='cpu')
    bits=[tok.encode(s,add_special_tokens=False) for s in ('0','1')]
    if any(len(v)!=1 for v in bits):raise ValueError('DIAGNOSTIC_SINGLE_TOKEN_BITS')
    rows={}
    with torch.inference_mode():
        for name in VARIANTS:
            out=model(input_ids=x,attention_mask=torch.ones_like(x),**SETTINGS[name])
            rows[name]=dict(settings=SETTINGS[name].copy(),bit_logits=out.logits[0,-1,[v[0] for v in bits]].double().tolist())
            del out
    return rows

def audit_numeric(record):
    n=record['numeric'];base=record['scores']['bit_logits'];trace=record['constrained']['trace']
    if set(n)!=set(VARIANTS):raise ValueError('NUMERIC_VARIANT_INVENTORY')
    vectors=[base,trace['first_bit_logits'],trace['masked_bit_logits']]
    for name in VARIANTS:
        if n[name]['settings']!=SETTINGS[name]:raise ValueError('NUMERIC_VARIANT_SETTINGS')
        vectors.append(n[name]['bit_logits'])
    if any(len(v)!=2 or not all(math.isfinite(x) for x in v) for v in vectors):raise ValueError('NUMERIC_NONFINITE')
    gap=lambda v:v[1]-v[0]
    basegap=gap(base);gengap=gap(vectors[1]);lp=record['scores']['bit_logp']
    return dict(run_id=record['run_id'],base_bit_logits=base,generation_bit_logits=vectors[1],
        logsoftmax_cancellation_abs_delta=abs(basegap-gap(lp)),
        generation_vs_base_gap_abs_delta=abs(gengap-basegap),
        last_projection_vs_base_gap_abs_delta=abs(gap(n['last_no_cache']['bit_logits'])-basegap),
        cache_vs_no_cache_gap_abs_delta=abs(gap(n['last_cache']['bit_logits'])-gap(n['last_no_cache']['bit_logits'])),
        generation_vs_last_no_cache_gap_abs_delta=abs(gengap-gap(n['last_no_cache']['bit_logits'])),
        generation_vs_last_cache_gap_abs_delta=abs(gengap-gap(n['last_cache']['bit_logits'])),
        same_forward_mask_exact=vectors[1]==vectors[2],diagnostic_only=True)
