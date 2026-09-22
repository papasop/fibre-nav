"""Frozen program inventory. No answer labels or Boolean oracle in execution specs."""
import hashlib,json
from pathlib import Path
MODEL='Qwen/Qwen3-1.7B'
REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
ID='QWEN_CHAIN_CPU_V9'
OPS=('OR','AND','XOR','EQUAL')
TOTAL=112
TOL=1e-5
CONFIG=dict(protocol_id=ID,model=MODEL,revision=REVISION,device='cpu',dtype='float32',
    threads=4,attention='sdpa',seed=84917,direct_cases=16,programs=32,
    slots=TOTAL,max_bounded_generations=112,free_generations=0,
    primary_candidate_forwards=336,numerical_diagnostic_forwards=224,
    unique_model_prompts=16,aligned_logit_atol=TOL,replay_atol=TOL,
    input_policy='Step2.a is the actual decoded Step1 bit; no oracle repair; flip control uses 1-actual_bit.',
    invalid_bit_policy='Record dependent slots as skipped; fail the total gate; continue independent programs.',
    training_steps=0,adapter_count=0,memory_write_operations=0,
    v8_final_pass_assumed=False,scope='External two-step controller, finite authored programs; no native planning, parameter memory or generalization claim.')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def programs():
    out=[]
    for i,(o1,o2) in enumerate((a,b) for a in OPS for b in OPS):
        for variant,t in enumerate((i%8,(i%8)^7)):
            out.append(dict(id=f'p{i:02d}_{variant}',op1=o1,op2=o2,a=(t>>2)&1,b=(t>>1)&1,c=t&1))
    return out
def schedule():
    plan=[]
    for a in (0,1):
        for b in (0,1):
            for op in OPS:
                plan.append(dict(id=f'direct-{a}{b}-{op}',kind='direct',fields=dict(op=op,a=a,b=b)))
    for q in programs():
        first=len(plan)
        for kind in ('step1','step2','flip_step2'):
            plan.append(dict(id=q['id']+'-'+kind,kind=kind,program=q,first_index=first))
    if len(plan)!=TOTAL:raise ValueError('Inventory')
    return plan
def verify_inventory():
    root=Path(__file__).resolve().parent
    if json.loads((root/'FROZEN_PLAN.json').read_text())!=schedule():raise ValueError('Frozen plan changed')
    if json.loads((root/'protocol.json').read_text())!=CONFIG:raise ValueError('Frozen config changed')
    return schedule()
