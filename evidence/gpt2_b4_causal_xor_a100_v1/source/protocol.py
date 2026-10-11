PROTOCOL='GPT2_B4_CAUSAL_XOR_A100_DEVELOPMENT_V1'
INITIAL=[0,0,1,1]
SUBSTITUTED=[1,0,1,1]
def controller(reader, blocked=False):
    # This external controller receives no oracle labels or target contents.
    a,b=(0,0) if blocked else (reader(0),reader(1))
    if a not in (0,1) or b not in (0,1): raise ValueError('Invalid decoded bit')
    return {'read_bits':[a,b],'operation':'XOR','destination':3,'value':a^b,
            'executor':'external deterministic controller','read_blocked':blocked}
def endpoint_pass(ev,drift,kl,cfg):
    return (ev['exact_accuracy']==1 and ev['heldout_accuracy']==1
        and ev['minimum_exact_margin']>=.05 and ev['minimum_heldout_margin']>=.05
        and drift<=cfg['response_budget'] and kl<=cfg['endpoint_kl_gate'])
