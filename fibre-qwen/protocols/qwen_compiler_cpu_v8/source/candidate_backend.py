"""All candidates scored symmetrically; no target labels enter inference."""
def score_candidates(model,tok,prompt):
    import torch
    bits=[tok.encode(s,add_special_tokens=False) for s in ('0','1')]
    if any(len(x)!=1 for x in bits) or bits[0]==bits[1]:raise ValueError('Requires distinct single-token bits')
    eos=tok.eos_token_id
    if not isinstance(eos,int) or eos in [x[0] for x in bits]:raise ValueError('Invalid EOS')
    prefix=tok.encode(prompt,add_special_tokens=False)
    if not prefix:raise ValueError('Empty prompt')
    with torch.inference_mode():
        x=torch.tensor([prefix])
        logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1].double()
        lp=logits.log_softmax(-1);bit_logp=[float(lp[t[0]]) for t in bits]
        eos_logp=[]
        for bit in bits:
            y=torch.tensor([prefix+bit])
            tail=model(input_ids=y,attention_mask=torch.ones_like(y),use_cache=False).logits[0,-1].double().log_softmax(-1)
            eos_logp.append(float(tail[eos]))
    return dict(bit_logits=[float(logits[t[0]]) for t in bits],bit_logp=bit_logp,eos_given_bit_logp=eos_logp)
