"""Runtime capture after Transformers has resolved generation defaults."""
import inspect,re,json

def json_safe(value):
    """Preserve nested runtime tensor values as ordinary JSON arrays/scalars."""
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if all(callable(getattr(value,k,None)) for k in ('detach','cpu','tolist')):
        return json_safe(value.detach().cpu().tolist())
    raise TypeError('Unsupported configuration value: '+type(value).__name__)

def content_parse(raw,truncated=False):
    """Accept one unique FINAL line at end; no answer searching or repair."""
    if truncated:return None
    lines=raw.strip().splitlines()
    if not lines:return None
    matches=[re.fullmatch(r'FINAL: ([01])',line.strip()) for line in lines]
    valid=[m for m in matches if m]
    if len(valid)!=1 or matches[-1] is None:return None
    if sum('FINAL:' in line for line in lines)!=1:return None
    return int(matches[-1].group(1))

def locked_generate(model,tok,enc,mask):
    captured=[]
    original=model._get_logits_processor
    signature=inspect.signature(original)
    had_local='_get_logits_processor' in model.__dict__
    previous=model.__dict__.get('_get_logits_processor')
    def audited(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        cfg=bound.arguments['generation_config']
        effective=json_safe(cfg.to_dict())
        json.dumps(effective,allow_nan=False)
        captured.append(effective)
        if cfg.do_sample is not False or cfg.num_beams!=1:
            raise RuntimeError('Effective decoding is not greedy; abort rather than report invalid comparisons')
        if cfg.max_new_tokens!=128 or cfg.repetition_penalty!=1.0:
            raise RuntimeError('Generation configuration differs from frozen protocol')
        return original(*args,**kwargs)
    model._get_logits_processor=audited
    try:
        result=model.generate(input_ids=enc,attention_mask=mask,
            do_sample=False,num_beams=1,num_return_sequences=1,
            max_new_tokens=128,use_cache=True,
            temperature=1.0,top_p=1.0,top_k=0,
            repetition_penalty=1.0,no_repeat_ngram_size=0,
            eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,
            bos_token_id=tok.bos_token_id)
        if len(captured)!=1:raise RuntimeError('Cannot attest effective generation configuration')
        return result,captured[0]
    finally:
        if had_local:model._get_logits_processor=previous
        else:delattr(model,'_get_logits_processor')
