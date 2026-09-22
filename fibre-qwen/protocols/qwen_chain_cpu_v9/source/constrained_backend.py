"""Bit-then-EOS grammar: syntax is enforced externally, bit value is model-selected.
No operation, input-pair or truth label is passed to this backend.
"""
from generation_audit import locked_generate

class BitThenEOS:
    def __init__(self,prompt_length,bit_ids,eos_id):
        if len(bit_ids)!=2 or len(set(bit_ids+[eos_id]))!=3:raise ValueError('CONSTRAINT_TOKEN_MAPPING')
        self.prompt_length=prompt_length;self.bit_ids=list(bit_ids);self.eos_id=eos_id;self.steps=[];self.first_bit_logits=None;self.masked_bit_logits=None;self.finite_bit_count=None
    def __call__(self,input_ids,scores):
        import torch
        if input_ids.shape[0]!=1 or scores.shape[0]!=1:raise ValueError('SINGLE_SEQUENCE_REQUIRED')
        step=input_ids.shape[1]-self.prompt_length
        if step!=len(self.steps) or step not in (0,1):raise ValueError('CONSTRAINT_STEP_ORDER')
        self.steps.append(step);masked=torch.full_like(scores,float('-inf'))
        if step==0:
            selected=scores[0,self.bit_ids]
            if not torch.isfinite(selected).all():raise ValueError('NONFINITE_BIT_LOGITS')
            self.first_bit_logits=selected.double().tolist()
            masked[:,self.bit_ids]=scores[:,self.bit_ids]
            self.masked_bit_logits=masked[0,self.bit_ids].double().tolist()
            self.finite_bit_count=int(torch.isfinite(masked).sum())
        else:
            if int(input_ids[0,-1]) not in self.bit_ids:raise ValueError('PREVIOUS_TOKEN_NOT_A_BIT')
            masked[:,self.eos_id]=0.0
        return masked

def constrained_evaluate(model,tok,prompt):
    import torch
    from transformers import LogitsProcessorList
    if not isinstance(prompt,str):raise TypeError('Prompt must be text only')
    bits=[tok.encode(x,add_special_tokens=False) for x in ('0','1')]
    if any(len(x)!=1 for x in bits):raise ValueError('SINGLE_TOKEN_BITS_REQUIRED')
    ids=[x[0] for x in bits];prefix=tok.encode(prompt,add_special_tokens=False)
    if not prefix:raise ValueError('EMPTY_PROMPT')
    x=torch.tensor([prefix],dtype=torch.long,device='cpu');grammar=BitThenEOS(len(prefix),ids,tok.eos_token_id)
    with torch.inference_mode():
        y,cfg=locked_generate(model,tok,x,torch.ones_like(x),extra_processors=LogitsProcessorList([grammar]))
    output=y[0,len(prefix):].tolist()
    return dict(raw=tok.decode(output,skip_special_tokens=True),output_ids=output,
        truncated=len(output)>=128 and output[-1]!=tok.eos_token_id,effective_config=cfg,
        policy=dict(decoder='bit_then_eos',allowed_token_ids=ids,forced_eos_token_id=tok.eos_token_id,max_output_tokens=2),
        trace=dict(steps=grammar.steps,first_bit_logits=grammar.first_bit_logits,masked_bit_logits=grammar.masked_bit_logits,finite_bit_count=grammar.finite_bit_count))
