"""CPU engineering check on random tiny Qwen3, not pretrained Qwen evidence."""
import torch,transformers
from transformers import Qwen3Config,Qwen3ForCausalLM
from generation_audit import locked_generate
from candidate_backend import score_candidates
from numerical_diagnostic import evaluate_numeric,audit_numeric
from constrained_backend import constrained_evaluate,BitThenEOS
class Tok:
    eos_token_id=2;bos_token_id=None
    def encode(self,s,add_special_tokens=False):return {'0':[15],'1':[16]}.get(s,[3,4,5])
    def decode(self,ids,skip_special_tokens=True):return ''.join({15:'0',16:'1'}.get(x,f'<{x}>') for x in ids if x!=2)
def main():
    torch.set_num_threads(1);torch.manual_seed(2705);tok=Tok()
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,max_position_embeddings=256,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,eos_token_id=2,bos_token_id=None,pad_token_id=2)).eval();model.requires_grad_(False)
    prompt='tiny CPU check';x=torch.tensor([tok.encode(prompt)])
    with torch.inference_mode():before,cfg_before=locked_generate(model,tok,x,torch.ones_like(x))
    scores=score_candidates(model,tok,prompt);result=constrained_evaluate(model,tok,prompt)
    numeric=evaluate_numeric(model,tok,prompt)
    report=audit_numeric(dict(run_id='tiny',scores=scores,constrained=result,numeric=numeric))
    assert report['same_forward_mask_exact']
    assert max(abs(a-b) for name in ('last_no_cache','last_cache') for a,b in zip(result['trace']['first_bit_logits'],numeric[name]['bit_logits']))<=1e-5
    expected=int(scores['bit_logp'][1]>scores['bit_logp'][0]);assert result['output_ids']==[[15,16][expected],2]
    assert result['raw']==str(expected) and not result['truncated'] and result['trace']['steps']==[0,1]
    gap=result['trace']['first_bit_logits'][1]-result['trace']['first_bit_logits'][0]
    delta=abs(gap-(scores['bit_logp'][1]-scores['bit_logp'][0]));assert delta<=1e-5
    with torch.inference_mode():after,cfg_after=locked_generate(model,tok,x,torch.ones_like(x))
    assert torch.equal(before,after) and cfg_before==cfg_after
    # Check the exact mask semantics, including a larger disallowed logit.
    grammar=BitThenEOS(3,[15,16],2);logits=torch.zeros(1,32);logits[0,15]=-4.;logits[0,16]=3.;logits[0,30]=100.
    masked=grammar(x,logits);assert torch.equal(masked[0,[15,16]],logits[0,[15,16]]) and int(masked.argmax())==16
    assert torch.isfinite(masked).sum()==2
    next_mask=grammar(torch.tensor([[3,4,5,16]]),logits);assert torch.isfinite(next_mask).sum()==1 and int(next_mask.argmax())==2
    try:grammar(torch.tensor([[3,4,5,16,2]]),logits)
    except ValueError:pass
    else:raise AssertionError('Third grammar step must be rejected')
    print('CONSTRAINT_TINY_CPU_OK',dict(torch=torch.__version__,transformers=transformers.__version__,model_type=model.config.model_type,numerical_diagnostic=report,chosen_bit=expected,gap_abs_delta=delta,free_output_unchanged=True,mask_preserves_bit_logits=True,EOS_forced_by_controller=True))
if __name__=='__main__':main()
