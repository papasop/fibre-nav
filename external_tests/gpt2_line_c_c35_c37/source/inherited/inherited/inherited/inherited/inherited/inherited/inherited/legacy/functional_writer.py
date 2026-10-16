"""Differentiable temporary chart overrides; never copy predicted tensors into Parameters."""
import torch
from torch.func import functional_call

def chart_spec(model,params):
 names={id(p):n for n,p in model.named_parameters()}
 result=[(names[id(p)],tuple(p.shape),p.numel()) for p in params]
 if any('lora_B.default.' not in n for n,_,_ in result):raise ValueError('Only default memory B allowed')
 return result

def flat_overrides(theta,spec):
 if theta.ndim!=1 or theta.numel()!=sum(n for _,_,n in spec):raise ValueError('Chart dimension mismatch')
 offsets=0;mapping={}
 for name,shape,size in spec:
  mapping[name]=theta[offsets:offsets+size].reshape(shape);offsets+=size
 return mapping

def last_logits(model,batch,theta,spec):
 out=functional_call(model,flat_overrides(theta,spec),(),
                     dict(batch,use_cache=False),strict=False)
 last=batch['attention_mask'].sum(1)-1
 return out.logits[torch.arange(last.numel(),device=last.device),last,:]

def objective_terms(logits,n_anchor,response_ids,reference,read_ids,target_bits):
 a=logits[:n_anchor];r=a[:,response_ids].mean(0)
 e=r-reference[:,response_ids].mean(0)
 logq=torch.log_softmax(a,dim=-1);logp=torch.log_softmax(reference,dim=-1)
 kl=(logp.exp()*(logp-logq)).sum(-1).mean()
 reads=logits[n_anchor:,read_ids];margin=reads[:,1]-reads[:,0]
 bits=target_bits.repeat(margin.numel()//target_bits.numel())
 signed=(2*bits-1)*margin
 return {'read':torch.nn.functional.softplus(1.-signed).mean(),
         'response':torch.nn.functional.smooth_l1_loss(e/.1,torch.zeros_like(e),beta=1.,reduction='mean'),
         'KL':torch.relu(kl/.005-1.),
         'measured_mean_KL':kl,'maximum_response_drift':e.abs().max(),
         'minimum_read_margin':signed.min()}

def combined_loss(mse,terms,weights):
 return weights['parameter_mse']*mse+sum(weights[n]*terms[n] for n in ('read','response','KL'))
