"""Budget-scaled offline training loss. It never defines evaluation acceptance."""
import torch

def budget_terms(logits,n_anchor,response_ids,reference,read_ids,target_bits,response_budget=.002,kl_budget=.005):
    a=logits[:n_anchor];e=a[:,response_ids].mean(0)-reference[:,response_ids].mean(0)
    logq=a.log_softmax(-1);logp=reference.log_softmax(-1)
    kl=(logp.exp()*(logp-logq)).sum(-1).mean()
    read=logits[n_anchor:,read_ids];margin=read[:,1]-read[:,0]
    if margin.numel()%target_bits.numel():raise ValueError('Complete address views required')
    signed=(2*target_bits.repeat(margin.numel()//target_bits.numel())-1)*margin
    return {'response':torch.nn.functional.smooth_l1_loss(e/response_budget,torch.zeros_like(e),beta=1.),
       'read':torch.nn.functional.softplus(1.-signed).mean(),'KL':torch.relu(kl/kl_budget-1.),
       'maximum_response_drift':e.abs().max(),'minimum_read_margin':signed.min(),'measured_mean_KL':kl}

def weighted_terms(terms,weights):return sum(weights[n]*terms[n] for n in ('read','response','KL'))
