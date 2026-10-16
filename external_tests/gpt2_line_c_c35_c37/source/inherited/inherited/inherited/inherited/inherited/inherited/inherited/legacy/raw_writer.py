"""Four learned branches conditioned on read contents and continuous state features.

No stored endpoints, geometry solvers, backward pass or repair inside forward.
"""
import torch

class StateWriter(torch.nn.Module):
 def __init__(self,input_dim=3147,output_dim=36864,width=64):
  super().__init__()
  self.experts=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(input_dim,width),torch.nn.Tanh(),torch.nn.Linear(width,output_dim)) for _ in range(4)])
 def initialise_c9(self,state):
  with torch.no_grad():
   for e in self.experts:
    e[0].weight[:32].zero_();e[0].weight[:32,:3077].copy_(state['net.0.weight']);e[0].bias[:32].copy_(state['net.0.bias'])
    e[2].weight.zero_();e[2].weight[:,:32].copy_(state['net.2.weight']);e[2].bias.copy_(state['net.2.bias'])
 def forward(self,x,routes):
  # Grouped branches retain differentiability during offline fitting.
  parts=[]
  for i,e in enumerate(self.experts):parts.append(e(x))
  all_outputs=torch.stack(parts,dim=1)
  return all_outputs[torch.arange(x.shape[0],device=x.device),routes]

def route_from_features(features):
 # Features are the original unnormalised C9 exact-prompt features.
 if features.shape[-1]!=3077:raise ValueError('Original C9 feature shape required')
 return 2*(features[...,3072]>=0).long()+(features[...,3073]>=0).long()

def prepared_input(feature,extra,c9_norm,extra_norm):
 return torch.cat([(feature-c9_norm['mean'])/c9_norm['scale'],(extra-extra_norm['mean'])/extra_norm['scale']],dim=-1)

def one_raw_proposal(head,x,route,output_scale):
 with torch.no_grad():return head(x.unsqueeze(0),route.reshape(1))[0]*output_scale

def strict_loss(logits,n_anchor,response_ids,reference,read_ids,target,teacher_mse,settings):
 anchors=logits[:n_anchor];e=anchors[:,response_ids].mean(0)-reference[:,response_ids].mean(0)
 response=(e/settings['internal_response_target']).square().max()
 r=logits[n_anchor:,read_ids];margin=r[:,1]-r[:,0];bits=target.repeat(margin.numel()//target.numel())
 read=torch.relu(settings['internal_read_target']-(2*bits-1)*margin).square().mean()/settings['internal_read_target']**2
 logp=reference.log_softmax(-1);kl=(logp.exp()*(logp-anchors.log_softmax(-1))).sum(-1).mean()
 kl_loss=torch.relu(kl/settings['kl_budget']-1.).square()
 loss=settings['loss_weights']['teacher_mse']*teacher_mse+response+read+kl_loss
 return loss,{'response':response,'read':read,'KL':kl_loss,'max_drift':e.abs().max(),'mean_KL':kl}
