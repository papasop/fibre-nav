"""Training-only feature maps and an immutable-base residual proposal module."""
import hashlib
import numpy as np
import torch
from c27_protocol import require,SETTINGS


def fit_map(values,ids,kind,components=8,floor=.001):
 x=np.asarray(values,dtype=np.float64);require(x.ndim==2 and x.shape[0]==len(ids) and len(set(ids))==len(ids),'Feature fit rows/identity mismatch')
 require(np.isfinite(x).all() and floor>0 and components>0,'Invalid feature fit')
 center=x.mean(0);xc=x-center;u,s,vh=np.linalg.svd(xc,full_matrices=False)
 basis=np.zeros((x.shape[1],components));take=min(components,len(s));basis[:,:take]=vh[:take].T
 scores=xc@basis;scales=np.maximum(np.sqrt(np.mean(scores*scores,axis=0)),floor)
 cp={'kind':kind,'fit_ids':list(ids),'center':torch.from_numpy(center).float(),'basis':torch.from_numpy(basis).float(),'scale':torch.from_numpy(scales).float(),
     'singular_values':torch.from_numpy(s),'floor':floor,'n_components':components,'fit_uses_development':False}
 require(torch.isfinite(cp['center']).all() and torch.isfinite(cp['basis']).all() and torch.isfinite(cp['scale']).all(),'Invalid fitted feature map')
 return cp


def map_features(values,cp):
 scores=((values-cp['center'].to(values))@cp['basis'].to(values))/cp['scale'].to(values)
 return torch.cat([scores,torch.ones((*scores.shape[:-1],1),device=scores.device,dtype=scores.dtype)],dim=-1)


def state_hash(state):
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()


def tensor_hash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


class ResidualWriter(torch.nn.Module):
 def __init__(self,base,feature_map,arm,output_dim=36864,route=2):
  super().__init__();require(arm in ('hidden_control','state_bypass'),'Unknown arm')
  self.base=base;self.arm=arm;self.route=route
  for p in self.base.parameters():p.requires_grad_(False)
  self.register_buffer('center',feature_map['center'].clone());self.register_buffer('basis',feature_map['basis'].clone());self.register_buffer('scale',feature_map['scale'].clone())
  self.register_buffer('matrix',torch.zeros(output_dim,feature_map['n_components']+1));self.zero=True
 def features(self,x):
  if self.arm=='hidden_control':v=self.base.experts[self.route][1](self.base.experts[self.route][0](x))
  else:v=x
  scores=((v-self.center)@self.basis)/self.scale
  return torch.cat([scores,torch.ones_like(scores[...,:1])],dim=-1)
 def set_matrix(self,matrix):
  matrix=torch.as_tensor(matrix,dtype=self.matrix.dtype,device=self.matrix.device)
  require(matrix.shape==self.matrix.shape and torch.isfinite(matrix).all(),'Invalid residual matrix')
  with torch.no_grad():self.matrix.copy_(matrix)
  self.zero=bool(torch.count_nonzero(matrix).item()==0)
 def forward(self,x,routes):
  original=self.base(x,routes)
  if self.zero:return original
  residual=self.features(x)@self.matrix.T
  return torch.where((routes==self.route).unsqueeze(-1),original+residual,original)


def activation_audit(base,x_by_case,ids_by_case):
 ans={}
 with torch.no_grad():
  for route,case in enumerate(('xor_00','xor_01','xor_10','xor_11')):
   x=x_by_case[case];layer=base.experts[route][0];z=torch.stack([layer(row.unsqueeze(0))[0] for row in x]);h=torch.tanh(z);j=(1-h.square()).clamp_min(0.)
   zd=z.detach().cpu().double().numpy();hd=h.detach().cpu().double().numpy();xd=x.detach().cpu().double().numpy();weight=layer.weight.detach().cpu().double().numpy()
   groups={}
   for name,a,b in [('GPT2_hidden',0,3072),('read_margins',3072,3076),('requested_bit',3076,3077),('protected_response',3077,3083),('parameter_sketch',3083,3147)]:
    if b>x.shape[-1]:continue
    contribution=xd[:,a:b]@weight[:,a:b].T
    groups[name]={'median_abs_preactivation_contribution':float(np.median(np.abs(contribution))),'maximum_abs_contribution':float(np.max(np.abs(contribution))),
                  'input_maximum_abs':float(np.max(np.abs(xd[:,a:b]))),'L2_contribution_norm':float(np.linalg.norm(contribution))}
   singular=np.linalg.svd(np.concatenate([hd,np.ones((len(hd),1))],axis=1),compute_uv=False)
   ans[case]={'ids':ids_by_case[case],'n':len(hd),'z_abs_median':float(np.median(np.abs(zd))),'z_abs_maximum':float(np.max(np.abs(zd))),
              'z_abs_above5_fraction':float(np.mean(np.abs(zd)>5)),'z_abs_above10_fraction':float(np.mean(np.abs(zd)>10)),
              'hidden_abs_above_0999_fraction':float(np.mean(np.abs(hd)>=.999)),'tanh_derivative_median':float(j.median().cpu()),
              'tanh_derivative_maximum':float(j.max().cpu()),'hidden_augmented_singular_values':singular.tolist(),
              'hidden_pairwise_maximum_L2':float(np.max(np.linalg.norm(hd[:,None,:]-hd[None,:,:],axis=-1))),
              'input_group_contributions_float64_diagnostic':groups,'bias_maximum_abs':float(layer.bias.detach().abs().max().cpu()),
              'precision':'z and tanh float32 on actual device; group decomposition and spectrum float64 diagnostic only'}
 return ans
