import torch,numpy as np
from c27_features import fit_map,map_features,tensor_hash,state_hash
from c28_protocol import SETTINGS,require

def feature_map(values,ids,arm):
 x=np.asarray(values,dtype=np.float64);require(x.ndim==2 and x.shape[1]==3147,'Original3147 input required')
 if arm=='pca_control':return fit_map(x,ids,arm,8,SETTINGS['scale_floor'])
 require(arm=='response_aware','Unknown arm');direct=SETTINGS['response_indices'];rest=np.array([i for i in range(3147) if i not in direct]);center=x.mean(0);basis=np.zeros((3147,8));basis[direct,range(6)]=1.
 pca=fit_map(x[:,rest],ids,'remaining_input_PCA',2,SETTINGS['scale_floor']);basis[rest,6:]=pca['basis'].double().numpy()
 score=(x-center)@basis;scale=np.maximum(np.sqrt(np.mean(score*score,axis=0)),SETTINGS['scale_floor'])
 return {'kind':arm,'fit_ids':list(ids),'fit_uses_development':False,'n_components':8,'center':torch.from_numpy(center).float(),'basis':torch.from_numpy(basis).float(),'scale':torch.from_numpy(scale).float(),'direct_response_indices':direct,'remaining_PCA_singular_values':pca['singular_values'],'floor':SETTINGS['scale_floor']}

class Map(torch.nn.Module):
 def __init__(self,cp):
  super().__init__()
  for k in ('center','basis','scale'):self.register_buffer(k,cp[k].clone())
 def forward(self,x):
  f=((x-self.center)@self.basis)/self.scale
  return torch.cat([f,torch.ones_like(f[...,:1])],-1)

class FourRouteWriter(torch.nn.Module):
 def __init__(self,base,maps,output_dim=36864):
  super().__init__();self.base=base
  for p in self.base.parameters():p.requires_grad_(False)
  self.maps=torch.nn.ModuleList([Map(cp) for cp in maps]);self.register_buffer('matrices',torch.zeros(4,output_dim,9));self.zero=[True]*4
 def set_route(self,route,matrix):
  m=torch.as_tensor(matrix,dtype=self.matrices.dtype,device=self.matrices.device);require(m.shape==self.matrices[route].shape and bool(torch.isfinite(m).all()),'Invalid route matrix')
  with torch.no_grad():self.matrices[route].copy_(m)
  self.zero[route]=bool(torch.count_nonzero(m)==0)
 def forward(self,x,routes):
  original=self.base(x,routes)
  if all(self.zero):return original
  result=original
  for route in range(4):
   if self.zero[route]:continue
   delta=self.maps[route](x)@self.matrices[route].T
   result=torch.where((routes==route).unsqueeze(-1),original+delta,result)
  return result
