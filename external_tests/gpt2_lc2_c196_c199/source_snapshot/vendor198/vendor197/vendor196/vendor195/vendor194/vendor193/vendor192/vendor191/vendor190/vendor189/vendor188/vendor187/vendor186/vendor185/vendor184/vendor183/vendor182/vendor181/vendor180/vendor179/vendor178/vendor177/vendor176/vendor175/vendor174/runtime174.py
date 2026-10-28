"""Same function and seals; six coordinate AD plus scalar R/G per anchor."""
from core174 import *
from runtime172 import FinalHeadProbes
class CoordinateProbes(FinalHeadProbes):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);r=clean_json(self.reference_file.read_bytes());r['protocol']=PROTOCOL;dump(self.reference_file,r);self.reference_file_hash=sha(self.reference_file);self.verify(True)
 def coefficient_record(self,sid,p,expected,gradient=False):
  need(not gradient or (self.batch is not None and self.batch['folder'].name=='anchor'),'Coordinate AD at anchor only');need(self.batch is not None and self.authorized_point is None,'Sealed batch before endpoint');self.verify_batch(p);need(len(self.batch['seen'])<84 and sid==IDS[len(self.batch['seen'])],'All84 ordered endpoint batch');self.authorized_point=self.batch['point_ids'][sid];t=self.torch
  try:
   with t.enable_grad() if gradient else t.no_grad():
    leaf=t.tensor(p.copy(),dtype=t.float64,requires_grad=gradient);point,e=self.map_head(leaf,sid);close(point.detach().numpy(),expected,'Actual current Torch/NumPy point',1e-12);o,kl=self.full_forward(point,gradient);parts=self.terms(o,kl,point,e,sid);row=dict(state_id=sid,case=sid.split('__')[0],parameters_digest=parameter_hash(p),FP64_point_sha256=point_hash(point.detach().numpy()),sealed_numpy_point_sha256=point_hash(expected),actual_FP64_point_artifact=self.batch['folder'].name+'/ACTUAL_TORCH_FP64_POINTS.npz',map_Linf=float(np.max(abs(point.detach().numpy()-expected))),observed=o.detach().numpy().tolist(),KL=float(kl.detach()),objective_parts={n:float(v.detach()) for n,v in parts.items()},FP64_acceptance_evaluated=False);compare(row['objective_parts'],losses(row['observed'],row['KL'],point.detach().numpy(),self.chain_numpy[sid+'__prepared'],e.detach().numpy(),row['case']),'Independent NumPy loss arithmetic',atol=1e-10,rtol=1e-9)
    if gradient:
     J=[]
     for j in range(6):
      self.calls['backward_attempts']+=1;g=t.autograd.grad(o[j],leaf,retain_graph=True,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1;self.calls['response_coordinate_backward_calls']+=1;need(bool(t.isfinite(g).all()) and g.shape==leaf.shape,'Finite signed response-coordinate gradient');J.append(g.detach().numpy().copy())
     row['response_coordinate_gradients']=np.stack(J).tolist();self.calls['response_Jacobian_blocks_completed']+=1
     for name,value,retain in [('response_gradient',parts['response_loss'],True),('guard_gradient',sum(v for n,v in parts.items() if n!='response_loss'),False)]:
      self.calls['backward_attempts']+=1;g=t.autograd.grad(value,leaf,retain_graph=retain,create_graph=False,allow_unused=False)[0];self.calls['backward_calls']+=1;need(bool(t.isfinite(g).all()) and g.shape==leaf.shape,'Finite scalar loss gradient');row[name]=g.detach().numpy().tolist()
     need(leaf.grad is None,'No accumulated parameter gradients')
  finally:self.authorized_point=None
  compare(row['FP64_point_sha256'],self.batch['point_ids'][sid],'Actual saved point identity');self.batch['seen'].append(sid);self.verify();return row
def factory(*a):return CoordinateProbes(*a)
