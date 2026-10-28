"""Full84 analytic fixtures; no real model evidence."""
import copy
from core160 import *
from fixture159 import context as context159

def context():
 old=context159();template=old['data']['d'];V=template['V'];before=old['data']['before'];after=old['data']['after'];d=dict(V=V.copy(),points={},weights={});points={};coefs={n:archive155.prev.interpolate(before,after,a) for n,a in LABELS.items()}
 for i,sp in enumerate(registry()):
  sid=sp['state_id'];q=copy.deepcopy(template[ 'points'][previous.FOCUS[i%2]]);d['points'][sid]=q;d['weights'][sid]=np.array([.2,.3,.5])
  for n in LABELS:points[n+'__'+sid]=arithmetic.proposal(q['prepared'],q['raw'],V,arithmetic.effective(coefs[n],d['weights'][sid],archive155.ARM))
 return dict(data=dict(d=d,before=before,after=after,coefs=coefs,points=points),baseline=dict(FP32_cached_response_reference=[0.]*6),baseline_logits=dict(FP32_logits_cast64=np.zeros((12,8))),completed159=dict(decision='C159_COEFFICIENT_CHAIN_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED'))
def inputs(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
 for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Synthetic full84 fixture marker only');paths[k]=p
 c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c

def loader(*args):return context()
class Analytic:
 def __init__(self,out,ctx,c,points,native,coefs,calls):
  self.c=c;self.points=points;self.native_points=native;self.coefs=coefs;self.calls=calls;calls['model_forwards']+=3;calls['functional_FP64_forwards']+=2;self.ids={n:array_digest(a) for n,a in c.items()}
 def evaluate(self,sid,label,gradient=False):
  self.calls['model_forwards']+=2;self.calls['functional_FP64_forwards']+=2;pt=self.points[(sid,label)];p=self.coefs[label];w=self.c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];o=np.r_[pt[:6],np.zeros(16)];parts=loss(o,0.,pt,self.c[sid+'__prepared'],e,sid.split('__')[0]);r=dict(state_id=sid,case=sid.split('__')[0],label=label,FP64_point_sha256=point_hash(pt),parameters_digest=parameter_hash(p),observed=o.tolist(),KL=0.,objective_parts=parts,FP64_acceptance_evaluated=False)
  if gradient:
   self.calls['backward_calls']+=2;self.calls['backward_attempts']+=2;u=self.c['V']@(self.c[sid+'__raw']-self.c[sid+'__prepared']);rr=o[:6];top=abs(rr)==max(abs(rr));ge=np.zeros_like(e)
   for i in range(6):
    factor=2*max(max(abs(rr))/.0018-1,0)*np.sign(rr[i])/.0018*top[i]/top.sum();ge[0,i]+=factor*u[i];ge[1,i]+=factor*np.roll(u,1)[i]
   r['response_gradient']=(w[:,None,None]*ge[None]).tolist();r['guard_gradient']=(w[:,None,None]*(2*.001*e/e.shape[-1])[None]).tolist()
  return r
 def native(self,sid,label='zero'):
  self.calls['model_forwards']+=2;p=self.native_points[(sid,label)];o=np.r_[p[:6].astype(float),np.zeros(16)];m=archive155.full_metric(o,0.,p,sid.split('__')[0]);q=archive155.endpoint_gates(m,sid.split('__')[0],p,self.c[sid+'__prepared'].astype(np.float32),True);return dict(state_id=sid,case=sid.split('__')[0],label=label,metrics=m,qualification=q,actual_physical_FP32_commit=False,synthetic_fixture=True)
 def verify(self,*args):compare({n:array_digest(a) for n,a in self.c.items()},self.ids,'Frozen full84 fixture constants')
