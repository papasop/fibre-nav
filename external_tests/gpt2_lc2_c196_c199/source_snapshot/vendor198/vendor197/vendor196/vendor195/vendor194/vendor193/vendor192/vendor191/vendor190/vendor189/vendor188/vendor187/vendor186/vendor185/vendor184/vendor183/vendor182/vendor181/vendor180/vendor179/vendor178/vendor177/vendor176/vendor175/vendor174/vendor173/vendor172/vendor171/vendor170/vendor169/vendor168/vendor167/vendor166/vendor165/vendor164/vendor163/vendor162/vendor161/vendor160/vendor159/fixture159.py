"""Analytic coefficient-map fixture only; never real GPT-2 evidence."""
import copy
from core159 import *
from map159 import chain_arithmetic

def context():
 rank=6;V=np.zeros((rank,DIM),np.float64);V[:,:rank]=np.eye(rank);before=np.zeros((3,2,rank),np.float64);before[:,0]=np.array([-.01,-.015,-.02])[:,None];before[:,1]=.002;after=before+np.arange(1,before.size+1).reshape(before.shape)*1e-6;d=dict(V=V,points={},weights={})
 for j,sid in enumerate(FOCUS):
  raw=np.zeros(DIM,np.float32);raw[:6]=np.linspace(.03,.08,6)*(j+1);d['points'][sid]=dict(raw=raw,prepared=np.zeros(DIM,np.float32));d['weights'][sid]=np.array([.2,.3,.5])
 coefs={n:previous.previous.parent.parent.prev.interpolate(before,after,a) for n,a in LABELS.items()};points={}
 for sid in FOCUS:
  q=d['points'][sid]
  for n in LABELS:points[n+'__'+sid]=previous.previous.parent.parent.prior.proposal(q['prepared'],q['raw'],V,previous.previous.parent.parent.prior.effective(coefs[n],d['weights'][sid],previous.previous.parent.ARM))
 data=dict(d=d,before=before,after=after,coefs=coefs,points=points,parent=dict(decision='SYNTHETIC_ONLY'));return dict(data=data,failed157=dict(decision='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION'),completed158=dict(decision='C158_SAME_RUN_TWO_STATE_FP64_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED'),replay_absolute={(sid,n):points[n+'__'+sid][:6].astype(float) for sid in FOCUS for n in ('zero','plus_one')},archived_gradient_arithmetic=dict(archived_FP32_response_gradient_dot_actual_delta={sid:0. for sid in FOCUS}))
def inputs(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
 for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Synthetic fixture marker only');paths[k]=p
 c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c

def loader(*args):return copy.deepcopy(context())
class Analytic:
 def __init__(self,out,c,points,directions,calls):
  self.points=points;self.directions=directions;self.calls=calls;self.c=constants(c['data']);calls['model_forwards']+=3;calls['functional_FP64_forwards']+=2;self.references={n:np.zeros(6) for n in REFERENCES};self.baseline_records={};self.ids={n:array_digest(a) for n,a in self.references.items()}
 def measure(self,sid,n):
  self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;return dict(state_id=sid,label=n,FP64_point_sha256=point_hash(self.points[(sid,n)]),absolute_response=self.points[(sid,n)][:6].tolist())
 def ad(self,sid):
  self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;self.calls['backward_calls']+=9;self.calls['backward_attempts']+=9;r=self.points[(sid,'zero')][:6];g=self.directions[sid][:6];chain={};losses={};u=self.c['V']@(self.c[sid+'__raw']-self.c[sid+'__prepared'])
  for n,b in self.references.items():
   rr=r-b;i=int(np.argmax(abs(rr)));factor=2*max(abs(rr[i])/.0018-1,0)*np.sign(rr[i])/.0018;ge=np.zeros((2,len(u)));ge[0,i]=factor*u[i];ge[1,i]=factor*np.roll(u,1)[i];gp=self.c[sid+'__weights'][:,None,None]*ge[None,...];losses[n]=float(factor*g[i]);chain[n]=chain_arithmetic(losses[n],ge,gp,self.c,sid)
  return dict(state_id=sid,label='zero',FP64_point_sha256=point_hash(self.points[(sid,'zero')]),absolute_response=r.tolist(),coordinate_derivatives=g.tolist(),response_loss_derivatives=losses,chain_pullbacks=chain)
 def verify(self,*args):compare({n:array_digest(a) for n,a in self.references.items()},self.ids,'Fixed references changed')
