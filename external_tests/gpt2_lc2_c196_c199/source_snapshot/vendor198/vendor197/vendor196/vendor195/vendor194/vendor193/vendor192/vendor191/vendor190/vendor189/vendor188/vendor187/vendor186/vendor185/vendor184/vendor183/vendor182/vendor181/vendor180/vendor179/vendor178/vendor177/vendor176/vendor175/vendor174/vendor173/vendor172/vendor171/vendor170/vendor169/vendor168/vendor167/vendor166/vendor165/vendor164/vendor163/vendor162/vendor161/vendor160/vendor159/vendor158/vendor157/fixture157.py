"""Small synthetic workflow context. Not GPT-2 measurements and never production evidence."""
import copy
from core157 import *
def context():
    points={}
    for j,sid in enumerate(FOCUS):
        a=np.zeros(DIM,np.float32);a[:6]=np.linspace(.002,.004,6);b=a.copy();b[:6]+=np.float32((j+1)*1e-5)
        for n,v in LABELS.items():points[n+'__'+sid]=(a.astype(float)+v*(b.astype(float)-a.astype(float))).astype(np.float32)
    base=dict(FP64_response_reference=[0.]*6,FP32_cached_response_reference=[0.]*6)
    return dict(data=dict(points=points,parent=dict(decision='SYNTHETIC_PARENT_ONLY')),baseline=base,
        replay_absolute={(sid,n):points[n+'__'+sid][:6].astype(float) for sid in FOCUS for n in ('zero','plus_one')},
        archived_gradient_arithmetic=dict(archived_FP32_response_gradient_dot_actual_delta={sid:0. for sid in FOCUS}))
class Analytic:
    def __init__(self,out,context,points,directions,calls):
        self.points=points;self.directions=directions;self.calls=calls;calls['model_forwards']+=2;calls['functional_FP64_forwards']+=1
    def measure(self,sid,n):
        self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;r=self.points[(sid,n)][:6]
        return dict(state_id=sid,label=n,FP64_point_sha256=point_hash(self.points[(sid,n)]),absolute_response=r.tolist())
    def ad(self,sid):
        self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;self.calls['backward_calls']+=8;self.calls['backward_attempts']+=8
        r=self.points[(sid,'zero')][:6];g=self.directions[sid][:6];i=int(np.argmax(abs(r)));loss=float(2*(abs(r[i])/.0018-1)*np.sign(r[i])*g[i]/.0018)
        return dict(state_id=sid,label='zero',absolute_response=r.tolist(),coordinate_derivatives=g.tolist(),response_loss_derivatives=dict(local=loss,shared=loss))
    def verify(self,*a):pass
def setup_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:
        paths[k]=root/(k+'.zip');paths[k].write_bytes(b'Synthetic workflow marker, not an archive')
    c=contract();c.update(production=False,pins={k:dict(sha256=sha(p),path=str(p)) for k,p in paths.items()});return paths,c
def loader(paths,out,c,budget):return copy.deepcopy(context())
