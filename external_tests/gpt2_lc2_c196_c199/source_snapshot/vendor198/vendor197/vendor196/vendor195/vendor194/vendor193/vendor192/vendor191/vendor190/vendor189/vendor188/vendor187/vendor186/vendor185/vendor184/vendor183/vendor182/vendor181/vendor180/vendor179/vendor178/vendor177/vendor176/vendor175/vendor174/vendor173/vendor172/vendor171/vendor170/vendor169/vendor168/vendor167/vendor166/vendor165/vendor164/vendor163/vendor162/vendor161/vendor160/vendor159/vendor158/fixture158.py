"""Small analytic protocol fixture only; no model or scientific measurement claims."""
import copy
from core158 import *
from fixture157 import context as old_context
def context():
    c=old_context();c['failed157']=dict(decision='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION');return c
def inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Synthetic protocol marker only');paths[k]=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
def loader(*args):return copy.deepcopy(context())
class Analytic:
    def __init__(self,out,c,points,directions,calls):
        self.points=points;self.directions=directions;self.calls=calls;calls['model_forwards']+=3;calls['functional_FP64_forwards']+=2;self.references={n:np.zeros(6) for n in REFERENCES};self.baseline_records=dict(native_logits=dict(maximum_absolute_error=1e-4,passed_original_replay_tolerance=False,comparison_role='cross_run_descriptive_only'))
        self.reference_ids={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()}
    def measure(self,sid,n):
        self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;r=self.points[(sid,n)][:6]
        return dict(state_id=sid,label=n,FP64_point_sha256=point_hash(self.points[(sid,n)]),absolute_response=r.tolist())
    def ad(self,sid):
        self.calls['model_forwards']+=1;self.calls['functional_FP64_forwards']+=1;self.calls['backward_calls']+=9;self.calls['backward_attempts']+=9;r=self.points[(sid,'zero')][:6];g=self.directions[sid][:6];losses={}
        for n,b in self.references.items():
            rr=r-b;i=int(np.argmax(abs(rr)));losses[n]=float(2*max(abs(rr[i])/.0018-1,0)*np.sign(rr[i])*g[i]/.0018)
        return dict(state_id=sid,label='zero',FP64_point_sha256=point_hash(self.points[(sid,'zero')]),absolute_response=r.tolist(),coordinate_derivatives=g.tolist(),response_loss_derivatives=losses)
    def verify(self,*args):need({n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in self.references.items()}==self.reference_ids,'Fixed references changed')
