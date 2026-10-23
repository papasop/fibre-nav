"""Analytic NumPy gradient backend; not a PyTorch/GPT2 validation claim."""
import contextlib,io,shutil
from core136 import *
from fixture135 import make_fixture as fixture135,Toy
from fixture133 import pack,metric
from worker135 import run as run135

def make_fixture(folder):
    folder=Path(folder);paths,c35=fixture135(folder/'parents');out=folder/'C135'
    with contextlib.redirect_stdout(io.StringIO()):s=run135(paths,out,c35,Toy)
    need(s['complete'] and s['interpretation_qualified'],'Synthetic C135 generation incomplete')
    shutil.copytree(ROOT/'vendor135',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p=pack(out,folder/'C135_SYNTHETIC.zip')
    return dict(paths,C135=p),dict(contract(),production=False,pins=dict(c35['pins'],C135=dict(path=str(p),sha256=sha(p))),source132=c35['source132'],source133=c35['source133'],source134=c35['source134'])

class Analytic:
    def __init__(self,rt,data,phase,calls,mode='normal'):
        self.rt=rt;self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.n=0;self.z=0;self.b=0;self.attempts=0
    def proposed(self,a,sid):
        p=self.data['points'][sid];return proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
    def zero(self,spec,a):
        sid=spec['state_id'];point=self.proposed(a,sid);err=self.phase.functional(sid,a,point);m=metric(point,spec['case'],True);self.calls['model_forwards']+=2;self.n+=1;self.z+=1
        if self.mode=='zero_drift':m['observed'][0]+=.001
        return dict(state_id=sid,observed=m['observed'],KL=m['KL'],point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err)
    def gradient(self,a,step,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];total=np.zeros_like(a)
        for spec in registry():
            budget();sid=spec['state_id'];p=self.data['points'][sid];point=self.proposed(a,sid);self.phase.functional(sid,a,point);m=metric(point,spec['case'],True);o=np.array(m['observed']);parts=objective(o,m['KL'],point,p['prepared'],a,spec['case'])
            self.attempts+=1
            if self.mode=='interrupt' and step==2:raise RuntimeError('Synthetic gradient interruption')
            self.calls['model_forwards']+=2;self.n+=1;self.b+=1
            excess=np.maximum(abs(o[:6])/RESPONSE_SAFETY-1,0);sq=excess**2;dr=2*excess*np.sign(o[:6])/RESPONSE_SAFETY;ties=sq==max(sq);gr=dr/6+dr*ties/max(1,sum(ties));gp=np.zeros(DIM);gp[:6]=gr
            gp[9]+=gr[0]*.2*(float(point[9])-float(point[10]));gp[10]-=gr[0]*.2*(float(point[9])-float(point[10]))
            signs=2*np.array(safe.target_bits(spec['case'],True)*4)-1;readgrad=-2*signs*np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)/MARGIN_SAFETY/16
            for k in range(16):gp[6+k%4]+=readgrad[k]
            disp=point.astype(float)-p['prepared'].astype(float);norm=float(np.linalg.norm(disp))
            if norm>RADIUS:gp+=2*(norm/RADIUS-1)/RADIUS*disp/norm
            v=self.data['V'];z=v@(p['raw'].astype(float)-p['prepared'].astype(float));g=((v@gp)*z+2*L2*a/len(a))/12
            if self.mode=='zero_gradient':g*=0
            total+=g;row=dict(state_id=sid,case=spec['case'],C136_membership='training',observed=m['observed'],KL=m['KL'],functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=0.,coefficient_digest=row_digest(a),objective_parts=parts,objective=sum(parts.values()),gradient_contribution=g.tolist());rows.append(row);dump(out/'GRADIENT_PROGRESS.json',rows)
        if self.mode=='missing_state':rows=rows[:-1]
        if self.mode=='wrong_digest':rows[0]['coefficient_digest']='0'*64
        if self.mode=='nonfinite':total[0]=np.nan
        return total,rows
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_zero_calls=self.z,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,full12_gradient_batches=self.b//12,maximum_point_parity_error_Linf=0.,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,new_Jacobians=0,new_SVD_calls=0)
