"""Analytic full-coordinate fixtures; never GPT2 or derivative-accuracy evidence."""
import contextlib,copy,io,shutil
from core145 import *
from fixture143 import worker_data as original_data
from fixture138 import metric
from fixture133 import pack

def data145():
    old=original_data();points={sid:{k:p[k] for k in ('prepared','raw','features')} for sid,p in old['points'].items()};d=dict(points=points,V=old['V'],a=old['initial_a'].copy(),C140_a=old['initial_a']+.001,writer=old['writer'],control=old['control'],modelpins=old['modelpins'],C143_summary=dict(decision='C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE'),C144_summary=dict(decision='C144_READONLY_JOINT84_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'));refs={}
    for sp in registry():
        sid=sp['state_id'];p=points[sid]
        for kind,point,w in [('prepared',p['prepared'],False),('raw',p['raw'],True),('C140',parent.proposal(p['prepared'],p['raw'],d['V'],d['C140_a'])[0],True),('C143',parent.proposal(p['prepared'],p['raw'],d['V'],d['a'])[0],True)]:refs[(sid,kind)]=[metric(point,sp['case'],w)]*2
    d['refs']=refs;return d

def dummy_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('Synthetic C145 input '+k).encode());paths[k]=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c

def emit(out,d,paths):
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});dump(out/'ALL84_EXPOSED_TRAINING_REGISTRY.json',registry());dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(d));dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'C144_BOUND_RECEIPT.json',dict(synthetic_fixture=True));np.savez_compressed(out/'BASELINE_C143_CONTROL.npz',a=d['a'],V=d['V'],C140_a=d['C140_a']);np.savez_compressed(out/'ALL84_FROZEN_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')})

class Toy:
    def __init__(self,out,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.pins=data['modelpins'];calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
    def metrics(self,point,case,write=True):
        self.phase.consume(point,case,write);prefix='preflight' if self.phase.phase=='preflight' else 'path';self.calls[prefix+'_metric_attempts']+=1;self.calls[prefix+'_metrics']+=1;self.calls['model_forwards']+=2;m=metric(point,case,write)
        if (self.mode=='preflight_drift' and prefix=='preflight') or (self.mode=='repeat_drift' and prefix=='path' and self.calls['path_metrics']>756):m['observed'][0]+=.0003;m['response_drift']=float(max(abs(x) for x in m['observed'][:6]));m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        if self.mode=='mutate_basis' and prefix=='path':self.data['V'][0,0]+=.0001
        return m
    def invariant(self):pass

class Analytic:
    def __init__(self,rt,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.n=0;self.p=0;self.g=0;self.b=0;self.attempts=0
    def point(self,sid):p=self.data['points'][sid];return parent.proposal(p['prepared'],p['raw'],self.data['V'],self.data['a'])[0]
    def parity(self,sp):
        sid=sp['state_id'];point=self.point(sid);err=self.phase.functional(sid,self.data['a'],point,'parity');m=metric(point,sp['case'],True);self.n+=1;self.p+=1;self.calls['model_forwards']+=2
        if self.mode=='parity_drift':m['observed'][0]+=.001
        return dict(state_id=sid,point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err,observed=m['observed'],KL=m['KL'])
    def gradient(self,sp,baseline):
        sid=sp['state_id'];p=self.data['points'][sid];point=self.point(sid);err=self.phase.functional(sid,self.data['a'],point,'gradient');m=metric(point,sp['case'],True);self.n+=1;self.calls['model_forwards']+=2
        if self.mode=='gradient_forward_drift':m['observed'][0]+=.001
        close(m['observed'],baseline['observed'],'Synthetic gradient forward parity',2e-6);self.attempts+=2;self.b+=2;self.g+=1;o=np.asarray(m['observed']);a=self.data['a'];parts=auditcore.loss(o,m['KL'],point,p['prepared'],a,sp['case']);gp=np.zeros(DIM);e=np.maximum(abs(o[:6])/.00125-1,0);s=e*e;dr=2*e*np.sign(o[:6])/.00125;ties=s==max(s);gp[:6]=dr/6+dr*ties/max(1,sum(ties))
        signs=2*np.array(safe.target_bits(sp['case'],True)*4)-1;rg=-2*signs*np.maximum(1-signs*o[6:]/.075,0)/.075/16
        for j in range(16):gp[6+j%4]+=rg[j]
        if m['KL']>.0045:gp[:6]+=2*(m['KL']/.0045-1)/.0045*point[:6]/6
        delta=point.astype(float)-p['prepared'].astype(float);norm=float(np.linalg.norm(delta))
        if norm>20:gp+=2*(norm/20-1)/20*delta/norm
        V=self.data['V'];z=V@(p['raw'].astype(float)-p['prepared'].astype(float));ga=(V@gp)*z+2*.001*a/len(a);mask=abs(o[:6])==max(abs(o[:6]));qgp=np.zeros(DIM);qgp[:6]=np.sign(o[:6])*mask/max(1,sum(mask))/.002;gq=(V@qgp)*z
        if self.mode=='nonfinite_gradient':gq[0]=np.nan
        digest='0'*64 if self.mode=='wrong_digest' else row_digest(a)
        return dict(**sp,baseline_coefficient_digest=digest,functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err,forward_parity_checked_before_backwards=True,observed=m['observed'],KL=m['KL'],original_objective_parts=parts,original_objective=sum(parts.values()),response_ratio=qvalue(o),original_gradient=ga.tolist(),response_gradient=gq.tolist())
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_parity_records=self.p,gradient_state_records=self.g,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,maximum_point_parity_error_Linf=0.,optimizer_objects=0,optimizer_updates=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def archive_inputs(root):
    from fixture144 import make_fixture
    from worker144 import run as run144
    root=Path(root);paths,c144,holder,folder=make_fixture(root/'parents');out=root/'C144'
    with contextlib.redirect_stdout(io.StringIO()):s=run144(paths,out,c144)
    need(s['complete'],'Synthetic C144 must audit fully');shutil.copytree(ROOT/'vendor144',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z=pack(out,root/'C144_SYNTHETIC.zip');paths=dict(paths,C144=z);c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},audit144=c144);return paths,c
