"""Analytic test double only; never a GPT2 execution or derivative certificate."""
import copy
from core148 import *
from fixture145 import data145,Toy
from fixture138 import metric

class Native(Toy):
    def metrics(self,point,case,write=True):
        m=super().metrics(point,case,write)
        if self.mode=='repeat_drift148' and self.phase.phase=='paths' and self.calls['path_metrics']>336:
            m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        if self.mode=='failed_final148' and self.phase.phase=='paths' and self.calls['path_metrics'] in (4,340):m['observed'][0]=.003;m['response_drift']=.003;m['gates']['response']=False;m['pass']=False
        return m

def worker_data():
    d=data145();d['initial_a']=d['a'].copy()
    for sp in registry():
        sid=sp['state_id'];d['refs'][(sid,'C146')]=copy.deepcopy(d['refs'][(sid,'C143')])
    U=np.array([coordinates(d['points'][sp['state_id']]['prepared'],d['points'][sp['state_id']]['raw'],d['V']) for sp in registry()]);router,W,meta=fit_router(U);d['router']=router;d['weights']={sp['state_id']:w for sp,w in zip(registry(),W)};return d,meta

def dummy_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('Synthetic C148 '+k).encode());paths[k]=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c

def loader(paths,out,c):
    d,meta=worker_data();dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',registry());dump(out/'ROUTER_DESIGN_AND_COVERAGE.json',meta);np.savez_compressed(out/'FROZEN_TRAINING_ONLY_ROUTER.npz',**d['router']);W=np.array([d['weights'][sp['state_id']] for sp in registry()]);U=np.array([coordinates(d['points'][sp['state_id']]['prepared'],d['points'][sp['state_id']]['raw'],d['V']) for sp in registry()]);np.savez_compressed(out/'ALL84_PREWRITE_ROUTING.npz',u=U,weights=W)
    for name,arrays in [('ALL84_TRAINING_INPUTS.npz',{sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()}),('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('INITIAL_C146_CONTROL.npz',dict(a=d['initial_a'])),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:np.savez_compressed(out/name,**arrays)
    dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'FROZEN_INPUT_IDENTITIES.json',fixed_digest(d));dump(out/'PARENT_AUDIT_RECEIPT.json',dict(synthetic_fixture=True));return d

class Analytic:
    def __init__(self,rt,d,phase,calls,mode='normal'):self.d=d;self.phase=phase;self.calls=calls;self.mode=mode;self.n=0;self.z=0;self.w=0;self.b=0;self.attempts=0;self.batches=0
    def proposed(self,params,sid,arm):
        p=self.d['points'][sid];a=effective(params,self.d['weights'][sid],arm);return proposal(p['prepared'],p['raw'],self.d['V'],a)[0]
    def parity(self,sp,params,arm):
        sid=sp['state_id'];point=self.proposed(params,sid,arm);err=self.phase.functional(sid,params,arm,point);m=metric(point,sp['case'],True);self.n+=1;self.calls['model_forwards']+=2
        if self.phase.phase=='functional_zero':self.z+=1
        else:self.w+=1
        if self.mode=='warm_drift' and self.phase.phase=='functional_warm':m['observed'][0]+=.001
        return dict(state_id=sid,arm=arm,point_sha256=safe.tensor_hash(point),observed=m['observed'],KL=m['KL'],point_parity_error_Linf=err)
    def gradient(self,params,step,arm,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
        for sp in registry():
            budget();sid=sp['state_id'];p=self.d['points'][sid];w=self.d['weights'][sid];a=effective(params,w,arm);point=self.proposed(params,sid,arm);err=self.phase.functional(sid,params,arm,point);m=metric(point,sp['case'],True);parts=loss(m['observed'],m['KL'],point,p['prepared'],a,sp['case']);self.n+=1;self.calls['model_forwards']+=2;self.attempts+=2;self.b+=2
            if self.mode=='interrupt' and step==0 and arm==TRAIN_ARMS[1]:raise RuntimeError('Synthetic local training interruption')
            o=np.array(m['observed']);q=max(abs(o[:6]))/.0018;ties=abs(o[:6])==max(abs(o[:6]));gp=np.zeros(DIM);gp[:6]=2*max(q-1,0)/.0018*np.sign(o[:6])*ties/ties.sum();guard=np.zeros(DIM);signs=2*np.array(safe.target_bits(sp['case'],True)*4)-1;rg=-2*signs*np.maximum(1-signs*o[6:]/.075,0)/.075/16
            for j in range(16):guard[6+j%4]+=rg[j]
            if m['KL']>.0045:guard[:6]+=2*(m['KL']/.0045-1)/.0045*point[:6]/6
            delta=point.astype(float)-p['prepared'].astype(float);norm=np.linalg.norm(delta)
            if norm>20:guard+=2*(norm/20-1)/20*delta/norm
            u=coordinates(p['prepared'],p['raw'],self.d['V']);gr=(self.d['V']@gp)*u;gg=(self.d['V']@guard)*u+2*.001*a/len(a)
            r=dict(**sp,arm=arm,observed=m['observed'],KL=m['KL'],objective_parts=parts,effective_response_gradient=gr.tolist(),effective_guard_gradient=gg.tolist(),response_gradient=lift_gradient(gr,w,arm).tolist(),guard_gradient=lift_gradient(gg,w,arm).tolist(),routing_weights=w.tolist(),parameters_digest=row_digest(params),effective_control_digest=row_digest(a),functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err);rows.append(r)
        self.batches+=1;g,agg=aggregate(rows,params.shape)
        if self.mode=='missing_row':rows=rows[:-1]
        if self.mode=='wrong_pullback':rows[0]['response_gradient']=np.ones(params.shape).tolist()
        if self.mode=='mutate_router':self.d['router']['mean'][0]+=.1
        if self.mode=='nonfinite':g.flat[0]=np.nan
        dump(out/'GRADIENT_PROGRESS.json',rows);dump(out/'AGGREGATION.json',agg);return g,rows
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_zero_calls=self.z,functional_warm_calls=self.w,full84_gradient_batches=self.batches,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,maximum_point_parity_error_Linf=0.,torch_optimizer_objects=0,GPT2_optimizer_updates=0)
