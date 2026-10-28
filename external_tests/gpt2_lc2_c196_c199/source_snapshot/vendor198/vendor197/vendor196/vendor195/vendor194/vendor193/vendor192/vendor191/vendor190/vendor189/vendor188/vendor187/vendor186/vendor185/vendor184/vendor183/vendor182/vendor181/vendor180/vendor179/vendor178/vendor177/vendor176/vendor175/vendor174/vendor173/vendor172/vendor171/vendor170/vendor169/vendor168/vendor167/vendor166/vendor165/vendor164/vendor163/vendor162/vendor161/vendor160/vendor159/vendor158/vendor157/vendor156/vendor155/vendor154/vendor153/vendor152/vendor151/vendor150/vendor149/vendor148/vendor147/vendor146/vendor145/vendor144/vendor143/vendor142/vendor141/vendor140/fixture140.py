"""Full-dimension analytic fixtures; never real GPT-2 or AD evidence."""
import contextlib,copy,io,shutil
from core140 import *
from fixture139 import data139,Toy as Toy139,archive138
from fixture138 import capture,metric
from worker139 import run as run139
from inputs139 import audit138
from fixture133 import pack

def dummy_inputs(root):
    paths={};pins={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('synthetic C140 input '+k).encode());paths[k]=p;pins[k]=dict(path=str(p),sha256=sha(p))
    return paths,dict(contract(),production=False,pins=pins)

def data140():
    d=data139();kernel,_=previous.freeze_kernel_coordinates(d)
    for sp in previous.registry():
        sid=sp['state_id'];theta=kernel[sid];raw,ff=safe.reconstruct_raw(d['writer'],d['control']['delta_sketch'],theta,capture(theta,int(sp['case'][-2])^int(sp['case'][-1])));d['points'][sid]=dict(spec=sp,prepared=theta,raw=raw,features=ff,preparation=metric(theta,sp['case'],False))
    d['initial_a']=d['a'].copy();refs={}
    for sp in registry():
        sid=sp['state_id'];p=d['points'][sid];warm=proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])[0]
        for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',warm,True)):refs[(sid,kind)]=[metric(point,sp['case'],write)]*2
    d.update(native_refs=refs,C139_summary=dict(decision='C139_EXPOSED_DIRECTION_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED'));return d

def loader(paths,out,c):
    d=data140();dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});dump(out/'ALL36_TRAINING_REGISTRY_FROZEN.json',registry());dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'C139_BOUND_RECEIPT.json',dict(synthetic_fixture=True));dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(d))
    np.savez_compressed(out/'ALL36_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')});np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=d['V']);np.savez_compressed(out/'INITIAL_C137_STEP24_CONTROL.npz',a=d['initial_a'],m=np.zeros_like(d['initial_a']),v=np.zeros_like(d['initial_a']));np.savez_compressed(out/'FROZEN_ORIGINAL_WRITER.npz',**d['writer']);np.savez_compressed(out/'FROZEN_ORIGINAL_CONTROL.npz',**d['control']);return d

class Toy:
    def __init__(self,out,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.pins=data['modelpins'];calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
    def metrics(self,point,case,write=True):
        self.phase.consume(point,case,write);prefix='preflight' if self.phase.phase=='preflight' else 'path';self.calls[prefix+'_metric_attempts']+=1;self.calls[prefix+'_metrics']+=1;self.calls['model_forwards']+=2;m=metric(point,case,write)
        # Deliberately nonlinear synthetic fault cases, no real-model claim.
        change=(self.mode=='preflight_drift' and self.phase.phase=='preflight') or (self.mode=='repeat_drift' and self.phase.phase=='paths' and self.calls['path_metrics']>108)
        if self.mode=='failed_final' and self.phase.phase=='paths' and self.calls['path_metrics'] in (3,111):m['observed'][0]=.003;change=False;m['response_drift']=.003;m['gates']['response']=False;m['pass']=False
        if change:m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        return m
    def invariant(self):pass

class Analytic:
    def __init__(self,rt,data,phase,calls,mode='normal'):self.rt=rt;self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.n=0;self.z=0;self.w=0;self.b=0;self.attempts=0;self.batches=0
    def proposed(self,a,sid):p=self.data['points'][sid];return proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
    def zero(self,sp,a):
        point=self.proposed(a,sp['state_id']);err=self.phase.functional(sp['state_id'],a,point);m=metric(point,sp['case'],True);self.calls['model_forwards']+=2;self.n+=1
        if self.phase.phase=='functional_warm':self.w+=1
        else:self.z+=1
        if self.mode=='zero_drift' and self.phase.phase=='functional_zero':m['observed'][0]+=.001
        if self.mode=='warm_drift' and self.phase.phase=='functional_warm':m['observed'][0]+=.001
        return dict(state_id=sp['state_id'],observed=m['observed'],KL=m['KL'],point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err)
    def gradient(self,a,step,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
        for sp in registry():
            budget();sid=sp['state_id'];p=self.data['points'][sid];point=self.proposed(a,sid);err=self.phase.functional(sid,a,point);met=metric(point,sp['case'],True);parts=objective(met['observed'],met['KL'],point,p['prepared'],a,sp['case']);self.attempts+=1
            if self.mode=='interrupt' and step==1:raise RuntimeError('Synthetic interrupted gradient batch')
            self.b+=1;self.n+=1;self.calls['model_forwards']+=2;o=np.array(met['observed']);gp=np.zeros(DIM);excess=np.maximum(abs(o[:6])/RESPONSE_SAFETY-1,0);sq=excess**2;dr=2*excess*np.sign(o[:6])/RESPONSE_SAFETY;ties=sq==max(sq);gp[:6]=dr/6+dr*ties/max(1,sum(ties))
            signs=2*np.array(safe.target_bits(sp['case'],True)*4)-1;readgrad=-2*signs*np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)/MARGIN_SAFETY/16
            for i in range(16):gp[6+i%4]+=readgrad[i]
            if met['KL']>KL_SAFETY:gp[:6]+=2*(met['KL']/KL_SAFETY-1)/KL_SAFETY*point[:6]/6
            delta=point.astype(float)-p['prepared'].astype(float);nn=float(np.linalg.norm(delta))
            if nn>RADIUS:gp+=2*(nn/RADIUS-1)/RADIUS*delta/nn
            v=self.data['V'];zz=v@(p['raw'].astype(float)-p['prepared'].astype(float));gg=(v@gp)*zz+2*L2*a/len(a)
            rows.append(dict(state_id=sid,case=sp['case'],C140_membership='training',observed=met['observed'],KL=met['KL'],functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err,coefficient_digest=row_digest(a),objective_parts=parts,objective=sum(parts.values()),state_gradient=gg.tolist()))
        self.batches+=1;g,agg=aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows])
        for r,w in zip(rows,agg['weights']):r.update(aggregate_weight=w,gradient_contribution=(w*np.array(r['state_gradient'])).tolist())
        if self.mode=='missing_state':rows=rows[:-1]
        if self.mode=='wrong_weight':rows[0]['aggregate_weight']+=.1
        if self.mode=='wrong_digest':rows[0]['coefficient_digest']='0'*64
        if self.mode=='nonfinite':g[0]=np.nan
        if self.mode=='mutate_basis':self.data['V'][0,0]+=.01
        dump(out/'GRADIENT_PROGRESS.json',rows);dump(out/'AGGREGATION.json',agg);return g,rows
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_zero_calls=self.z,functional_warm_calls=self.w,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,full36_gradient_batches=self.batches,maximum_point_parity_error_Linf=0.,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def archive139(root):
    root.mkdir();z138,d,c6=archive138(root/'parent138');c=dict(contract(),production=False,pins=dict(c6['pins'],C138=dict(path=str(z138),sha256=sha(z138))));paths={k:Path(v['path']) for k,v in c['pins'].items()}
    with contextlib.closing(Archive(z138)) as z:audit138(z,d,root,c)
    initial=copy.deepcopy(d);out=root/'C139'
    def oldloader(paths,out,c):
        dd=copy.deepcopy(initial);dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C139=True) for k in previous.INPUT_KEYS});dump(out/'C138_PREPARATION_AND_ENDPOINT_AUDIT.json',dd['C138_audit']);return dd
    with contextlib.redirect_stdout(io.StringIO()):s=run139(paths,out,c,Toy139,oldloader)
    need(s['complete'],'Synthetic C139 fixture incomplete');shutil.copytree(ROOT/'vendor139',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z139=pack(out,root/'C139_SYNTHETIC.zip');c['pins']['C139']=dict(path=str(z139),sha256=sha(z139));return z139,initial,c
