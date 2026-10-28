"""Full-coordinate analytic training fixtures; not model/AD/generalisation evidence."""
import contextlib,copy,io,shutil
from core143 import *
from fixture140 import data140
from fixture138 import capture,metric
from fixture133 import pack

def worker_data():
    d=data140();d['initial_a']=d['initial_a']-.002
    state=dict(d,V=d['V'],historical_hashes=set(d['historical_hashes']))
    fresh,_=prior.fresh_coordinates(state)
    for sp in prior.registry():
        sid=sp['state_id'];theta=fresh[sid];raw,full=safe.reconstruct_raw(d['writer'],d['control']['delta_sketch'],theta,capture(theta,int(sp['case'][-2])^int(sp['case'][-1])));d['points'][sid]=dict(prepared=theta,raw=raw,features=full)
    refs={}
    for sp in registry():
        sid=sp['state_id'];p=d['points'][sid];warm=proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])[0]
        for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',warm,True)]:refs[(sid,kind)]=[metric(point,sp['case'],write)]*2
    d.update(native_refs=refs,C142_summary=dict(decision='C142_READONLY_CONFIRMATION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'),C141_parent_decision='C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE');return d

def dummy_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('synthetic C143 '+k).encode());paths[k]=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c

def emit_bindings(out,d,paths):
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C143=True) for k,p in paths.items()});dump(out/'ALL84_TRAINING_REGISTRY_FROZEN.json',[dict(r,prepared_sha256=safe.tensor_hash(d['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(d['points'][r['state_id']]['raw'])) for r in registry()]);dump(out/'MODEL_PINS_REFERENCE.json',d['modelpins']);dump(out/'FROZEN_INPUT_IDENTITIES.json',frozen_digest(d));dump(out/'C142_BOUND_RECEIPT.json',dict(synthetic_fixture=True,parent_decision_reclassified=False))
    np.savez_compressed(out/'ALL84_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')});np.savez_compressed(out/'FROZEN_SHARED_ROW_BASIS.npz',V=d['V']);np.savez_compressed(out/'INITIAL_C140_STEP12_CONTROL.npz',a=d['initial_a'],m=np.zeros_like(d['initial_a']),v=np.zeros_like(d['initial_a']));np.savez_compressed(out/'FROZEN_ORIGINAL_WRITER.npz',**d['writer']);np.savez_compressed(out/'FROZEN_ORIGINAL_CONTROL.npz',**d['control'])

def loader(paths,out,c):
    d=worker_data();emit_bindings(out,d,paths);return d

# Native and analytic backends operate on one declared synthetic function.
class Toy:
    def __init__(self,out,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.pins=data['modelpins'];calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
    def metrics(self,point,case,write=True):
        self.phase.consume(point,case,write);prefix='preflight' if self.phase.phase=='preflight' else 'path';self.calls[prefix+'_metric_attempts']+=1;self.calls[prefix+'_metrics']+=1;self.calls['model_forwards']+=2;m=metric(point,case,write)
        # Deliberately nonlinear synthetic fault cases, no real-model claim.
        change=(self.mode=='preflight_drift' and self.phase.phase=='preflight') or (self.mode=='repeat_drift' and self.phase.phase=='paths' and self.calls['path_metrics']>252)
        if self.mode=='failed_final' and self.phase.phase=='paths' and self.calls['path_metrics'] in (3,255):m['observed'][0]=.003;change=False;m['response_drift']=.003;m['gates']['response']=False;m['pass']=False
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
            rows.append(dict(state_id=sid,case=sp['case'],C143_membership='training',observed=met['observed'],KL=met['KL'],functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err,coefficient_digest=row_digest(a),objective_parts=parts,objective=sum(parts.values()),state_gradient=gg.tolist()))
        self.batches+=1;g,agg=aggregate([r['objective'] for r in rows],[r['state_gradient'] for r in rows])
        for r,w in zip(rows,agg['weights']):r.update(aggregate_weight=w,gradient_contribution=(w*np.array(r['state_gradient'])).tolist())
        if self.mode=='missing_state':rows=rows[:-1]
        if self.mode=='wrong_weight':rows[0]['aggregate_weight']+=.1
        if self.mode=='wrong_digest':rows[0]['coefficient_digest']='0'*64
        if self.mode=='nonfinite':g[0]=np.nan
        if self.mode=='mutate_basis':self.data['V'][0,0]+=.01
        dump(out/'GRADIENT_PROGRESS.json',rows);dump(out/'AGGREGATION.json',agg);return g,rows
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_zero_calls=self.z,functional_warm_calls=self.w,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,full84_gradient_batches=self.batches,maximum_point_parity_error_Linf=0.,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)


def archive_inputs(root):
    from fixture142 import make_fixture
    from worker142 import run as run142
    root=Path(root);paths,c=make_fixture(root/'originals');folder=root/'originals'
    # Add the C140 model-pin record required by the real training runtime binding.
    dump(folder/'p140/MODEL_PINS_REFERENCE.json',dict(revision='SYNTHETIC_LINEAR_NOT_GPT2',device='cpu',dtype='float32',torch='fixture'))
    pack(folder/'p140',paths['C140']);c['pins']['C140']['sha256']=sha(paths['C140'])
    link=clean_json((folder/'p141/INPUTS_FROZEN.json').read_bytes());link['C140']['sha256']=sha(paths['C140']);dump(folder/'p141/INPUTS_FROZEN.json',link)
    seals=clean_json((folder/'p141/PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json').read_bytes());seals['artifact_seals']['INPUTS_FROZEN.json']=sha(folder/'p141/INPUTS_FROZEN.json');dump(folder/'p141/PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json',seals);pack(folder/'p141',paths['C141']);c['pins']['C141']['sha256']=sha(paths['C141'])
    out=root/'C142'
    with contextlib.redirect_stdout(io.StringIO()):s=run142(paths,out,c)
    need(s['complete'],'Synthetic parent audit must complete');shutil.copytree(ROOT/'vendor142',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z=pack(out,root/'C142_SYNTHETIC.zip')
    paths=dict(paths,C142=z);c143=contract();c143.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c143
