"""Full-coordinate analytic fixtures only; no real-model or AD claims."""
import contextlib,copy,io,shutil
from core146 import *
from fixture145 import data145,archive_inputs as parents145,Toy
from fixture138 import metric
from fixture133 import pack
from inputs146 import emit

def data146():
    d=data145();d['C145_summary']=dict(decision='C145_FIXED_DIRECTIONAL_DIAGNOSTIC_COMPLETE_NO_CANDIDATE_SELECTED');return d
def dummy_inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths={}
    for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('Synthetic C146 original '+k).encode());paths[k]=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
def bindings(out,d,paths):emit(out,d,paths,dict(synthetic_fixture=True,no_probe_selected=True))

class Native(Toy):
    def metrics(self,point,case,write=True):
        m=super().metrics(point,case,write)
        # Parent fixture uses a nine-configuration repeat length; C146 has three arms.
        if self.mode=='repeat146' and self.phase.phase=='paths' and self.calls['path_metrics']>252:
            m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        if self.mode=='failed_final' and self.phase.phase=='paths' and self.calls['path_metrics'] in (3,255):m['observed'][0]=.003;m['response_drift']=.003;m['gates']['response']=False;m['pass']=False
        return m

class Analytic:
    def __init__(self,rt,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.n=0;self.z=0;self.w=0;self.b=0;self.attempts=0;self.batches=0
    def proposed(self,a,sid):p=self.data['points'][sid];return proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
    def parity(self,sp,a):
        sid=sp['state_id'];point=self.proposed(a,sid);err=self.phase.functional(sid,a,point);m=metric(point,sp['case'],True);self.n+=1;self.calls['model_forwards']+=2
        if self.phase.phase=='functional_zero':self.z+=1
        else:self.w+=1
        if (self.mode=='zero_drift' and self.phase.phase=='functional_zero') or (self.mode=='warm_drift' and self.phase.phase=='functional_warm'):m['observed'][0]+=.001
        return dict(state_id=sid,point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err,observed=m['observed'],KL=m['KL'])
    def gradient(self,a,step,out,budget):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
        for sp in registry():
            budget();sid=sp['state_id'];p=self.data['points'][sid];point=self.proposed(a,sid);err=self.phase.functional(sid,a,point);met=metric(point,sp['case'],True);parts=objective(met['observed'],met['KL'],point,p['prepared'],a,sp['case']);self.n+=1;self.calls['model_forwards']+=2;self.attempts+=2
            if self.mode=='interrupt' and step==1:raise RuntimeError('Synthetic interrupted training')
            self.b+=2;o=np.asarray(met['observed']);rpoint=np.zeros(DIM);gpoint=np.zeros(DIM);q=max(abs(o[:6]))/RESPONSE_SAFETY;ties=abs(o[:6])==max(abs(o[:6]));rpoint[:6]=2*max(q-1,0)/RESPONSE_SAFETY*np.sign(o[:6])*ties/ties.sum()
            signs=2*np.array(safe.target_bits(sp['case'],True)*4)-1;rg=-2*signs*np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)/MARGIN_SAFETY/16
            for j in range(16):gpoint[6+j%4]+=rg[j]
            if met['KL']>KL_SAFETY:gpoint[:6]+=2*(met['KL']/KL_SAFETY-1)/KL_SAFETY*point[:6]/6
            delta=point.astype(float)-p['prepared'].astype(float);norm=float(np.linalg.norm(delta))
            if norm>RADIUS:gpoint+=2*(norm/RADIUS-1)/RADIUS*delta/norm
            V=self.data['V'];z=V@(p['raw'].astype(float)-p['prepared'].astype(float));gr=(V@rpoint)*z;gg=(V@gpoint)*z+2*L2*a/len(a)
            rows.append(dict(**sp,observed=met['observed'],KL=met['KL'],objective_parts=parts,response_gradient=gr.tolist(),guard_gradient=gg.tolist(),coefficient_digest=row_digest(a),functional_point_sha256=safe.tensor_hash(point),point_parity_error_Linf=err))
        self.batches+=1;g,agg=aggregate(rows,len(a))
        for r,w,v in zip(rows,agg['response_weights'],agg['state_gradient_contributions']):r.update(response_weight=w,guard_weight=1/N,gradient_contribution=v)
        if self.mode=='missing_state':rows=rows[:-1]
        if self.mode=='wrong_weight':rows[0]['response_weight']+=.1
        if self.mode=='wrong_digest':rows[0]['coefficient_digest']='0'*64
        if self.mode=='nonfinite':g[0]=np.nan
        if self.mode=='mutate_basis':self.data['V'][0,0]+=.01
        dump(out/'GRADIENT_PROGRESS.json',rows);dump(out/'AGGREGATION.json',agg);return g,rows
    def verify(self,rehash=False,check_chart=True):pass
    def stats(self):return dict(functional_endpoint_calls=self.n,functional_zero_calls=self.z,functional_warm_calls=self.w,full84_gradient_batches=self.batches,scalar_backward_attempts=self.attempts,scalar_backward_calls=self.b,maximum_point_parity_error_Linf=0.,torch_optimizer_objects=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0)

def archive_inputs(root):
    from worker145 import run as run145
    from fixture145 import Analytic as Gradient145
    import fixture145
    from inputs145 import load as original_loader
    root=Path(root);paths,c145=parents145(root/'parents');out=root/'C145'
    # The inherited archive fixture deliberately injects a C143 gate failure.
    # Replay those exact recorded native points in this archive-reader fixture;
    # this is not a derivative-accuracy or continuous-model construction.
    reference={};original_metric=fixture145.metric
    def replay_metric(point,case,write):return copy.deepcopy(reference.get((safe.tensor_hash(point),case,write),original_metric(point,case,write)))
    def loader(paths,out,c):
        d=original_loader(paths,out,c)
        for (sid,kind),refs in d['refs'].items():
            for r in refs:reference[(r['theta_sha256'],sid.split('__')[0],kind!='prepared')]=r
        return d
    fixture145.metric=replay_metric
    try:
        with contextlib.redirect_stdout(io.StringIO()):s=run145(paths,out,c145,lambda out,d,ph,calls:Toy(out,d,ph,calls),lambda rt,d,ph,calls:Gradient145(rt,d,ph,calls),loader)
    finally:fixture145.metric=original_metric
    need(s['complete'],str(s['errors']));shutil.copytree(ROOT/'vendor145',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p=pack(out,root/'C145_SYNTHETIC.zip');paths=dict(paths,C145=p)
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},binding145=c145);return paths,c
