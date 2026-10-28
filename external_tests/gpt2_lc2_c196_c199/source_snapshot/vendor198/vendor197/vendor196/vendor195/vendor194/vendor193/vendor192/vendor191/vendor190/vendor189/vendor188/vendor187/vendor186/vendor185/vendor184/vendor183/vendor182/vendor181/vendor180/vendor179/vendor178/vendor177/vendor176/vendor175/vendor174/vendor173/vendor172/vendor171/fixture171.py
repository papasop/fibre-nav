"""Pure NumPy synthetic C170 schema fixture, not real model/solver evidence."""
import copy,contextlib,io,shutil
from core171 import *
from fixture169 import Analytic as Base,pack,observed
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);r=clean_json((self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json').read_bytes());r['protocol']=prior.PROTOCOL;dump(self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json',r)
 def preseal(self,p,points,folder,tag):
  super().preseal(p,points,folder,tag);folder=Path(folder);r=clean_json((folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json').read_bytes());r['model_forwards_so_far']=self.calls['model_forwards'];r['backward_calls_so_far']=self.calls['backward_calls'];dump(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',r);return dict(folder=folder.name,**r,receipt_sha256=sha(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'))
 def coefficient_record(self,sid,p,expected,gradient=False):
  need(sid==IDS[len(self.batch['seen'])],'Synthetic batch ordering');self.batch['seen'].append(sid);pt=self.batch['points'][sid];exact(pt,expected,'Synthetic point');_,e=mapping(self.chain,p,sid);o=observed(pt,sid);self.calls['model_forwards']+=2;self.calls['functional_FP64_forwards']+=2;r=dict(state_id=sid,case=sid.split('__')[0],parameters_digest=parameter_hash(p),FP64_point_sha256=point_hash(pt),sealed_numpy_point_sha256=point_hash(expected),actual_FP64_point_artifact=self.batch['folder'].name+'/ACTUAL_TORCH_FP64_POINTS.npz',map_Linf=0.,observed=o.tolist(),KL=0.,objective_parts=losses(o,0.,pt,self.chain[sid+'__prepared'],e,sid.split('__')[0]),FP64_acceptance_evaluated=False)
  if gradient:
   self.calls['backward_calls']+=2;self.calls['backward_attempts']+=2;u=self.chain['V']@(self.chain[sid+'__raw']-self.chain[sid+'__prepared']);a=o[:6];top=abs(a)==max(abs(a));factor=2*max(max(abs(a))/.0018-1,0)/.0018;gradpoint=factor*np.sign(a)*top/top.sum();ge=np.stack([gradpoint*u,gradpoint*np.roll(u,1)]);w=self.chain[sid+'__weights'];r['response_gradient']=(w[:,None,None]*ge[None]).tolist();r['guard_gradient']=(w[:,None,None]*(2*.001*e/e.shape[-1])[None]).tolist()
  return r
class Result(dict):
 def __getattr__(self,k):return self[k]
def synthetic_solver(cost,**kw):
 m=len(cost)-1;return Result(x=np.r_[-np.ones(m),1.],fun=-1.,success=True,status=0,message='Synthetic prescribed vector only; no LP actually solved')
def fixture(root):
 from worker170 import run
 root=Path(root);root.mkdir(parents=True);rank=6;p=np.zeros((3,2,rank));V=np.zeros((rank,DIM));V[:,:rank]=np.eye(rank);data=dict(V=V,points={},weights={})
 for i,sid in enumerate(IDS):
  raw=np.zeros(DIM,np.float32);raw[:6]=.0022 if i<5 else .0019 if i<9 else .001;raw[6]=i*1e-5;data['points'][sid]=dict(prepared=np.zeros(DIM,np.float32),raw=raw);data['weights'][sid]=np.array([.2,.3,.5])
 ctx=dict(data=dict(d=data,before=p,after=p,coefs={}),initial_head=p,baseline=dict(FP32_cached_response_reference=[0.]*6,FP64_response_reference=[0.]*6),baseline_logits=dict(FP32_logits_cast64=np.zeros((12,8)),FP64_logits=np.zeros((12,8))),modelpins=dict(response_ids=list(range(6))),C169_receipt=dict(decision=prior.prior.DECISION,synthetic_context_injected=True));chain=prior.source168.old.parent.parent.previous.constants(ctx['data']);native=[];frames=[]
 for sid in IDS:
  pt,e=mapping(chain,p,sid);o=observed(pt,sid);met=archive155.full_metric(observed(pt.astype(np.float32),sid),0.,pt.astype(np.float32),sid.split('__')[0]);qual=archive155.endpoint_gates(met,sid.split('__')[0],pt.astype(np.float32),chain[sid+'__prepared'].astype(np.float32),True);native.append(dict(state_id=sid,case=sid.split('__')[0],metrics=met,qualification=qual));frames.append(dict(state_id=sid,observed=o.tolist(),KL=0.))
 ctx.update(C168_FP64_anchor_records=frames,C168_native_anchor_records={'0':copy.deepcopy(native),'1':copy.deepcopy(native)});paths={}
 for k in prior.INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream marker only');paths[k]=q
 cp=prior.contract();cp.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()})
 with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'C170',cp,Analytic,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic C170 schema generation: '+str(s['errors']));shutil.copytree(ROOT/'vendor170',root/'C170/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C170']=pack(root/'C170',root/'C170.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C170'
