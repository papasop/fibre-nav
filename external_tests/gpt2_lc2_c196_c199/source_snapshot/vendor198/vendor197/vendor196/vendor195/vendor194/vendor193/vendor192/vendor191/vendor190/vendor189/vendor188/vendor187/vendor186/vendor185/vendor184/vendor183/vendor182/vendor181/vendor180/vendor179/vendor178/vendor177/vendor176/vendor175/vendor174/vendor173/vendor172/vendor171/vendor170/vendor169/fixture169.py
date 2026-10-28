"""NumPy-only synthetic C168 schema fixture; no real model or historical evidence."""
import copy,shutil,zipfile
from core169 import *
from baseline158 import record
def pack(root,path):
 root=Path(root);dump(root/'RESULT_MANIFEST.json',{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json'})
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(root).as_posix())
 return path
def observed(point,sid):
 case=sid.split('__')[0];x,y=int(case[-2]),int(case[-1]);sign=2*np.tile([x,y,1,x^y],4)-1;return np.r_[np.asarray(point,float)[:6],.2*sign]
class Analytic:
 def __init__(self,out,ctx,chain,points,native,coefs,calls):
  self.out=Path(out);self.out.mkdir(parents=True);self.chain=chain;self.calls=calls;self.registered={};self.batch=None;self.point_tensors_saved=0;calls['model_forwards']+=3;calls['functional_FP64_forwards']+=2
  logits=ctx['baseline_logits']['FP32_logits_cast64'];zero=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float)
  for name,a in [('NATIVE_FP32_BASELINE_LOGITS',logits),('NATIVE_FP32_CACHED_RESPONSE',zero),('FUNCTIONAL_FP64_BASELINE_LOGITS',logits),('WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',logits)]:
   with contextlib.redirect_stdout(__import__('io').StringIO()):record(self.out,name,a,a,'within_run_repeat' if name.startswith('WITHIN') else 'cross_run_descriptive_only')
  refs=dict(current_FP64=zero,archived_FP64=zero,archived_FP32=zero);dump(self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json',dict(protocol=prior.PROTOCOL,model_forwards_so_far=3,state_probes_so_far=0,references={k:v.tolist() for k,v in refs.items()},reference_tensor_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in refs.items()}))
 def preseal(self,p,points,folder,tag):
  folder=Path(folder);arrays={sid:points[(sid,'zero')].copy() for sid in IDS};np.savez_compressed(folder/'ACTUAL_TORCH_FP64_POINTS.npz',**arrays);bindings=[dict(state_id=sid,actual_Torch_FP64_sha256=point_hash(arrays[sid]),NumPy_point_sha256=point_hash(points[(sid,'zero')]),Torch_vs_NumPy_Linf=0.) for sid in IDS];dump(folder/'ACTUAL_TORCH_FP64_POINT_BINDINGS.json',bindings);receipt=dict(tag=tag,head_digest=parameter_hash(p),all84_state_ids=IDS,model_forwards_so_far=3,backward_calls_so_far=0,sealed_before_this_batch=True,artifact_seals={n:sha(folder/n) for n in ('HEAD.npz','POINT_BINDINGS.json','ACTUAL_TORCH_FP64_POINTS.npz','ACTUAL_TORCH_FP64_POINT_BINDINGS.json')});dump(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',receipt);self.registered[folder.name]=dict(folder=folder,points=arrays,head=p.copy(),seen=[]);self.point_tensors_saved+=84;return dict(folder=folder.name,**receipt,receipt_sha256=sha(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'))
 def activate(self,name,p):self.batch=self.registered[name];exact(self.batch['head'],p,'Synthetic batch head')
 def verify_batch(self,p):
  exact(self.batch['head'],p,'Synthetic head identity');compare(self.batch['seen'],IDS,'Synthetic complete batch')
 def coefficient_record(self,sid,p,expected,gradient=False):
  need(gradient is False,'No fixture AD');need(sid==IDS[len(self.batch['seen'])],'Synthetic order');self.batch['seen'].append(sid);pt=self.batch['points'][sid];exact(pt,expected,'Synthetic actual point');_,e=mapping(self.chain,p,sid);o=observed(pt,sid);self.calls['model_forwards']+=2;self.calls['functional_FP64_forwards']+=2
  return dict(state_id=sid,case=sid.split('__')[0],parameters_digest=parameter_hash(p),FP64_point_sha256=point_hash(pt),sealed_numpy_point_sha256=point_hash(expected),actual_FP64_point_artifact=self.batch['folder'].name+'/ACTUAL_TORCH_FP64_POINTS.npz',map_Linf=0.,observed=o.tolist(),KL=0.,objective_parts=losses(o,0.,pt,self.chain[sid+'__prepared'],e,sid.split('__')[0]),FP64_acceptance_evaluated=False)
 def native_candidate(self,sid,pt,digest):
  compare(safe.tensor_hash(pt),digest);o=observed(pt,sid);met=archive155.full_metric(o,0.,pt,sid.split('__')[0]);q=archive155.endpoint_gates(met,sid.split('__')[0],pt,self.chain[sid+'__prepared'].astype(np.float32),True);self.calls['model_forwards']+=2;return dict(state_id=sid,case=sid.split('__')[0],metrics=met,qualification=q,actual_physical_FP32_commit=False,synthetic_fixture=True)
 def verify(self,*a):pass
def fixture(root):
 from worker168 import run
 root=Path(root);root.mkdir(parents=True);rank=6;p=np.zeros((3,2,rank));d=np.ones_like(p);V=np.zeros((rank,DIM));V[:,:rank]=np.eye(rank);data=dict(V=V,points={},weights={})
 for i,sid in enumerate(IDS):
  raw=np.zeros(DIM,np.float32);raw[:6]=(.0022 if i<5 else .0019 if i<9 else .001);raw[6]=i*1e-5;data['points'][sid]=dict(prepared=np.zeros(DIM,np.float32),raw=raw);data['weights'][sid]=np.array([.2,.3,.5])
 ctx=dict(data=dict(d=data,before=p,after=p,coefs={}),initial_head=p,fixed_C167_direction=d,baseline=dict(FP32_cached_response_reference=[0.]*6,FP64_response_reference=[0.]*6),baseline_logits=dict(FP32_logits_cast64=np.zeros((12,8)),FP64_logits=np.zeros((12,8))),modelpins=dict(response_ids=list(range(6))),C166_receipt=dict(decision='C166_READONLY_DIRECTION_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'),C167_receipt=dict(decision='C167_RECORDED_LINEAR_PROBLEM_DIAGNOSTIC_COMPLETE_NO_MODEL_ACCEPTANCE',classification='RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED',synthetic_injected=True))
 chain=prior.old.parent.parent.previous.constants(ctx['data']);grad=[];native=[]
 for sid in IDS:
  pt,e=mapping(chain,p,sid);o=observed(pt,sid);met=archive155.full_metric(observed(pt.astype(np.float32),sid),0.,pt.astype(np.float32),sid.split('__')[0]);q=archive155.endpoint_gates(met,sid.split('__')[0],pt.astype(np.float32),chain[sid+'__prepared'].astype(np.float32),True);native.append(dict(state_id=sid,case=sid.split('__')[0],metrics=met,qualification=q));grad.append(dict(state_id=sid,case=sid.split('__')[0],observed=o.tolist(),KL=0.,objective_parts=losses(o,0.,pt,chain[sid+'__prepared'],e,sid.split('__')[0]),response_gradient=np.ones_like(p).tolist(),guard_gradient=np.zeros_like(p).tolist()))
 ctx.update(C165_gradient_records=grad,C165_FP64_anchor_records=grad,C165_native_anchor_records={'0':copy.deepcopy(native),'1':copy.deepcopy(native)});paths={}
 for k in prior.INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC INJECTED UPSTREAM ONLY');paths[k]=q
 cp=prior.contract();cp.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()})
 with contextlib.redirect_stdout(__import__('io').StringIO()):s=run(paths,root/'C168',cp,Analytic,lambda *a:copy.deepcopy(ctx))
 need(s['complete'],'Synthetic C168 worker complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor168',root/'C168/source',ignore=shutil.ignore_patterns('__pycache__'));paths['C168']=pack(root/'C168',root/'C168.zip');c=contract();c.update(production=False,parent=cp,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx,root/'C168'
