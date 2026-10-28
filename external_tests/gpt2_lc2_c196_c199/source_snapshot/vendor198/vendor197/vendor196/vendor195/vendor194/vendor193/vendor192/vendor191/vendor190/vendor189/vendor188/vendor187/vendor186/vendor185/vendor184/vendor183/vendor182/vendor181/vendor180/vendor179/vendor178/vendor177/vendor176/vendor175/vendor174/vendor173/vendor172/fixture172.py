"""NumPy-only synthetic context/direction; not a model or real original audit."""
import copy,contextlib,io,shutil
from core172 import *
from fixture171 import Analytic,synthetic_solver,pack,observed
def context_fixture(root):
 root=Path(root);root.mkdir(parents=True);rank=6;p=np.zeros((3,2,rank));V=np.zeros((rank,DIM));V[:,:rank]=np.eye(rank);data=dict(V=V,points={},weights={})
 for i,sid in enumerate(IDS):
  raw=np.zeros(DIM,np.float32);raw[:6]=.0022 if i<5 else .0019 if i<9 else .001;raw[6]=i*1e-5;data['points'][sid]=dict(prepared=np.zeros(DIM,np.float32),raw=raw);data['weights'][sid]=np.array([.2,.3,.5])
 ctx=dict(data=dict(d=data,before=p,after=p,coefs={}),initial_head=p,C170_final_head=p.copy(),baseline=dict(FP32_cached_response_reference=[0.]*6,FP64_response_reference=[0.]*6),baseline_logits=dict(FP32_logits_cast64=np.zeros((12,8)),FP64_logits=np.zeros((12,8))),modelpins=dict(response_ids=list(range(6))),C171_receipt=dict(decision=prior.DECISION,synthetic_context_injected=True));chain=control.source168.old.parent.parent.previous.constants(ctx['data']);native=[];frames=[]
 for sid in IDS:
  pt,e=mapping(chain,p,sid);o=observed(pt,sid);met=archive155.full_metric(observed(pt.astype(np.float32),sid),0.,pt.astype(np.float32),sid.split('__')[0]);qual=archive155.endpoint_gates(met,sid.split('__')[0],pt.astype(np.float32),chain[sid+'__prepared'].astype(np.float32),True);native.append(dict(state_id=sid,case=sid.split('__')[0],metrics=met,qualification=qual));frames.append(dict(state_id=sid,observed=o.tolist(),KL=0.))
 ctx.update(C170_final_FP64=frames,C170_final_native={'0':copy.deepcopy(native),'1':copy.deepcopy(native)});paths={}
 for k in INPUT_KEYS:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream context injection only');paths[k]=q
 c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx
