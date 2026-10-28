"""Nonzero C176-final continuation fixture, injected upstream, NumPy only."""
import copy,contextlib,io
from core178 import *
from fixture176 import fixture as upstream,Analytic as Base,synthetic_solver
from worker176 import run as run176
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);f=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';r=clean_json(f.read_bytes());r['protocol']=PROTOCOL;dump(f,r)

def adopt(ctx,folder):
 folder=Path(folder)
 with np.load(folder/'FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:p=z['P'].copy()
 frames=clean_json((folder/'FINAL_FP64_RECORDS_ALL84.json').read_bytes())['final_FP64'];native=clean_json((folder/'ALL336_NATIVE_ENDPOINTS.json').read_bytes())
 ctx.update(C177_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True),C176_final_head=p,C176_final_FP64=frames,C176_final_native={str(t):native['native_final'+str(t)] for t in (0,1)})
 return ctx

def fixture(root):
 root=Path(root);root.mkdir(parents=True);paths,cp,ctx=upstream(root/'upstream')
 with contextlib.redirect_stdout(io.StringIO()):s=run176(paths,root/'C176',cp,Base,lambda *a:copy.deepcopy(ctx),synthetic_solver)
 need(s['complete'],'Synthetic prior C176: '+str(s['errors']));adopt(ctx,root/'C176')
 need(np.any(ctx['C176_final_head']!=ctx['C170_final_head']),'Continuation fixture must not alias older C170 head')
 for k in INPUT_KEYS:
  if k not in paths:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream injected, not original science');paths[k]=q
 c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx
