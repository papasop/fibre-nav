"""Synthetic nonzero C180-final continuation anchor; upstream explicitly injected."""
import copy
from core182 import *
from fixture181 import fixture as upstream
from fixture180 import Analytic as Base,synthetic_solver
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);f=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';r=clean_json(f.read_bytes());r['protocol']=PROTOCOL;dump(f,r)
def adopt(ctx,folder):
 folder=Path(folder)
 with np.load(folder/'FIXED_FINAL_HEAD_STEP8.npz',allow_pickle=False) as z:p=z['P'].copy()
 frames=clean_json((folder/'FINAL_FP64_RECORDS_ALL84.json').read_bytes())['final_FP64'];native=clean_json((folder/'ALL336_NATIVE_ENDPOINTS.json').read_bytes())
 ctx.update(C181_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True),C180_final_head=p,C180_final_FP64=frames,C180_final_native={str(t):native['native_final'+str(t)] for t in (0,1)})
 return ctx
def fixture(root):
 root=Path(root);paths,_,ctx,src=upstream(root/'upstream');adopt(ctx,src);need(np.any(ctx['C180_final_head']!=ctx['C178_final_head']),'C180 continuation anchor must differ from C178')
 for k in INPUT_KEYS:
  if k not in paths:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream context only, no original science');paths[k]=q
 c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx
