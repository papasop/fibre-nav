"""Full NumPy-only synthetic protocol; original contexts/measurements injected."""
import copy
from core176 import *
from fixture174 import fixture as upstream,Analytic as Base,synthetic_solver
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);p=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';r=clean_json(p.read_bytes());r['protocol']=PROTOCOL;dump(p,r)
def fixture(root):
 paths,_,ctx=upstream(root);ctx.update(C175_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True),C174_anchor_FP64=copy.deepcopy(ctx['C172_anchor_FP64']),C174_anchor_native=copy.deepcopy(ctx['C172_anchor_native']))
 for k in INPUT_KEYS:
  if k not in paths:q=Path(root)/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream context only');paths[k]=q
 c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx
