"""NumPy-only synthetic whole protocol; no model or numerical LP in fixture."""
import copy
from core174 import *
from fixture172 import context_fixture as old_context
from fixture171 import Analytic as Base,synthetic_solver,pack
class Analytic(Base):
 def __init__(self,*a):
  super().__init__(*a);f=self.out/'REFERENCES_FROZEN_BEFORE_FOCUS.json';r=clean_json(f.read_bytes());r['protocol']=PROTOCOL;dump(f,r)
 def coefficient_record(self,sid,p,expected,gradient=False):
  row=super().coefficient_record(sid,p,expected,gradient)
  if gradient:
   u=self.chain['V']@(self.chain[sid+'__raw']-self.chain[sid+'__prepared']);w=self.chain[sid+'__weights'];J=[]
   for j in range(6):ge=np.stack([self.chain['V'][:,j]*u,self.chain['V'][:,j]*np.roll(u,1)]);J.append(w[:,None,None]*ge[None])
   row['response_coordinate_gradients']=np.stack(J).tolist();self.calls['backward_calls']+=6;self.calls['backward_attempts']+=6;self.calls['response_coordinate_backward_calls']+=6;self.calls['response_Jacobian_blocks_completed']+=1
  return row
def fixture(root):
 root=Path(root);paths,_,ctx=old_context(root);ctx.update(C173_receipt=dict(decision=prior.DECISION,synthetic_upstream_injected=True),C172_anchor_FP64=ctx['C170_final_FP64'],C172_anchor_native=ctx['C170_final_native'])
 for k in INPUT_KEYS:
  if k not in paths:q=root/(k+'.zip');q.write_bytes(b'SYNTHETIC upstream marker only');paths[k]=q
 c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});return paths,c,ctx
