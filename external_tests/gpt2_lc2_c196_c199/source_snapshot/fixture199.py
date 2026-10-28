"""Synthetic native observations; actual fixed NumPy generation/raw/router maps."""
import copy
from core199 import *
from fixture198 import fixture as base_fixture,observed_metric
def fixture(root):
 root=Path(root);paths,_,ctx=base_fixture(root/'upstream');ctx.update(C198_receipt=dict(decision='SYNTHETIC'),C198_fresh_points={str(j):copy.deepcopy(ctx['C196_final_points']) for j in (0,1)},C198_fresh_native=copy.deepcopy(ctx['C196_final_native']))
 p=root/'C198.zip';p.write_bytes(b'SYNTHETIC parent marker only');paths['C198']=p;c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c,ctx
def history(paths,ctx,out,budget=lambda:None):
 h=dict(state_ids=set(IDS),seeds=set(),hashes={safe.tensor_hash(q[name]) for q in ctx['data']['d']['points'].values() for name in ('prepared','raw')});dump(Path(out)/'BOUND_HISTORY_NOVELTY_INVENTORY.json',dict(synthetic_fixture=True,programme_wide_nonaccess_proved=False));return h
class Fake:
 def __init__(self,out,ctx,calls,mode='normal'):
  self.ctx=ctx;self.calls=calls;self.mode=mode;calls['model_forwards']=1;calls['initialization_forwards']=1;self.newids=[r['state_id'] for r in registry()]
 def features(self,theta,desired):
  i=self.calls['feature_calls'];sid=IDS[i] if i<84 else self.newids[(i-84)%48];x=np.zeros(3147);x[3076]=desired
  if self.mode=='new_repeat_shift' and i>=132:x[0]=1.
  if self.mode=='mutate_head' and i==84:self.ctx['C196_final_head'][0,0,0]+=.001
  self.calls['feature_attempts']+=1;self.calls['feature_calls']+=1;self.calls['model_forwards']+=3;return x,np.asarray(observed_metric(sid,theta,False)['observed'])
 def metrics(self,point,case,write=True):
  i=self.calls['metric_calls'];sid=IDS[i%84] if i<252 else self.newids[(i-252)%48];fail=False
  if self.mode=='known_failure' and i<84 and sid==IDS[0]:fail=True
  if self.mode=='new_endpoint_failure' and i>=348 and sid==self.newids[0]:fail=True
  if self.mode=='new_preparation_failure' and 252<=i<348 and sid==self.newids[0]:fail=True
  self.calls['metric_attempts']+=1;self.calls['metric_calls']+=1;self.calls['model_forwards']+=2;return observed_metric(sid,point,write,fail)
 def verify(self,rehash=False):pass
 def close(self):pass
