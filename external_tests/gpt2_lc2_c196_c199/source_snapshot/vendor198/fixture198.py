"""Synthetic observables only; real NumPy writer map, no GPT-2 execution."""
import copy
from core198 import *
def fixture(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);rank=6;V=np.zeros((rank,DIM));V[:,:rank]=np.eye(rank);sketch=np.zeros((DIM,64),np.float32);sketch[:64]=np.eye(64,dtype=np.float32)
 delta=np.zeros(DIM);delta[:6]=.001;coeff=np.zeros((1,DIM));coeff[0,:6]=.01
 d=dict(V=V,writer=dict(mean=np.zeros(3147),scale=np.ones(3147),delta_mean=delta,training_z=np.eye(1,3147),coefficients=coeff),control=dict(delta_sketch=sketch),router=dict(mean=np.zeros(rank),scale=np.ones(rank),centers=np.zeros((3,rank))),points={},weights={})
 head=np.full((3,2,rank),.01);ctx=dict(data=dict(d=d),C196_final_head=head,baseline=dict(FP32_cached_response_reference=[0.]*6),baseline_logits=dict(FP32_logits_cast64=np.zeros((12,8))),C196_final_points={},C196_final_native={'0':[],'1':[]},modelpins={})
 for sid in IDS:
  prepared=np.zeros(DIM,np.float32);capture=np.zeros(3147);case=sid.split('__')[0];capture[3076]=int(case[-2])^int(case[-1]);r=fresh_proposal(d,head,prepared,capture);d['points'][sid]=dict(prepared=prepared,raw=r['raw'],features=r['full']);d['weights'][sid]=r['weights'];ctx['C196_final_points'][sid]=r['point']
  met=observed_metric(sid,r['point'],True)
  for rnd in ('0','1'):ctx['C196_final_native'][rnd].append(dict(state_id=sid,metrics=copy.deepcopy(met)))
 paths={}
 for k in INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(('SYNTHETIC marker '+k).encode());paths[k]=p
 c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c,ctx
def observed_metric(sid,point,write,fail=False):
 case=sid.split('__')[0];x,y=int(case[-2]),int(case[-1]);truth=[x,y,1,x^y if write else 1-(x^y)];o=np.r_[np.full(6,.003 if fail else .001),(2*np.tile(truth,4)-1)*.2];m=archive155.full_metric(o,0.,point,case)
 if not write:
  m['bits']=truth;m['exact_margins']=[.2]*4;m['development_margins']=[.2]*12;m['gates']=dict(bits_match=True,response=not fail,KL=True,exact_read_margin=True,development_read_margin=True);m['pass']=not fail
 return m
class Fake:
 def __init__(self,out,ctx,calls,mode='normal'):
  self.ctx=ctx;self.calls=calls;self.mode=mode;self.calls['model_forwards']+=1;self.calls['initialization_forwards']=1;self.verify_calls=0
 def features(self,theta,desired):
  i=self.calls['feature_calls'];rnd=i//84;self.calls['feature_attempts']+=1;self.calls['feature_calls']+=1;self.calls['model_forwards']+=3;x=np.zeros(3147);x[3076]=desired
  if self.mode=='fresh_shift' or self.mode=='repeat_shift' and rnd==1:x[0]=1.
  if self.mode=='mutate_head' and i==0:self.ctx['C196_final_head'][0,0,0]+=.001
  sid=IDS[i%84];return x,np.asarray(observed_metric(sid,theta,False)['observed'])
 def metrics(self,point,case,write=True):
  i=self.calls['metric_calls'];sid=next(s for s in IDS if s.split('__')[0]==case) # replaced below by fixed request-order index
  if i<168:sid=IDS[i%84]
  else:sid=IDS[(i-168)%84]
  self.calls['metric_attempts']+=1;self.calls['metric_calls']+=1;self.calls['model_forwards']+=2;fail=bool(sid==IDS[0] and ((self.mode=='failed_endpoint' and write) or (self.mode=='failed_preparation' and not write)))
  return observed_metric(sid,point,write,fail)
 def verify(self,rehash=False):self.verify_calls+=1
 def close(self):self.verify(True)
