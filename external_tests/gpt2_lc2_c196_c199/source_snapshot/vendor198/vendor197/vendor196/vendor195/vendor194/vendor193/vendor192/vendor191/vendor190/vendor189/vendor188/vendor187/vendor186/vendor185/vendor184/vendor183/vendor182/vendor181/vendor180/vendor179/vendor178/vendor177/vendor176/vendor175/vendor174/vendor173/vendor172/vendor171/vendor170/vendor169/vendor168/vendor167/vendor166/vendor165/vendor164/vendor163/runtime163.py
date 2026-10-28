"""Same FP64/physical FP32 runtime, with actual pre-forward FP64 point archives."""
from core163 import *
from runtime161 import Trainer
class SmoothTrainer(Trainer):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);receipt=clean_json(self.reference_file.read_bytes());receipt['protocol']=PROTOCOL;dump(self.reference_file,receipt);self.reference_file_hash=sha(self.reference_file);self.batch=None;self.authorized_point=None;self.point_tensors_saved=0;self.verify(True)
 def map_head(self,p,sid):
  t=self.torch;c=self.chain;w=c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);return c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*t.roll(u,1)),e
 def prepare_batch(self,p,points,folder,tag):
  need(self.authorized_point is None,'No active model call while sealing')
  if self.batch is not None:compare(self.batch['seen'],IDS,'Previous complete84 batch before new head')
  folder=Path(folder);need(not (folder/'ACTUAL_TORCH_FP64_POINTS.npz').exists(),'Fresh actual-point artifact');arrays={};bindings=[];t=self.torch
  with t.no_grad():
   q=t.tensor(p.copy(),dtype=t.float64)
   for sid in IDS:
    pt,_=self.map_head(q,sid);a=pt.detach().numpy().copy();farray(a,(DIM,));need(a.dtype==np.float64,'Actual FP64 point storage');err=close(a,points[(sid,'zero')],'All84 actual Torch/NumPy map before forward',1e-12);arrays[sid]=a;bindings.append(dict(state_id=sid,actual_Torch_FP64_sha256=point_hash(a),NumPy_point_sha256=point_hash(points[(sid,'zero')]),Torch_vs_NumPy_Linf=err))
  np.savez_compressed(folder/'ACTUAL_TORCH_FP64_POINTS.npz',**arrays);dump(folder/'ACTUAL_TORCH_FP64_POINT_BINDINGS.json',bindings);receipt=dict(tag=tag,head_digest=parameter_hash(p),all84_state_ids=IDS,model_forwards_so_far=self.calls['model_forwards'],backward_calls_so_far=self.calls['backward_calls'],sealed_before_this_batch=True,artifact_seals={n:sha(folder/n) for n in ('HEAD.npz','POINT_BINDINGS.json','ACTUAL_TORCH_FP64_POINTS.npz','ACTUAL_TORCH_FP64_POINT_BINDINGS.json')});dump(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',receipt);self.batch=dict(folder=folder,receipt=receipt,receipt_sha256=sha(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'),point_ids={sid:point_hash(a) for sid,a in arrays.items()},seen=[]);self.point_tensors_saved+=84;self.verify_batch(p);return dict(folder=folder.name,**receipt,receipt_sha256=self.batch['receipt_sha256'])
 def verify_batch(self,p):
  need(self.batch is not None,'Actual point batch must be sealed');b=self.batch;compare(parameter_hash(p),b['receipt']['head_digest'],'Actual batch coefficient head');compare(sha(b['folder']/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'),b['receipt_sha256'],'Actual point receipt')
  for n,h in b['receipt']['artifact_seals'].items():compare(sha(b['folder']/n),h,'Actual pre-forward point artifact/'+n)
 def full_forward(self,point,graph=False):
  need(self.authorized_point is not None,'No unregistered FP64 endpoint');compare(point_hash(point.detach().numpy()),self.authorized_point,'Actual graph point equals saved FP64 coordinate bytes before forward');return super().full_forward(point,graph)
 def coefficient_record(self,sid,p,expected,gradient):
  need(self.batch is not None and self.authorized_point is None,'Sealed batch before coefficient endpoint');self.verify_batch(p);need(len(self.batch['seen'])<84 and sid==IDS[len(self.batch['seen'])],'All84 fixed endpoint order');self.authorized_point=self.batch['point_ids'][sid]
  try:row=super().coefficient_record(sid,p,expected,gradient)
  finally:self.authorized_point=None
  compare(row['FP64_point_sha256'],self.batch['point_ids'][sid],'Recorded actual point identity');self.batch['seen'].append(sid);row['actual_FP64_point_artifact']=self.batch['folder'].name+'/ACTUAL_TORCH_FP64_POINTS.npz';return row
def factory(out,ctx,c,points,native,coefs,calls):return SmoothTrainer(out,ctx,c,points,native,coefs,calls)
