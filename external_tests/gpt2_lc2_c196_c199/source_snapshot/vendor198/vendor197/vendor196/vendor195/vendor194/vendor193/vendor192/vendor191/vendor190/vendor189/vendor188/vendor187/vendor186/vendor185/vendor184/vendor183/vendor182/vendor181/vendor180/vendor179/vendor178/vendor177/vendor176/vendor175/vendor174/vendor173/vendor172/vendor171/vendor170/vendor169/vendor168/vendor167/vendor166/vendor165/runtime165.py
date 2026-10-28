"""Unchanged pinned model; preregistered, sealed point batches can be activated once."""
from core165 import *
from runtime163 import SmoothTrainer
class DirectionRuntime(SmoothTrainer):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);r=clean_json(self.reference_file.read_bytes());r['protocol']=PROTOCOL;dump(self.reference_file,r);self.reference_file_hash=sha(self.reference_file);self.registered={};self.verify(True)
 def preseal(self,p,points,folder,tag):
  folder=Path(folder);need(folder.name not in self.registered,'No duplicate registered batch');need(self.authorized_point is None,'No active forward while registering');t=self.torch;arrays={};bindings=[]
  with t.no_grad():
   leaf=t.tensor(p.copy(),dtype=t.float64)
   for sid in IDS:
    pt,_=self.map_head(leaf,sid);a=pt.detach().numpy().copy();need(a.dtype==np.float64,'Actual double point');farray(a,(DIM,));err=close(a,points[(sid,'zero')],'Independent preregistered Torch/NumPy map',1e-12);arrays[sid]=a;bindings.append(dict(state_id=sid,actual_Torch_FP64_sha256=point_hash(a),NumPy_point_sha256=point_hash(points[(sid,'zero')]),Torch_vs_NumPy_Linf=err))
  np.savez_compressed(folder/'ACTUAL_TORCH_FP64_POINTS.npz',**arrays);dump(folder/'ACTUAL_TORCH_FP64_POINT_BINDINGS.json',bindings);receipt=dict(tag=tag,head_digest=parameter_hash(p),all84_state_ids=IDS,model_forwards_so_far=self.calls['model_forwards'],backward_calls_so_far=self.calls['backward_calls'],sealed_before_this_batch=True,artifact_seals={n:sha(folder/n) for n in ('HEAD.npz','POINT_BINDINGS.json','ACTUAL_TORCH_FP64_POINTS.npz','ACTUAL_TORCH_FP64_POINT_BINDINGS.json')});dump(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',receipt);batch=dict(folder=folder,receipt=receipt,receipt_sha256=sha(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'),point_ids={sid:point_hash(a) for sid,a in arrays.items()},seen=[]);self.registered[folder.name]=batch;self.point_tensors_saved+=N;return dict(folder=folder.name,**receipt,receipt_sha256=batch['receipt_sha256'])
 def activate(self,name,p):
  need(self.authorized_point is None,'No active endpoint')
  if self.batch is not None:compare(self.batch['seen'],IDS,'Finish previous complete84 batch')
  need(name in self.registered,'Only sealed registered batch');b=self.registered[name];need(not b['seen'],'No batch reuse');self.batch=b;self.verify_batch(p)
def factory(*args):return DirectionRuntime(*args)
