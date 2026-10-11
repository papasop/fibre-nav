# Test the patched archived writer's real loop on a tiny constrained endpoint.
import sys,types,tempfile
from pathlib import Path
import torch
for name in ('transformers','peft'):
 m=types.ModuleType(name)
 for key in ('AutoTokenizer','AutoModelForCausalLM','LoraConfig','get_peft_model'):setattr(m,key,object)
 sys.modules[name]=m
import writer_core as w
cfg={'write_steps':6,'endpoint_kl_gate':.005,'response_budget':.002,'learning_rate':.025,'retraction_steps':0,
 'semantic_pull_weight':1,'semantic_consistency_weight':1,'worst_item_weight':1,'worst_item_temperature':.07,'margin_weight':1,'margin_target':1}
p=torch.nn.Parameter(torch.zeros(2));params=[p]
def loss(*args,**kwargs):
 x=(p[1]-1)**2;z=x*0;return x,x,z,z,z,z
w.semantic_memory_loss=loss
w.anchor_metrics=lambda *a:(float(abs(p[0].detach())),0.)
w.anchor_kl_tensor=lambda *a:p.sum()*0
w.kl_barrier=lambda x,c:x*0
w.kl_barrier_scalar=lambda *a:0.
w.response_vector=lambda *a:p[:1]
w.retract=lambda *a:0.
w.training_selection_tuple=lambda *a:(float(p[1].detach()),)
w.read_accuracy=lambda *a:1.
w.signed_margins=lambda *a:[float(p[1].detach())]*4
w.view_accuracies=lambda *a:[1.]
args=(None,None,params,None,[],[0],torch.zeros(1),torch.zeros(1),['a']*4,[0]*4,['a']*4,[0]*4,['a']*4,['a']*4,[0]*4,[0,1],'semantic_current',1,'cpu')
def run(c):
 a=list(args);a[3]=c;return w.run_write(*a)
full=run(cfg);expected=p.detach().clone()
p.data.zero_();saved={}
class Interrupt(Exception):pass
def interrupt(s):
 saved.update(s);raise Interrupt()
try:run(dict(cfg,_checkpoint=interrupt))
except Interrupt:pass
assert saved['next_step']==5
# Serialize/restore the exact state through torch's safe weights loader.
with tempfile.TemporaryDirectory() as t:
 path=Path(t)/'progress.pt';torch.save(saved,path)
 resume=torch.load(path,weights_only=True)
 p.data.zero_();restored=run(dict(cfg,_resume=resume))
assert torch.equal(expected,p.detach()),(expected,p)
assert restored['history']==full['history']
assert restored['selected_checkpoint_step']==full['selected_checkpoint_step']
print('ARCHIVED_WRITER_RESUME_MATCHES_UNINTERRUPTED_OK')
