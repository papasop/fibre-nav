# Exercise orchestration and causal controls without downloading model weights.
import sys,types,tempfile,json
from pathlib import Path
import torch
for name in ('transformers','peft'):
 m=types.ModuleType(name);m.__version__='synthetic-test'
 for k in ('AutoTokenizer','AutoModelForCausalLM','LoraConfig','get_peft_model'):setattr(m,k,object)
 sys.modules[name]=m
import worker_b4 as engine
engine.a100_device=lambda torch:(torch.device("cpu"),"A100 mocked for CPU workflow test")
p=torch.nn.Parameter(torch.zeros(4))
class Model:
 config=types.SimpleNamespace(_commit_hash='synthetic-revision',use_cache=False)
 def named_parameters(self):return [('lora_B',p)]
model=Model();tok=types.SimpleNamespace(vocab_size=1000)
w=engine.w
w.make_model=lambda cfg,device:(tok,model,[p])
w.single_token_pair=lambda tok:((' red',10),(' blue',20))
w.next_logits=lambda *args:torch.zeros(12,1000)
w.anchor_metrics=lambda *args:(0.,0.)
engine.audit.content_bits=lambda *args:[int(x) for x in p.detach().tolist()]
# Use declared target token ids solely in the explicit fake write actuator.
def actuator(w,model,tok,params,cfg,anchors,response_ids,response0,logits0,exact,train,held,targets,ids,seed,device,steps):
 w.assign_flat(params,torch.tensor([ids.index(t) for t in targets],dtype=torch.float32))
 return {'history':[],'selected_checkpoint_step':steps}
engine.audit.write_program=actuator

def evaluate(w,model,tok,exact,held,targets,ids,device):
 correct=[int(x) for x in p.detach().tolist()]==[ids.index(t) for t in targets]
 return {'exact_accuracy':float(correct),'heldout_accuracy':float(correct),'minimum_exact_margin':1.,'minimum_heldout_margin':1.}
engine.audit.evaluate=evaluate
with tempfile.TemporaryDirectory() as t:
 out=Path(t)/'run';sys.argv=['worker','--out',str(out)]
 engine.main();summary=json.loads((out/'SUMMARY.json').read_text())
 assert summary['b4_pass'] and not summary['b5_claim'] and not summary['independent_confirmation']
 assert len(json.loads((out/'records.json').read_text()))==4
 from archive_io import restore_archive
 restore_archive(out.parent/'gpt2_b4_a100_results.zip',Path(t)/'restored')
 # Selected completed phases must not invoke the actuator when resumed.
 p.data.zero_()
 def forbidden(*args,**kwargs):raise AssertionError('Completed phase was retrained')
 engine.audit.write_program=forbidden;engine.main()
 assert json.loads((out/'SUMMARY.json').read_text())['b4_pass']
print('B4_SEQUENCE_CAUSAL_CONTROLS_AND_COMPLETED_SKIP_OK (synthetic actuator, not GPT-2)')
