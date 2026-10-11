# Workflow test with a synthetic write actuator, not GPT-2 evidence.
import sys,types,tempfile,json,contextlib,io
from pathlib import Path
import torch
for name in ('transformers','peft'):
 module=types.ModuleType(name);module.__version__='synthetic'
 for k in ('AutoTokenizer','AutoModelForCausalLM','LoraConfig','get_peft_model'):setattr(module,k,object)
 sys.modules[name]=module
import worker_b4 as engine
engine.a100_device=lambda torch:(torch.device('cpu'),'synthetic A100')
p=torch.nn.Parameter(torch.zeros(4));tokenizer=types.SimpleNamespace(vocab_size=1000)
class Model:
 config=types.SimpleNamespace(_commit_hash='607a30d783dfa663caf39e06633721c8d4cfcd7e',use_cache=False)
 def named_parameters(self):return [('lora_B',p)]
def factory(cfg,device):
 p.data.zero_();return tokenizer,Model(),[p]
w=engine.w;w.make_model=factory;w.single_token_pair=lambda tok:((' red',10),(' blue',20))
w.next_logits=lambda *args:torch.zeros(12,1000);w.anchor_metrics=lambda *args:(0.,0.)
engine.audit.content_bits=lambda *args:[int(x) for x in p.detach().tolist()]
calls=[]
def actuator(w,model,tok,params,cfg,anchors,response_ids,response0,logits0,exact,train,dev,targets,ids,seed,device,steps):
 assert len(dev)==12, 'New evaluation leaked into writer'
 calls.append(cfg['seed']);w.assign_flat(params,torch.tensor([ids.index(t) for t in targets],dtype=torch.float32))
 return {'history':[],'selected_checkpoint_step':steps}
engine.audit.write_program=actuator
fail_seed=None
def evaluate(w,model,tok,exact,held,targets,ids,device):
 correct=[int(x) for x in p.detach().tolist()]==[ids.index(t) for t in targets]
 # Controlled scientific failure of first seed; later seeds still must finish.
 correct=correct and not (fail_seed is not None and current_seed()==fail_seed)
 return {'exact_accuracy':float(correct),'heldout_accuracy':float(correct),'minimum_exact_margin':float(correct),'minimum_heldout_margin':float(correct),'exact_margins':[float(correct)]*4,'heldout_margins':[float(correct)]*len(held)}
active_seed=None
original_factory=w.make_model
def marked_factory(cfg,device):
 global active_seed
 active_seed=cfg['seed'];return original_factory(cfg,device)
w.make_model=marked_factory
def current_seed():return active_seed
engine.audit.evaluate=evaluate
with tempfile.TemporaryDirectory() as td:
 out=Path(td)/'run';sys.argv=['worker','--out',str(out)]
 with contextlib.redirect_stdout(io.StringIO()):engine.main()
 s=json.loads((out/'SUMMARY.json').read_text());assert s['all_gates_pass'] and s['n_passed']==1 and s['new_seed_passed']==0
 assert len(calls)==5
 assert s['development_only'] and not s['full_b4_confirmation']
 for c in json.loads((out/'seeds'/'82931'/'cases.json').read_text()).values():
  assert c['checks']['reset_changes_computed_result'] and c['checks']['restore_recovers_contents_and_computation']
 from archive_io import restore_archive
 restored=Path(td)/'restored';restore_archive(out.parent/'gpt2_b4_minimal_causal_a100_results.zip',restored)
 # Resume from actual archived endpoint state partway through a seed: no
 # completed writes may be repeated, incomplete remaining cases must continue.
 sd=restored/'seeds'/str(engine.SEEDS[0]);(sd/'SUMMARY.json').unlink();(sd/'evaluation.json').unlink()
 cases=json.loads((sd/'cases.json').read_text());cases.pop('01');(sd/'cases.json').write_text(json.dumps(cases))
 records=json.loads((sd/'records.json').read_text())
 for name in [k for k in records if k.startswith('01_')]:records.pop(name);(sd/(name+'.pt')).unlink()
 (sd/'records.json').write_text(json.dumps(records));sys.argv=['worker','--out',str(restored)]
 before=len(calls)
 with contextlib.redirect_stdout(io.StringIO()):engine.main()
 assert len(calls)-before==5
 assert json.loads((restored/'SUMMARY.json').read_text())['all_gates_pass']
 # Already-complete protocol skips all training.
 before=len(calls)
 with contextlib.redirect_stdout(io.StringIO()):engine.main()
 assert len(calls)==before
 fail_seed=engine.SEEDS[0];failed=Path(td)/'failed';sys.argv=['worker','--out',str(failed)]
 with contextlib.redirect_stdout(io.StringIO()):engine.main()
 sf=json.loads((failed/'SUMMARY.json').read_text());assert sf['complete'] and not sf['all_gates_pass'] and sf['n_passed']==0
print('PASS: synthetic five-stage causal workflow; reset/restore; auxiliary data isolation; partial/archive resume; completed skip; failed endpoint rejects claim')
