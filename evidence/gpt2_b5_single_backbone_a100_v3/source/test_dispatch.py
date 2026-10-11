import unittest
from b5_protocol import *
from controller_data import training_samples,TRAIN_TUPLES,HELDOUT,prompt
from adapter_runtime import activate
from pathlib import Path
import ast
class Tests(unittest.TestCase):
 def test_adapter_routing_freezes_all_but_selected_B(self):
  class Param:
   def requires_grad_(self,value):self.requires_grad=value
  class Model:
   def __init__(self):self.ps={n:Param() for n in ('base.weight','lora_A.default.weight','lora_B.default.weight','lora_A.controller.weight','lora_B.controller.weight')}
   def set_adapter(self,name):self.active=name
   def named_parameters(self):return self.ps.items()
  model=Model()
  activate(model,'default',True)
  self.assertEqual([n for n,p in model.ps.items() if p.requires_grad],['lora_B.default.weight'])
  activate(model,'controller',True)
  self.assertEqual([n for n,p in model.ps.items() if p.requires_grad],['lora_B.controller.weight'])
  activate(model,'controller',False)
  self.assertFalse(any(p.requires_grad for p in model.ps.values()))
 def test_one_backbone_loader(self):
  root=Path(__file__).resolve().parent
  count=0
  for name in ('worker_b5.py','controller_train.py'):
   tree=ast.parse((root/name).read_text())
   count+=sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='make_model' for n in ast.walk(tree))
  self.assertEqual(count,1)
 def test_split_disjoint(self):
  self.assertFalse(set(TRAIN_TUPLES)&set(HELDOUT));self.assertEqual(len(TRAIN_TUPLES),22)
  rows=training_samples();self.assertEqual(len(rows),440)
  self.assertTrue(all(tuple(r['task']) not in HELDOUT for r in rows))
 def test_initial_prompt_has_no_values_or_plan(self):
  text=prompt(HELDOUT[0],[])
  self.assertNotIn('R0=',text);self.assertNotIn('Action: READ',text)
 def test_teacher_feedback_matches_runtime(self):
  self.assertTrue(any('Feedback: WRITE accepted' in r['prompt'] for r in training_samples()))
 def test_runtime_never_supplies_sequence(self):
  regs={};writes=[]
  self.assertEqual(dispatch('WRITE 11 R2',regs,lambda a:0,lambda a,b:writes.append((a,b))),'ERROR missing computed register')
  self.assertEqual(writes,[])
 def test_correct_sequence(self):
  bits=[0,1,1,0];start=bits.copy();regs={};trace=[]
  def write(a,b):bits[int(a,2)]=b;return True
  for a in ('READ 00 R0','READ 01 R1','XOR R0 R1 R2','WRITE 11 R2','HALT'):
   trace.append({'action':a,'feedback':dispatch(a,regs,lambda x:bits[int(x,2)],write)})
  self.assertTrue(all(acceptance(trace,start,bits,True,True).values()))
 def test_trivial_halt_rejected(self):
  self.assertFalse(all(acceptance([], [0,1,1,0],[0,1,1,0],True,True).values()))
 def test_wrong_address_rejected(self):
  t=[{'action':'WRITE 10 R2','feedback':'WRITE accepted'}]
  self.assertFalse(acceptance(t,[0,1,1,0],[0,1,1,1],True,True)['output_write_executed'])
 def test_no_hidden_value_in_task(self):
  self.assertNotIn('R0=',TASK);self.assertNotIn('R1=',TASK)
  self.assertEqual(len(set(ACTIONS)),len(ACTIONS))
if __name__=='__main__':unittest.main(verbosity=2)
