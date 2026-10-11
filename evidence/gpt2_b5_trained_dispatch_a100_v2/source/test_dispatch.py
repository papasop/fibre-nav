import unittest
from b5_protocol import *
from controller_data import training_samples,TRAIN_TUPLES,HELDOUT,prompt
class Tests(unittest.TestCase):
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
