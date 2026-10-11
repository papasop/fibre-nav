import unittest,tempfile,json,hashlib,zipfile
from pathlib import Path
from protocol import *
from archive_io import restore_archive
class Tests(unittest.TestCase):
 def test_xor_all_inputs_and_reader_calls(self):
  for a in range(2):
   for b in range(2):
    calls=[]
    def reader(i):calls.append(i);return [a,b][i]
    self.assertEqual(controller(reader)['value'],a^b);self.assertEqual(calls,[0,1])
 def test_read_block_control_cannot_access_state(self):
  def fail(i):raise AssertionError('blocked read executed')
  self.assertEqual(controller(fail,True)['value'],0)
 def test_content_intervention_changes_result(self):
  self.assertNotEqual(controller(lambda i:INITIAL[i])['value'],controller(lambda i:SUBSTITUTED[i])['value'])
 def test_invalid_read_rejected(self):
  with self.assertRaises(ValueError):controller(lambda i:2)
 def test_all_acceptance_budgets(self):
  cfg={'response_budget':.002,'endpoint_kl_gate':.005}
  ev={'exact_accuracy':1,'heldout_accuracy':1,'minimum_exact_margin':.05,'minimum_heldout_margin':.05}
  self.assertTrue(endpoint_pass(ev,.002,.005,cfg))
  for key,val in [('exact_accuracy',.9),('heldout_accuracy',.9),('minimum_exact_margin',.049),('minimum_heldout_margin',-.1)]:
   bad=dict(ev);bad[key]=val;self.assertFalse(endpoint_pass(bad,0,0,cfg))
  self.assertFalse(endpoint_pass(ev,.003,0,cfg));self.assertFalse(endpoint_pass(ev,0,.006,cfg))
 def test_archive_roundtrip_and_corruption(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.zip';files={'identity.json':json.dumps({'protocol':PROTOCOL}).encode(),'phase.pt':b'checkpoint bytes'}
   def write(corrupt=False):
    with zipfile.ZipFile(p,'w') as z:
     for n,b in files.items():z.writestr(n,b+b'wrong' if corrupt and n=='phase.pt' else b)
     z.writestr('MANIFEST.json',json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()}))
   write();restore_archive(p,Path(t)/'out');self.assertEqual((Path(t)/'out/phase.pt').read_bytes(),files['phase.pt'])
   write(True)
   with self.assertRaises(ValueError):restore_archive(p,Path(t)/'out2')
 def test_original_source_is_preserved(self):
  root=Path(__file__).parent;m=json.loads((root/'parent_manifest.json').read_text())
  self.assertEqual(hashlib.sha256((root/'writer_core_original.py').read_bytes()).hexdigest(),m['writer_core.py'])
if __name__=='__main__':unittest.main()
