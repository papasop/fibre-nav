"""Synthetic C35->C36->C37 lineage with real archive verification; no GPT-2 claim."""
import sys,tempfile,unittest,contextlib,io,zipfile,shutil
from pathlib import Path
from unittest.mock import patch
from c37_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
import torch
from c37_protocol import *
from c37_parent import prepare,validate
from worker_c37 import main,freeze,centre,summarize,prepare_new,audit_parent
from c36_protocol import SETTINGS as C36_SETTINGS
from worker_c36 import main as c36_main
import test_c36_torch as parent_fixture
from test_c36_torch import fake_identity
from test_c35_torch import MockRuntime
from c31_runtime import load,save_tensor,tensor_hash

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);print('C37 SELFTEST SETUP 1/3: synthetic C35 archive',flush=True)
  parent_fixture.Tests.setUpClass();cls.temp=tempfile.TemporaryDirectory();cls.r=Path(cls.temp.name)
  cls.c36patch=patch.dict(C36_SETTINGS,{'expected_parent_freeze_sha256':parent_fixture.Tests.freeze_sha});cls.c36patch.start()
  from c36_parent import prepare as prepare36
  from c36_protocol import identity as id36,specs as specs36
  cls.c36=cls.r/'c36';cls.c36.mkdir()
  with contextlib.redirect_stdout(io.StringIO()),patch('worker_c36.snapshot'):
   dump(cls.c36/'identity.json',id36(ROOT/'inherited'));prepare36(parent_fixture.Tests.parent_zip,cls.c36,ROOT/'inherited');dump(cls.c36/'NEW_STATE_SPECS.json',specs36());c36_main(cls.c36,runtime_factory=MockRuntime,identity_checker=fake_identity)
  cls.c37patch=patch.dict(SETTINGS,{'expected_parent_freeze_sha256':sha(cls.c36/'C36_FROZEN.json'),'expected_writer_freeze_sha256':parent_fixture.Tests.freeze_sha});cls.c37patch.start()
  print('C37 SELFTEST SETUP 2/3: verify synthetic C36 and copy frozen context',flush=True)
  cls.parent_zip=cls.r/'c36.zip';m={}
  with zipfile.ZipFile(cls.parent_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for p in cls.c36.rglob('*'):
    if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts:
     n=p.relative_to(cls.c36).as_posix();z.write(p,n);m[n]=sha(p)
   for n,h in read(ROOT/'C36_SOURCE_PINS.json').items():z.write(ROOT/'inherited'/n,'source/'+n);m['source/'+n]=h
   z.writestr('MANIFEST.json',json.dumps(m))
  cls.template=cls.r/'template';cls.template.mkdir()
  with contextlib.redirect_stdout(io.StringIO()):
   dump(cls.template/'identity.json',identity(ROOT));prepare(cls.parent_zip,cls.template,ROOT);dump(cls.template/'NEW_STATE_SPECS.json',specs())
  print('C37 SELFTEST SETUP 3/3: independent test copies ready',flush=True)
 @classmethod
 def tearDownClass(cls):cls.c37patch.stop();cls.c36patch.stop();cls.temp.cleanup();parent_fixture.Tests.tearDownClass()
 def setUp(self):
  print('C37 SELFTEST CASE',self._testMethodName,flush=True);self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)/'session/run';shutil.copytree(self.template,self.out)
 def tearDown(self):self.temp.cleanup()
 def run_main(self,rt=MockRuntime):
  with contextlib.redirect_stdout(io.StringIO()),patch('worker_c37.snapshot'):return main(self.out,runtime_factory=rt,identity_checker=fake_identity)
 def test_full32_changed_centres_and_completed_resume(self):
  before=sha(self.parent_zip);s=self.run_main();self.assertTrue(s['complete']);self.assertEqual(s['decision'],'C37_BOTH_FROZEN_WRITERS_CENTRE_TRANSFER_PASS')
  self.assertEqual(s['new_local']['selected6']['complete_passed'],32);self.assertEqual(s['centre_preparation_Jacobian_files'],16)
  self.assertEqual(sha(self.parent_zip),before);self.assertFalse(s['internal_parameter_update'])
  def forbidden(*a):raise AssertionError('Completed resume loaded model')
  self.assertEqual(self.run_main(forbidden),s)
 def test_interrupt_resumes_same_sealed_candidate(self):
  class Interrupt(MockRuntime):
   def evaluate(self,*a,**kw):
    if self.evaluation_calls==14:raise RuntimeError('C37 mid-endpoint interruption')
    return super().evaluate(*a,**kw)
  with self.assertRaisesRegex(RuntimeError,'mid-endpoint'):self.run_main(Interrupt)
  p=self.out/'c35/proposals'/specs()[0]['id']/'POINTS.pt';h=sha(p);fr=sha(self.out/'C37_FROZEN.json')
  self.assertTrue(self.run_main()['complete']);self.assertEqual(sha(p),h);self.assertEqual(sha(self.out/'C37_FROZEN.json'),fr)
 def test_centre_is_prewrite_preparation_not_written_point(self):
  c=centres()[0];t=centre(self.out,c);d=self.out/'parent_c36/c35';expected=load(d/'new_states'/c['id']/'prepared.pt')['theta'];self.assertTrue(torch.equal(t,expected))
  self.assertFalse(torch.equal(t,load(d/'proposals'/c['id']/'POINTS.pt')['points']['raw']))
 def test_context_weight_tamper_rejected(self):
  p=self.out/'c35/fit/xor_00/HEAD.pt';p.write_bytes(p.read_bytes()+b'changed')
  with self.assertRaises(ValueError):validate(self.out,ROOT)
 def test_original_parent_tamper_rejected(self):
  p=self.out/'parent_c36/c35/new_states'/centres()[0]['id']/'prepared.pt';p.write_bytes(p.read_bytes()+b'changed')
  with self.assertRaises(ValueError):validate(self.out,ROOT)
 def test_freeze_required_before_generation(self):
  (self.out/'c35/new_states'/specs()[0]['id']).mkdir(parents=True);dump(self.out/'PARENT_AUDIT.json',{'test':True})
  with self.assertRaises(ValueError):freeze(self.out)
 def test_collapsed_perturbation_is_invalid_without_resampling(self):
  with contextlib.redirect_stdout(io.StringIO()):dump(self.out/'PARENT_AUDIT.json',audit_parent(self.out,fake_identity))
  freeze(self.out);base=centre(self.out,centres()[0])
  class Collapsed(MockRuntime):
   def correct_preparation(self,initial,folder):return base.clone(),[],0.
  with contextlib.redirect_stdout(io.StringIO()):
   prep,inp,valid=prepare_new(Collapsed(self.out/'c35',ROOT),self.out,specs()[0])
  r=read(self.out/'c35/new_states'/specs()[0]['id']/'COMPLETE.json')
  self.assertFalse(valid);self.assertFalse(r['distinctness_passed']);self.assertEqual(r['prepared_distance_from_centre'],0.)
  self.assertEqual(r['generation_retries'],0);self.assertFalse(r['replacement_used'])
 def test_invalid_state_and_partial_branch_remain_visible(self):
  self.run_main();rs=[read(self.out/'c35/results'/s['id']/'COMPLETE.json') for s in specs()]
  rs[0]['preparation_valid']=False
  for a in ARMS:rs[0]['arms'][a]['complete_accepted']=False
  s=summarize(self.out,rs);self.assertEqual(s['invalid_preparations'],1);self.assertEqual(s['new_local']['selected6']['n_requested'],32);self.assertEqual(s['new_local']['selected6']['complete_passed'],31)
if __name__=='__main__':unittest.main(verbosity=2)
