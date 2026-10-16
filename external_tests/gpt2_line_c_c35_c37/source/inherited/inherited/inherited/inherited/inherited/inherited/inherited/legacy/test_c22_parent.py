"""C22 extraction uses a genuinely verified inherited capsule fixture."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import c22_parent as p
import c21_parent as old
from test_c21_parent import fixture as old_fixture
from test_c16 import write_zip

def js(x):return json.dumps(x).encode()
def digest(x):return hashlib.sha256(x).hexdigest()

class CapsuleTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.stack=ExitStack();self.addCleanup(self.stack.close)
  original,pins,patches=old_fixture()
  for n,v in patches.items():self.stack.enter_context(patch.object(old,n,v))
  self.stack.enter_context(patch.object(old,'pins_at_root',return_value=pins))
  base=self.root/'c20.zip';write_zip(base,original);training=self.root/'training.zip';old.make_capsule(base,training)
  cp=b'actual C21 final checkpoint';source=b'original C21 source';self.pins={'x.py':digest(source)}
  self.stack.enter_context(patch.object(p,'START_CP_SHA',digest(cp)));self.stack.enter_context(patch.object(p,'pins_at_root',return_value=self.pins))
  self.members={'parent_capsule.zip':training.read_bytes(),p.START_MEMBER:cp,'source/x.py':source,
   'identity.json':js({'protocol':p.previous.PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':self.pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_capsule_sha256':digest(training.read_bytes())}),
   'SUMMARY.json':js({'complete':True,'decision':'C21_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':60,'passed':50}},'original36_training':{'calibrated':{'passed':30}},'prospective_local_state_development':{'calibrated':{'n_requested':16}}}),
   'WRITER_FROZEN.json':js({'selected_checkpoint':p.START_MEMBER,'head_sha256':p.START_HEAD_SHA}),
   'heads/calibrated/PROGRESS.json':js({'checkpoint':'checkpoint_000016.pt','rounds':16,'checkpoint_sha256':digest(cp)}),
   'FINAL_TRAINING.json':js({'fixture':True}),'SPLIT.json':js(p.previous.training_specs())}
  self.parent=self.root/'c21.zip';write_zip(self.parent,self.members);self.output=self.root/'c22_capsule.zip'
 def create(self):p.make_capsule(self.parent,self.output);return p.validate_capsule(self.output)
 def rewrite(self,change):
  with zipfile.ZipFile(self.output) as z:m={n:z.read(n) for n in z.namelist() if n!='MANIFEST.json'}
  change(m);write_zip(self.output,m)
 def test_required_coverage_byte_identity_and_original_untouched(self):
  before=p.sha(self.parent);self.create();self.assertEqual(p.sha(self.parent),before)
  with zipfile.ZipFile(self.output) as z:
   self.assertEqual(set(z.namelist()),p.required_members(self.pins));self.assertEqual(z.read('start_checkpoint.pt'),self.members[p.START_MEMBER]);self.assertEqual(z.read('training_capsule.zip'),self.members['parent_capsule.zip'])
   self.assertEqual(json.loads(z.read('PROVENANCE.json'))['parent_c21_sha256'],before)
 def test_changed_checkpoint_rejected_with_valid_manifest(self):
  self.create();self.rewrite(lambda m:m.update({'start_checkpoint.pt':b'bad'}))
  with self.assertRaisesRegex(ValueError,'checkpoint'):p.validate_capsule(self.output)
 def test_missing_training_capsule_rejected(self):
  self.create();self.rewrite(lambda m:m.pop('training_capsule.zip'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_nested_training_capsule_changed_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'training_capsule.zip':b'bad'}))
  with self.assertRaises(ValueError):p.validate_capsule(self.output)
 def test_changed_source_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'C21_RECORDS/source/x.py':b'changed'}))
  with self.assertRaisesRegex(ValueError,'source changed'):p.validate_capsule(self.output)
 def test_receipt_digest_mismatch_rejected(self):
  self.create()
  def change(m):
   a=json.loads(m['PROVENANCE.json']);a['copied_members']['start_checkpoint.pt']['sha256']='0'*64;m['PROVENANCE.json']=js(a)
  self.rewrite(change)
  with self.assertRaisesRegex(ValueError,'origin digest'):p.validate_capsule(self.output)
 def test_receipt_coverage_mismatch_rejected(self):
  self.create()
  def change(m):
   a=json.loads(m['PROVENANCE.json']);a['copied_members'].pop('start_checkpoint.pt');m['PROVENANCE.json']=js(a)
  self.rewrite(change)
  with self.assertRaisesRegex(ValueError,'origin coverage'):p.validate_capsule(self.output)
 def test_incomplete_C21_rejected(self):
  a=json.loads(self.members['SUMMARY.json']);a['complete']=False;self.members['SUMMARY.json']=js(a);write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'completed'):p.make_capsule(self.parent,self.output)
 def test_wrong_training_cohort_rejected(self):
  a=json.loads(self.members['SUMMARY.json']);a['training']['calibrated']['passed']=49;self.members['SUMMARY.json']=js(a);write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'cohort'):p.make_capsule(self.parent,self.output)
 def test_wrong_frozen_head_rejected(self):
  a=json.loads(self.members['WRITER_FROZEN.json']);a['head_sha256']='wrong';self.members['WRITER_FROZEN.json']=js(a);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.make_capsule(self.parent,self.output)
 def test_failed_import_keeps_existing_output(self):
  self.output.write_bytes(b'keep');self.members[p.START_MEMBER]=b'wrong';write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.make_capsule(self.parent,self.output)
  self.assertEqual(self.output.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
