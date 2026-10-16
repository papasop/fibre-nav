import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import c23_parent as p
from test_c22_parent import CapsuleTests as LegacyCapsuleTests
from test_c16 import write_zip

def js(x):return json.dumps(x).encode()
def digest(x):return hashlib.sha256(x).hexdigest()

class CapsuleTests(unittest.TestCase):
 def setUp(self):
  legacy=LegacyCapsuleTests('test_required_coverage_byte_identity_and_original_untouched')
  with redirect_stdout(io.StringIO()):legacy.setUp();legacy.create()
  self.addCleanup(legacy.doCleanups);self.root=legacy.root;self.stack=ExitStack();self.addCleanup(self.stack.close)
  training=legacy.output.read_bytes();cp=b'actual C22 checkpoint';source=b'actual C22 fixture source';self.pins={'x.py':digest(source)}
  self.stack.enter_context(patch.object(p,'START_CP_SHA',digest(cp)));self.stack.enter_context(patch.object(p,'pins_at_root',return_value=self.pins))
  self.members={'parent_capsule.zip':training,p.START_MEMBER:cp,'source/x.py':source,
   'identity.json':js({'protocol':p.previous.PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':self.pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_capsule_sha256':digest(training)}),
   'SUMMARY.json':js({'complete':True,'decision':'C22_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':60,'passed':54}},'original36_training':{'calibrated':{'passed':32}},'prospective_local_state_development':{'calibrated':{'n_requested':16}}}),
   'WRITER_FROZEN.json':js({'selected_checkpoint':p.START_MEMBER,'head_sha256':p.START_HEAD_SHA}),
   'heads/calibrated/PROGRESS.json':js({'checkpoint':'checkpoint_000015.pt','rounds':15,'accepted_updates':13,'done':True,'checkpoint_sha256':digest(cp)}),
   'FINAL_TRAINING.json':js({'fixture':True}),'SPLIT.json':js(p.previous.training_specs()),
   'calibration/round_000014/START.json':js({'head_sha256':p.START_HEAD_SHA,'case':'xor_10','round':14})}
  base='calibration/round_000014/'
  for t in p.previous.training_specs():
   if t['case']=='xor_10':self.members[base+'states/'+t['id']+'/JACOBIAN.pt']=(t['id']+' original geometry').encode()
  for ri in range(3):self.members[base+'COEFFICIENTS_%02d.pt'%ri]=('coefficient'+str(ri)).encode()
  for rdi in range(3):
   for ri in range(3):self.members[base+'DIRECTION_INFO_%02d_%02d.json'%(rdi,ri)]=js({'fixture':[rdi,ri]})
  self.parent=self.root/'c22_result.zip';write_zip(self.parent,self.members);self.output=self.root/'c23_parent.zip'
 def create(self):
  with redirect_stdout(io.StringIO()):p.make_capsule(self.parent,self.output)
  return p.validate_capsule(self.output)
 def rewrite(self,change):
  with zipfile.ZipFile(self.output) as z:m={n:z.read(n) for n in z.namelist() if n!='MANIFEST.json'}
  change(m);write_zip(self.output,m)
 def test_byte_identity_required_coverage_and_input_untouched(self):
  before=p.sha(self.parent);self.create();self.assertEqual(p.sha(self.parent),before)
  with zipfile.ZipFile(self.output) as z:
   self.assertEqual(set(z.namelist()),p.required_members(self.pins));self.assertEqual(z.read('start_checkpoint.pt'),self.members[p.START_MEMBER]);self.assertEqual(z.read('training_capsule.zip'),self.members['parent_capsule.zip'])
   self.assertEqual(sum(n.startswith('geometry/') for n in z.namelist()),15);self.assertEqual(sum(n.startswith('coefficients/') for n in z.namelist()),3)
 def test_wrong_checkpoint_even_valid_manifest_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'start_checkpoint.pt':b'bad'}))
  with self.assertRaisesRegex(ValueError,'checkpoint'):p.validate_capsule(self.output)
 def test_missing_cached_geometry_rejected(self):
  self.create();sid=next(t['id'] for t in p.previous.training_specs() if t['case']=='xor_10');self.rewrite(lambda m:m.pop('geometry/'+sid+'/JACOBIAN.pt'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_missing_coefficient_rejected(self):
  self.create();self.rewrite(lambda m:m.pop('coefficients/COEFFICIENTS_00.pt'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_geometry_wrong_origin_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'GEOMETRY_START.json':js({'head_sha256':'wrong','case':'xor_10','round':14})}))
  with self.assertRaisesRegex(ValueError,'geometry'):p.validate_capsule(self.output)
 def test_incomplete_parent_rejected(self):
  s=json.loads(self.members['SUMMARY.json']);s['complete']=False;self.members['SUMMARY.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'completed'):p.validate_parent(self.parent)
 def test_wrong_actual_origin_head_rejected(self):
  s=json.loads(self.members['WRITER_FROZEN.json']);s['head_sha256']='wrong';self.members['WRITER_FROZEN.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_wrong_cached_round_rejected(self):
  self.members['calibration/round_000014/START.json']=js({'head_sha256':p.START_HEAD_SHA,'case':'xor_10','round':13});write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'round14'):p.validate_parent(self.parent)
 def test_replaced_nested_capsule_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'training_capsule.zip':b'bad'}))
  with self.assertRaises(ValueError):p.validate_capsule(self.output)
 def test_changed_source_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'C22_RECORDS/source/x.py':b'bad'}))
  with self.assertRaisesRegex(ValueError,'source changed'):p.validate_capsule(self.output)
 def test_receipt_digest_changed_rejected(self):
  self.create()
  def mutate(m):
   s=json.loads(m['PROVENANCE.json']);s['copied_members']['start_checkpoint.pt']['sha256']='wrong';m['PROVENANCE.json']=js(s)
  self.rewrite(mutate)
  with self.assertRaisesRegex(ValueError,'origin digest'):p.validate_capsule(self.output)
 def test_failed_import_keeps_previous_output(self):
  self.output.write_bytes(b'keep');self.members[p.START_MEMBER]=b'wrong';write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.make_capsule(self.parent,self.output)
  self.assertEqual(self.output.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
