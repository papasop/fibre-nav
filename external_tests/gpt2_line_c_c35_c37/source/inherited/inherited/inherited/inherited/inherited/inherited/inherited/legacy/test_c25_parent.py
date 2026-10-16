import hashlib,io,json,unittest,zipfile
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import c25_parent as p
from test_c24_parent import CapsuleTests as LegacyTests
from test_c16 import write_zip

def js(x):return json.dumps(x).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
def metric(d=.001):
 g={k:True for k in p.previous.GATES};g['response']=d<=.002
 return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d]*6}

class CapsuleTests(unittest.TestCase):
 def setUp(self):
  legacy=LegacyTests('test_real_nested_validation_and_originals_untouched')
  with redirect_stdout(io.StringIO()):legacy.setUp();legacy.create()
  self.addCleanup(legacy.doCleanups);self.root=legacy.root;self.stack=ExitStack();self.addCleanup(self.stack.close)
  capsule=legacy.output.read_bytes();cp=b'actual final C24 tensors';source=b'actual C24 source fixture';self.pins={'x.py':digest(source)}
  self.stack.enter_context(patch.object(p,'START_CP_SHA',digest(cp)));self.stack.enter_context(patch.object(p,'pins_at_root',return_value=self.pins))
  self.rows={t['id']:{'reference_valid':True,'metrics':metric()} for t in p.previous.training_specs()}
  self.members={'parent_capsule.zip':capsule,p.START_MEMBER:cp,'source/x.py':source,
   'identity.json':js({'protocol':p.previous.PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':self.pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_capsule_sha256':digest(capsule)}),
   'SUMMARY.json':js({'complete':True,'decision':'C24_RAW_INSPECTED_LOCAL_STATE_FAIL','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':60,'passed':60}},'inspected_local_state_development':{'calibrated':{'n_requested':16,'passed':9}}}),
   'WRITER_FROZEN.json':js({'selected_checkpoint':p.START_MEMBER,'head_sha256':p.START_HEAD_SHA}),
   'heads/calibrated/PROGRESS.json':js({'checkpoint':'checkpoint_000008_final.pt','checkpoint_sha256':digest(cp),'rounds':8,'accepted_updates':8,'done':True}),
   'FINAL_TRAINING.json':js(self.rows),'SPLIT.json':js({'fixture':True})}
  for t in p.previous.test_specs():self.members['evaluation/inspected/'+t['id']+'/COMPLETE.json']=js({'id':t['id'],'reference_valid':True,'arms':{'calibrated':metric()}})
  self.parent=self.root/'c24_result.zip';write_zip(self.parent,self.members);self.output=self.root/'c25_capsule.zip'
 def create(self):
  with redirect_stdout(io.StringIO()):p.make_capsule(self.parent,self.output)
  return p.validate_capsule(self.output)
 def rewrite(self,change):
  with zipfile.ZipFile(self.output) as z:m={n:z.read(n) for n in z.namelist() if n!='MANIFEST.json'}
  change(m);write_zip(self.output,m)
 def test_real_nested_validation_and_byte_identity(self):
  before=p.sha(self.parent);self.create();self.assertEqual(before,p.sha(self.parent))
  with zipfile.ZipFile(self.output) as z:self.assertEqual(set(z.namelist()),p.required_members(self.pins));self.assertEqual(z.read('start_checkpoint.pt'),self.members[p.START_MEMBER]);self.assertEqual(z.read('c24_capsule.zip'),self.members['parent_capsule.zip'])
 def test_wrong_actual_checkpoint_rejected(self):
  self.members[p.START_MEMBER]=b'bad';write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_incomplete_parent_rejected(self):
  s=json.loads(self.members['SUMMARY.json']);s['complete']=False;self.members['SUMMARY.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_wrong_cohort_rejected(self):
  s=json.loads(self.members['SUMMARY.json']);s['training']['calibrated']['passed']=59;self.members['SUMMARY.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_wrong_actual_final_training_rejected(self):
  rows=json.loads(self.members['FINAL_TRAINING.json']);rows[next(iter(rows))]['metrics']=metric(.003);self.members['FINAL_TRAINING.json']=js(rows);write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'all pass'):p.validate_parent(self.parent)
 def test_missing_promoted_record_rejected(self):
  t=p.previous.test_specs()[0];self.members.pop('evaluation/inspected/'+t['id']+'/COMPLETE.json');write_zip(self.parent,self.members)
  with self.assertRaises(KeyError):p.validate_parent(self.parent)
 def test_missing_nested_capsule_rejected(self):
  self.create();self.rewrite(lambda m:m.pop('c24_capsule.zip'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_changed_nested_capsule_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'c24_capsule.zip':b'bad'}))
  with self.assertRaises(ValueError):p.validate_capsule(self.output)
 def test_changed_origin_receipt_rejected(self):
  self.create()
  def change(m):
   x=json.loads(m['PROVENANCE.json']);x['copied_members']['start_checkpoint.pt']['sha256']='wrong';m['PROVENANCE.json']=js(x)
  self.rewrite(change)
  with self.assertRaisesRegex(ValueError,'origin digest'):p.validate_capsule(self.output)
 def test_failed_import_does_not_overwrite(self):
  self.output.write_bytes(b'keep');self.members[p.START_MEMBER]=b'bad';write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.make_capsule(self.parent,self.output)
  self.assertEqual(self.output.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
