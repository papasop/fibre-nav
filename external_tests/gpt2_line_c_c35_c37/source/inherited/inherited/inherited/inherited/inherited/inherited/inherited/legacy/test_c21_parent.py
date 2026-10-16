"""Exercise compact-parent extraction and fail-closed replay on synthetic archives."""
import copy,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import c21_parent as p
from test_c16 import write_zip

def digest(b):return hashlib.sha256(b).hexdigest()
def js(obj):return json.dumps(obj).encode()
def zipped(m):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for n,v in m.items():z.writestr(n,v)
 return b.getvalue()
def fixture():
 source=b'original source';pins={'x.py':digest(source)};cp=b'real C20 checkpoint';c9cp=b'original C9 checkpoint'
 c8=zipped({'states/'+sid+'/jacobian.pt':('J '+sid).encode() for sid in p.previous.STATES})
 c9=zipped({'parent_c8.zip':c8,p.CHECKPOINT:c9cp})
 m15={'parent_c9.zip':c9,'STATE_SKETCH.pt':b'fixed sketch'}
 for t in p.previous.original_training_specs():
  sid=t['id']
  for n in ('prepared.pt','teacher.pt','REFERENCE.json'):m15['states/training/'+sid+'/'+n]=(sid+' '+n).encode()
  m15['evaluation/training/'+sid+'/ONLINE_INPUT.pt']=(sid+' input').encode()
 c15=zipped(m15);c17=zipped({'parent_c15.zip':c15});c18=zipped({'parent_c17.zip':c17});c19=zipped({'parent_c18.zip':c18})
 top={'parent_c19.zip':c19,p.START_MEMBER:cp,'source/x.py':source,
 'identity.json':js({'protocol':p.previous.PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':pins,'start_head_sha256':p.previous.START_HEAD_SHA,'parent_c19_sha256':digest(c19)}),
 'SUMMARY.json':js({'complete':True,'decision':'C20_RAW_TRAINING_FUNCTION_FAILURE','writer_head_sha256':p.START_HEAD_SHA,'training':{'calibrated':{'n_requested':60,'passed':36}},'prospective_local_state_development':{'calibrated':{'n_requested':16}}}),
 'WRITER_FROZEN.json':js({'selected_checkpoint':p.START_MEMBER,'head_sha256':p.START_HEAD_SHA}),
 'heads/calibrated/PROGRESS.json':js({'checkpoint':'checkpoint_000016.pt','rounds':16,'checkpoint_sha256':digest(cp)}),
 'FINAL_TRAINING.json':js({'fixture':True}),'SPLIT.json':js(p.previous.training_specs())}
 for t in p.previous.supplemental_training_specs():
  for n in ('prepared.pt','teacher.pt','INPUT.pt','REFERENCE.json'):top['states/coverage_training/'+t['id']+'/'+n]=(t['id']+' '+n).encode()
 patches={'START_CP_SHA':digest(cp),'CHECKPOINT_SHA':digest(c9cp),'C15_SHA':digest(c15),'C9_SHA':digest(c9),'C8_SHA':digest(c8)}
 return top,pins,patches

class CapsuleTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.members,self.pins,patches=fixture();self.stack=ExitStack();self.addCleanup(self.stack.close)
  for n,v in patches.items():self.stack.enter_context(patch.object(p,n,v))
  self.stack.enter_context(patch.object(p,'pins_at_root',return_value=self.pins))
  self.parent=self.root/'parent.zip';write_zip(self.parent,self.members);self.output=self.root/'capsule.zip'
 def create(self):p.make_capsule(self.parent,self.output);return p.validate_capsule(self.output)
 def rewrite(self,change):
  with zipfile.ZipFile(self.output) as z:m={n:z.read(n) for n in z.namelist() if n!='MANIFEST.json'}
  change(m);write_zip(self.output,m)
 def test_raw_bytes_required_coverage_and_input_unchanged(self):
  before=p.sha(self.parent);self.create();self.assertEqual(p.sha(self.parent),before)
  with zipfile.ZipFile(self.output) as z:
   self.assertEqual(set(z.namelist()),p.required_capsule_members(self.pins));self.assertEqual(z.read('start_checkpoint.pt'),self.members[p.START_MEMBER]);self.assertEqual(z.read('C20_RECORDS/source/x.py'),self.members['source/x.py'])
   self.assertFalse(any(n.startswith('parent_c') and n.endswith('.zip') for n in z.namelist()))
   provenance=json.loads(z.read('PROVENANCE.json'));self.assertEqual(provenance['parent_c20_sha256'],before)
   self.assertEqual(sum(n.endswith('/prepared.pt') for n in z.namelist()),60)
 def test_wrong_checkpoint_rejected_with_valid_manifest(self):
  self.create();self.rewrite(lambda m:m.update({'start_checkpoint.pt':b'changed'}))
  with self.assertRaisesRegex(ValueError,'checkpoint'):p.validate_capsule(self.output)
 def test_missing_training_tensor_rejected_with_valid_manifest(self):
  self.create();sid=p.previous.training_specs()[0]['id'];self.rewrite(lambda m:m.pop('training/'+sid+'/teacher.pt'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_wrong_c9_checkpoint_rejected_with_valid_manifest(self):
  self.create();self.rewrite(lambda m:m.update({'c9_checkpoint.pt':b'changed'}))
  with self.assertRaisesRegex(ValueError,'checkpoint'):p.validate_capsule(self.output)
 def test_receipt_sha_mismatch_rejected(self):
  self.create()
  def mutate(m):
   r=json.loads(m['PROVENANCE.json']);r['copied_members']['STATE_SKETCH.pt']['sha256']='0'*64;m['PROVENANCE.json']=js(r)
  self.rewrite(mutate)
  with self.assertRaisesRegex(ValueError,'origin digest'):p.validate_capsule(self.output)
 def test_missing_receipt_rejected(self):
  self.create()
  def mutate(m):
   r=json.loads(m['PROVENANCE.json']);r['copied_members'].pop('STATE_SKETCH.pt');m['PROVENANCE.json']=js(r)
  self.rewrite(mutate)
  with self.assertRaisesRegex(ValueError,'origin coverage'):p.validate_capsule(self.output)
 def test_changed_archived_source_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'C20_RECORDS/source/x.py':b'changed'}))
  with self.assertRaisesRegex(ValueError,'source changed'):p.validate_capsule(self.output)
 def test_nested_lineage_mismatch_rejected(self):
  self.members['parent_c19.zip']=b'not the original';write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'lineage'):p.make_capsule(self.parent,self.output)
 def test_failed_import_does_not_replace_existing_capsule(self):
  self.output.write_bytes(b'keep old output');self.members[p.START_MEMBER]=b'bad';write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.make_capsule(self.parent,self.output)
  self.assertEqual(self.output.read_bytes(),b'keep old output')
 def test_incomplete_parent_is_rejected(self):
  s=json.loads(self.members['SUMMARY.json']);s['complete']=False;self.members['SUMMARY.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'completed'):p.make_capsule(self.parent,self.output)

if __name__=='__main__':unittest.main()
