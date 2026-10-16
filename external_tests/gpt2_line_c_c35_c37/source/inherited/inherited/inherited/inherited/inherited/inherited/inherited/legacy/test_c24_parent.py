import copy,hashlib,io,json,unittest,zipfile
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import c24_parent as p
from test_c23_parent import CapsuleTests as LegacyTests
from test_c16 import write_zip

def js(x):return json.dumps(x).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
def metric(d):
 g={k:True for k in p.old.GATES};g['response']=d<=.002
 return {'drift':d,'kl':.001,'gates':g,'episode_pass':all(g.values()),'response_residual':[d]*6}
def cohort(failing=.003):
 bad={'xor_10__v0','xor_10__c15_train0','xor_10__c15_train3','xor_10__c15_train4','xor_10__c20_train3','xor_10__c20_train5'}
 return {t['id']:{'reference_valid':True,'metrics':metric(failing if t['id'] in bad else .001)} for t in p.old.training_specs()}

class CapsuleTests(unittest.TestCase):
 def setUp(self):
  legacy=LegacyTests('test_byte_identity_required_coverage_and_input_untouched')
  with redirect_stdout(io.StringIO()):legacy.setUp()
  self.addCleanup(legacy.doCleanups);self.root=legacy.root;self.stack=ExitStack();self.addCleanup(self.stack.close)
  for t in p.old.test_specs():
   b='states/new/'+t['id']+'/'
   for n in ('prepared.pt','teacher.pt','INPUT.pt'):legacy.members[b+n]=(t['id']+n).encode()
   legacy.members[b+'REFERENCE.json']=js({'spec':t,'valid':True})
   legacy.members['evaluation/new/'+t['id']+'/COMPLETE.json']=js({'id':t['id'],'reference_valid':True,'arms':{'calibrated':metric(.001)}})
  write_zip(legacy.parent,legacy.members)
  with redirect_stdout(io.StringIO()):legacy.create()
  self.companion=legacy.parent;capsule=legacy.output.read_bytes();source=b'original C23 fixture source';self.pins={'x.py':digest(source)}
  self.stack.enter_context(patch.object(p,'pins_at_root',return_value=self.pins))
  self.rows=cohort();self.best={'spec':p.START_SPEC,'candidate_head_sha256':p.START_HEAD_SHA,'full60_verified':True,'head_committed':False,'quality':p.old.quality(self.rows)}
  audit={'spec':p.START_SPEC,'head_sha256':p.START_HEAD_SHA,'full60_verified':True,'replay_error':None,'guard':p.old.guard(cohort(.004),self.rows),'full60_rows':self.rows}
  self.members={'parent_capsule.zip':capsule,p.LAYER_MEMBER:b'actual candidate layer','source/x.py':source,
   'identity.json':js({'protocol':p.previous.PROTOCOL,'settings':p.previous.SETTINGS,'source_hashes':self.pins,'origin_head_sha256':p.previous.START_HEAD_SHA,'parent_capsule_sha256':digest(capsule),'parent_c22_sha256':p.sha(self.companion)}),
   'SUMMARY.json':js({'complete':True,'decision':'C23_FULL60_ACCEPTABLE_CANDIDATE_FOUND','n_completed_candidate_specs':90,'n_unique_candidate_heads':70,'origin_head_unchanged':True,'no_head_committed':True,'best_candidate':self.best}),
   'BEST_CANDIDATE.json':js(self.best),p.AUDIT_MEMBER:js(audit)}
  for i in range(90):self.members['candidates/candidate_%03d/COMPLETE.json'%i]=b'{}';self.members['zero_controls/state_%03d.json'%i]=b'{}'
  self.parent=self.root/'c23_result.zip';write_zip(self.parent,self.members);self.output=self.root/'c24_capsule.zip'
 def create(self):
  with redirect_stdout(io.StringIO()):p.make_capsule(self.parent,self.companion,self.output)
  return p.validate_capsule(self.output)
 def rewrite(self,change):
  with zipfile.ZipFile(self.output) as z:m={n:z.read(n) for n in z.namelist() if n!='MANIFEST.json'}
  change(m);write_zip(self.output,m)
 def test_real_nested_validation_and_originals_untouched(self):
  before=[p.sha(x) for x in (self.parent,self.companion)];self.create();self.assertEqual(before,[p.sha(x) for x in (self.parent,self.companion)])
  with zipfile.ZipFile(self.output) as z:
   self.assertEqual(set(z.namelist()),p.required_members(self.pins));self.assertEqual(z.read('CANDIDATE_LAYER.pt'),self.members[p.LAYER_MEMBER]);self.assertEqual(z.read('c23_capsule.zip'),self.members['parent_capsule.zip'])
   self.assertEqual(sum(n.startswith('evaluation_states/') for n in z.namelist()),64)
 def test_wrong_candidate_head_rejected(self):
  b=json.loads(self.members['BEST_CANDIDATE.json']);b['candidate_head_sha256']='wrong';self.members['BEST_CANDIDATE.json']=js(b);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_incomplete_scan_rejected(self):
  s=json.loads(self.members['SUMMARY.json']);s['complete']=False;self.members['SUMMARY.json']=js(s);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_missing_grid_item_rejected(self):
  self.members.pop('candidates/candidate_089/COMPLETE.json');write_zip(self.parent,self.members)
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_parent(self.parent)
 def test_false_full60_verification_rejected(self):
  a=json.loads(self.members[p.AUDIT_MEMBER]);a['full60_verified']=False;self.members[p.AUDIT_MEMBER]=js(a);write_zip(self.parent,self.members)
  with self.assertRaises(ValueError):p.validate_parent(self.parent)
 def test_wrong_companion_archive_rejected(self):
  info=p.validate_parent(self.parent);info['identity']['parent_c22_sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'original C22'):p.validate_companion(self.companion,info)
 def test_missing_inspected_state_rejected(self):
  self.create();sid=p.old.test_specs()[0]['id'];self.rewrite(lambda m:m.pop('evaluation_states/'+sid+'/prepared.pt'))
  with self.assertRaisesRegex(ValueError,'coverage'):p.validate_capsule(self.output)
 def test_changed_inspected_state_spec_rejected(self):
  self.create();sid=p.old.test_specs()[0]['id'];self.rewrite(lambda m:m.update({'evaluation_states/'+sid+'/REFERENCE.json':js({'spec':{},'valid':True})}))
  with self.assertRaises(ValueError):p.validate_capsule(self.output)
 def test_changed_nested_capsule_rejected(self):
  self.create();self.rewrite(lambda m:m.update({'c23_capsule.zip':b'bad'}))
  with self.assertRaises(ValueError):p.validate_capsule(self.output)
 def test_changed_receipt_rejected(self):
  self.create()
  def change(m):
   x=json.loads(m['PROVENANCE.json']);x['copied_members']['CANDIDATE_LAYER.pt']['sha256']='wrong';m['PROVENANCE.json']=js(x)
  self.rewrite(change)
  with self.assertRaisesRegex(ValueError,'receipt'):p.validate_capsule(self.output)
 def test_failed_import_preserves_existing_output(self):
  self.output.write_bytes(b'keep');self.members.pop(p.LAYER_MEMBER);write_zip(self.parent,self.members)
  with self.assertRaises(KeyError):p.make_capsule(self.parent,self.companion,self.output)
  self.assertEqual(self.output.read_bytes(),b'keep')

if __name__=='__main__':unittest.main()
