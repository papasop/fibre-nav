"""Whole synthetic conditional protocol, including retained confirmation failures."""
import copy,tempfile,unittest,contextlib,io
from core199 import *
from worker199 import run
from fixture199 import fixture,history,Fake
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'input')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def go(self,mode='normal',c=None):
  out=self.root/self._testMethodName
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,c or self.c,lambda *a:copy.deepcopy(self.ctx),lambda out,ctx,calls:Fake(out,ctx,calls,mode),history)
  return s,out
 def test_full624_records_1429_synthetic_forwards_48_each_round(self):
  s,out=self.go();self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['local_confirmation_passed']);self.assertEqual(s['records_completed'],624);self.assertEqual(s['new_model_forwards'],1429);self.assertEqual(s['confirmation_states_evaluated'],48);self.assertEqual(s['confirmation_states_exposed'],48);self.assertTrue(s['all48_coordinates_sealed_before_model']);self.assertTrue(s['all96_proposals_sealed_before_new_endpoints']);self.assertEqual(s['endpoint_summaries']['0']['passed'],48);self.assertEqual(s['per_case_endpoint_summaries']['xor_11']['1']['passed'],16);self.assertFalse(s['all44_originals_independently_bound']);self.assertFalse(s['real_model_execution_claim']);self.assertTrue(s['synthetic_fixture'])
  for k,v in no_training().items():self.assertEqual(s[k],v)
 def test_new_write_failure_retains48_and_both_rounds(self):
  s,_=self.go('new_endpoint_failure');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['local_confirmation_passed']);self.assertEqual(s['endpoint_summaries']['0']['passed'],47);self.assertEqual(s['records_completed'],624);self.assertFalse(s['LC2_closed'])
 def test_invalid_new_preparation_kept_and_endpoints_still_measured(self):
  s,_=self.go('new_preparation_failure');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['valid_preparations_both_rounds'],47);self.assertEqual(s['endpoint_summaries']['1']['failed_gates']['preparation'],1);self.assertEqual(s['records_completed'],624)
 def test_preflight_failure_stops_before_new_measurements(self):
  s,out=self.go('known_failure');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['known_preflight_passed']);self.assertFalse(s['confirmation_evaluation_complete']);self.assertEqual(s['records_completed'],336);self.assertEqual(s['confirmation_states_evaluated'],0);self.assertEqual(s['confirmation_states_exposed'],0);self.assertEqual(s['new_model_forwards'],757);self.assertTrue((out/'ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz').is_file())
 def test_new_repeat_point_change_prevents_pass_even_all_gates_pass(self):
  s,_=self.go('new_repeat_shift');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['endpoint_summaries']['0']['passed'],48);self.assertEqual(s['endpoint_summaries']['1']['passed'],48);self.assertFalse(s['native_repeats_consistent']);self.assertFalse(s['local_confirmation_passed'])
 def test_frozen_head_mutation_stops_and_preserves_partial(self):
  s,out=self.go('mutate_head');self.assertFalse(s['complete']);self.assertLess(s['records_completed'],624);self.assertTrue((out/'SUMMARY.json').exists());self.assertIn('Frozen proposal pipeline',str(s['errors']))
 def test_wrong_whole_hash_blocks_before_model(self):
  c=copy.deepcopy(self.c);c['pins']['C198']['sha256']='0'*64;s,_=self.go(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
if __name__=='__main__':unittest.main(verbosity=2)
