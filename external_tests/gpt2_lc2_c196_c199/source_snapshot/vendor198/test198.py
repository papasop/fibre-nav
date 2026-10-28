"""End-to-end synthetic protocol tests; never claim real GPT-2 results."""
import copy,tempfile,unittest,contextlib,io
from core198 import *
from fixture198 import fixture,Fake
from worker198 import run
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'inputs')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def go(self,mode='normal',c=None):
  out=self.root/self._testMethodName
  with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,c or self.c,lambda *a:copy.deepcopy(self.ctx),lambda out,ctx,calls:Fake(out,ctx,calls,mode))
  return s,out
 def test_full_synthetic_672_records_1513_counted_forwards_no_training(self):
  s,out=self.go();self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['frozen_training_replay_passed']);self.assertEqual(s['new_model_forwards'],1513);self.assertEqual(s['records_completed'],672);self.assertEqual(s['runtime_calls']['feature_calls'],168);self.assertEqual(s['runtime_calls']['metric_calls'],504);self.assertTrue(s['all_fresh_proposals_sealed_before_endpoints']);self.assertFalse(s['all43_originals_independently_bound']);self.assertFalse(s['real_model_execution_claim']);self.assertTrue(s['synthetic_fixture']);self.assertEqual(len(clean_json((out/'ALL252_REPEAT_CHECKS.json').read_bytes())),252)
  for k,v in no_training().items():self.assertEqual(s[k],v)
 def test_failed_endpoint_keeps_full84_and_both_rounds(self):
  s,_=self.go('failed_endpoint');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['frozen_training_replay_passed']);self.assertEqual(s['records_completed'],672);self.assertEqual(s['fresh_endpoint_summaries']['0']['passed'],83);self.assertEqual(s['fresh_endpoint_summaries']['1']['failed_ids'],[IDS[0]]);self.assertFalse(s['LC2_closed'])
 def test_fresh_shift_never_substituted_with_archived_features_or_routes(self):
  s,_=self.go('fresh_shift');self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['archived_control_replay_passed']);self.assertGreater(s['fresh_feature_diagnostics']['capture_Linf_vs_archive'],0);self.assertGreater(s['fresh_feature_diagnostics']['point_Linf_vs_C196'],0);self.assertEqual(s['records_completed'],672)
 def test_repeat_point_change_blocks_acceptance_even_when_all_gates_pass(self):
  s,_=self.go('repeat_shift');self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['fresh_end_to_end_all84_passed_both_rounds']);self.assertFalse(s['native_repeats_consistent']);self.assertFalse(s['frozen_training_replay_passed']);self.assertEqual(s['records_completed'],672)
 def test_failed_preparation_remains_denominator_failure(self):
  s,_=self.go('failed_preparation');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['all84_preparations_passed_both_rounds']);self.assertEqual(s['fresh_endpoint_summaries']['0']['failed_gates']['preparation'],1);self.assertEqual(s['records_completed'],672)
 def test_frozen_head_mutation_stops_and_packages_partial_summary(self):
  s,out=self.go('mutate_head');self.assertFalse(s['complete']);self.assertLess(s['records_completed'],672);self.assertIn('Full frozen proposal pipeline',str(s['errors']));self.assertTrue((out/'SUMMARY.json').exists())
 def test_wrong_whole_original_blocks_before_context_or_model(self):
  c=copy.deepcopy(self.c);c['pins']['C196']['sha256']='0'*64;s,_=self.go(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
 def test_missing_original_blocks_before_model(self):
  with contextlib.redirect_stdout(io.StringIO()):s=run({},self.root/'missing',self.c,lambda *a:(_ for _ in ()).throw(AssertionError('No loader')))
  self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
if __name__=='__main__':unittest.main(verbosity=2)
