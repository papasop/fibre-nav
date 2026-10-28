"""Acceptance uses every native gate, retains failures and requires both repeats."""
import copy,tempfile,unittest
from core181 import *
from diagnostics181 import qualification_diagnostics
class Checks(unittest.TestCase):
 def fixture(self):
  native={}
  for arm in ('initial','final'):
   for rnd in (0,1):native['native_'+arm+str(rnd)]=[dict(state_id=sid,metrics=dict(observed=[.001]*22,KL=0.),qualification=dict(complete_pass=True,failed_gates=[])) for sid in IDS]
  return dict(native=native),[dict(state_id=sid,response_loss_by_frame=[0.]*9,FP64_response_Linf_by_frame=[.001]*9) for sid in IDS]
 def fail(self,a,arm,i,gates):
  for rnd in (0,1):a['native']['native_'+arm+str(rnd)][i]['qualification']=dict(complete_pass=False,failed_gates=gates)
 def run_diag(self,a,states):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);s=qualification_diagnostics(a,t.name,states);return s,Path(t.name)
 def test_response_pass_with_failed_read_margin_is_not_rescue(self):
  a,states=self.fixture();self.fail(a,'initial',0,['response']);self.fail(a,'final',0,['exact_read_margin']);s,out=self.run_diag(a,states);self.assertEqual(s['rescued_ids_both_native_rounds'],[]);self.assertEqual(s['final_native_failure_ids'],[IDS[0]]);self.assertEqual(s['remaining_failure_trajectories_exported'],1)
 def test_rescue_exports_all_gates_for_both_rounds(self):
  a,states=self.fixture();self.fail(a,'initial',0,['response']);s,out=self.run_diag(a,states);self.assertEqual(s['rescued_ids_both_native_rounds'],[IDS[0]]);self.assertEqual(s['full_native_outcomes_by_round']['0']['rescued'],1);rr=clean_json((out/'RESCUED_AND_DESTROYED_FULL_GATE_RECORDS.json').read_bytes());self.assertEqual(len(rr),2);self.assertEqual([r['round'] for r in rr],[0,1]);self.assertTrue(all(r['final_native_qualification']['complete_pass'] for r in rr))
 def test_destroyed_state_never_dropped(self):
  a,states=self.fixture();self.fail(a,'final',7,['KL']);s,out=self.run_diag(a,states);self.assertEqual(s['destroyed_ids_both_native_rounds'],[IDS[7]]);self.assertEqual(s['full_native_outcomes_by_round']['0']['destroyed'],1);self.assertEqual(len(clean_json((out/'ALL168_PAIRED_NATIVE_STATE_GATES.json').read_bytes())),168)
 def test_round_disagreement_blocks_consistent_rescue_claim(self):
  a,states=self.fixture();self.fail(a,'initial',0,['response']);a['native']['native_final1'][0]['qualification']=dict(complete_pass=False,failed_gates=['response'])
  with self.assertRaises(ValueError):self.run_diag(a,states)
 def test_missing_failure_trajectory_blocks(self):
  a,states=self.fixture();self.fail(a,'final',0,['response'])
  with self.assertRaises(ValueError):self.run_diag(a,states[1:])
 def test_permuted_native_identity_blocks(self):
  a,states=self.fixture();a['native']['native_final0'].reverse()
  with self.assertRaises(ValueError):self.run_diag(a,states)
 def test_all_native_pass_keeps_complete_denominator_and_no_failure_rows(self):
  a,states=self.fixture();s,out=self.run_diag(a,states);self.assertEqual(s['remaining_failure_trajectories_exported'],0);self.assertEqual(s['full_native_outcomes_by_round']['0']['both_passed'],84);self.assertEqual(len(clean_json((out/'ALL168_PAIRED_NATIVE_STATE_GATES.json').read_bytes())),168)
 def test_FP64_response_pass_does_not_override_native_failure(self):
  a,states=self.fixture();self.fail(a,'initial',0,['response']);self.fail(a,'final',0,['response']);s,out=self.run_diag(a,states);self.assertEqual(s['rescued_ids_both_native_rounds'],[]);rr=clean_json((out/'FINAL_NATIVE_FAILURE_TRAJECTORIES.json').read_bytes());self.assertEqual(rr[0]['FP64_trajectory']['FP64_response_Linf_by_frame'][-1],.001);self.assertFalse(rr[0]['native_endpoint_records'][0]['final_complete_pass'])
if __name__=='__main__':unittest.main(verbosity=2)
