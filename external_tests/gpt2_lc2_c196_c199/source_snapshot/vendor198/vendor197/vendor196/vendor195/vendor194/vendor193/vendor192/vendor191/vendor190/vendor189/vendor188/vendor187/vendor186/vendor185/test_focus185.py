"""Focus arithmetic controls only; no archive, model or derivative truth claim."""
import copy,tempfile,unittest
from core185 import *
from focus185 import focus_report

def fixture():
 frames=[[dict(state_id=sid,observed=[.001]*6+[0.]*16) for sid in IDS] for _ in range(9)];native={}
 for name in ('native_initial0','native_initial1','native_final0','native_final1'):
  native[name]=[dict(state_id=sid,metrics=dict(observed=[.001]*6+[0.]*16),qualification=dict(complete_pass=True,failed_gates=[])) for sid in IDS]
 return dict(frames=frames,native=native)
def endpoint(a,index,before,after):
 for rnd in (0,1):
  for arm,passed in [('initial',before),('final',after)]:a['native']['native_'+arm+str(rnd)][index]['qualification']=dict(complete_pass=passed,failed_gates=[] if passed else ['response'])
def drift(a,index,values):
 for rr,v in zip(a['frames'],values):rr[index]['observed'][:6]=[v]*6
class Checks(unittest.TestCase):
 def run_report(self,a):
  with tempfile.TemporaryDirectory() as t:return focus_report(a,t)
 def test_rescue_full_native_both_rounds_with_FP64_crossing(self):
  a=fixture();endpoint(a,0,False,True);drift(a,0,[.0021,.00208,.00205,.00201,.00199,.00197,.00195,.00193,.00191]);s=self.run_report(a);self.assertEqual(s['rescued_ids_both_rounds'],[IDS[0]]);self.assertEqual(s['rescued_and_final_FP64_response_pass_ids'],[IDS[0]]);self.assertEqual(s['native_outcomes_by_round']['0']['final_passed'],84);self.assertFalse(s['LC2_closed']);self.assertFalse(s['FP64_full_acceptance_evaluated'])
 def test_native_rescue_with_FP64_failure_retained(self):
  a=fixture();endpoint(a,0,False,True);drift(a,0,[.0021]*9);s=self.run_report(a);self.assertEqual(s['rescued_but_final_FP64_response_fail_ids'],[IDS[0]]);self.assertEqual(s['rescued_and_final_FP64_response_pass_ids'],[])
 def test_safety_entry_of_already_passed_state_is_not_rescue(self):
  a=fixture();drift(a,0,[.0019]*4+[.00179]*5);s=self.run_report(a);self.assertEqual(s['rescued_ids_both_rounds'],[]);self.assertEqual(s['focus_state_ids'],[]);self.assertEqual(s['training_states_retained'],84)
 def test_response_entry_exit_reentry_all_steps_retained(self):
  a=fixture();endpoint(a,0,False,True);drift(a,0,[.0021,.00199,.00201,.00198,.00197,.00196,.00195,.00194,.00193])
  with tempfile.TemporaryDirectory() as t:
   focus_report(a,t);r=clean_json((Path(t)/'RESCUED_DESTROYED_AND_REMAINING_FAILURES.json').read_bytes())[0];self.assertEqual(r['FP64_response_entry_steps'],[1,3]);self.assertEqual(r['FP64_response_exit_steps'],[2]);self.assertEqual(r['FP64_safety_entry_steps'],[])
 def test_destroyed_state_remains_failure_and_in_focus(self):
  a=fixture();endpoint(a,1,True,False);drift(a,1,[.0019]+[.0021]*8);s=self.run_report(a);self.assertEqual(s['destroyed_ids_both_rounds'],[IDS[1]]);self.assertEqual(s['remaining_native_failed_ids_both_rounds'],[IDS[1]])
 def test_still_failed_improvement_not_rescue(self):
  a=fixture();endpoint(a,2,False,False);drift(a,2,np.linspace(.0023,.00201,9));s=self.run_report(a);self.assertEqual(s['rescued_ids_both_rounds'],[]);self.assertEqual(s['remaining_native_failed_ids_both_rounds'],[IDS[2]])
 def test_all_pass_empty_focus_does_not_close_LC2(self):
  s=self.run_report(fixture());self.assertEqual(s['focus_state_ids'],[]);self.assertEqual(s['training_states_retained'],84);self.assertEqual(s['native_endpoint_records'],336);self.assertFalse(s['LC2_closed'])
 def test_native_repeat_outcome_disagreement_blocks(self):
  a=fixture();a['native']['native_final1'][0]['qualification']['complete_pass']=False
  with self.assertRaises(ValueError):self.run_report(a)
 def test_permuted_and_missing_registry_block(self):
  for key in ('native_initial0','native_final1'):
   a=fixture();a['native'][key].reverse()
   with self.assertRaises(ValueError):self.run_report(a)
  a=fixture();a['frames'][0].pop()
  with self.assertRaises(ValueError):self.run_report(a)
 def test_nonfinite_observable_blocks(self):
  a=fixture();a['frames'][0][0]['observed'][0]=float('nan')
  with self.assertRaises(ValueError):self.run_report(a)
if __name__=='__main__':unittest.main(verbosity=2)
