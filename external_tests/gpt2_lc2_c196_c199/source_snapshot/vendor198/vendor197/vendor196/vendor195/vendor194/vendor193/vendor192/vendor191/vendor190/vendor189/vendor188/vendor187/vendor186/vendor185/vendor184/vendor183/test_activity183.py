"""Activity set changes never substitute for full native acceptance."""
import copy,tempfile,unittest
from core183 import *
from activity183 import activity_diagnostics
class Checks(unittest.TestCase):
 def fixture(self,values):
  frames=[]
  for value in values:
   frames.append([dict(state_id=sid,observed=[value if i==0 else .001]*6+[.2]*16,objective_parts=dict(response_loss=max((value if i==0 else .001)/SAFETY-1,0)**2)) for i,sid in enumerate(IDS)])
  native={}
  for arm in ('initial','final'):
   for rnd in (0,1):native['native_'+arm+str(rnd)]=[dict(state_id=sid,qualification=dict(complete_pass=True)) for sid in IDS]
  return dict(frames=frames,problems=[dict(active=np.array([i for i,r in enumerate(ff) if r['objective_parts']['response_loss']>0],dtype=np.int64)) for ff in frames[:8]],native=native)
 def run_diag(self,a):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);return activity_diagnostics(a,t.name),Path(t.name)
 def test_exit_safety_is_not_new_native_rescue(self):
  s,p=self.run_diag(self.fixture([.0019]+[.0017]*8));self.assertEqual(s['initial_active_count'],1);self.assertEqual(s['final_active_count'],0);self.assertEqual(s['initial_active_but_final_inactive_ids'],[IDS[0]]);r=s['states_with_activity_changes'][0];self.assertEqual(r['left_active_steps'],[1]);self.assertTrue(r['native_full_endpoints_by_round']['0']['initial_full_pass']);self.assertTrue(r['native_full_endpoints_by_round']['0']['final_full_pass']);self.assertEqual(s['step_summaries'][0]['crossed_into_FP64_response_budget_ids'],[])
 def test_response_budget_crossing_without_activity_exit(self):
  s,p=self.run_diag(self.fixture([.0021]+[.0019]*8));self.assertEqual(s['initial_active_count'],1);self.assertEqual(s['final_active_count'],1);self.assertEqual(s['step_summaries'][0]['left_active_ids'],[]);self.assertEqual(s['step_summaries'][0]['crossed_into_FP64_response_budget_ids'],[IDS[0]])
 def test_reentry_kept_when_initial_final_counts_equal(self):
  s,p=self.run_diag(self.fixture([.0019,.0017]+[.0019]*7));self.assertEqual(s['initial_active_count'],s['final_active_count']);self.assertEqual(s['step_summaries'][1]['reentered_active_ids'],[IDS[0]]);r=s['states_with_activity_changes'][0];self.assertEqual(r['left_active_steps'],[1]);self.assertEqual(r['reentered_active_steps'],[2])
 def test_first_new_entry_not_reentry(self):
  s,p=self.run_diag(self.fixture([.0017,.0019]+[.0017]*7));self.assertEqual(s['step_summaries'][0]['entered_active_ids'],[IDS[0]]);self.assertEqual(s['step_summaries'][0]['reentered_active_ids'],[]);self.assertEqual(s['step_summaries'][1]['left_active_ids'],[IDS[0]])
 def test_still_failed_improvement_not_activity_exit(self):
  s,p=self.run_diag(self.fixture([.0021]+[.00205]*8));self.assertEqual(s['states_with_activity_changes'],[]);self.assertEqual(s['final_active_count'],1);self.assertEqual(len(clean_json((p/'ALL672_ACTIVITY_TRANSITIONS.json').read_bytes())),672)
 def test_exact_safety_boundary_inclusive(self):
  s,p=self.run_diag(self.fixture([.0019]+[.0018]*8));self.assertEqual(s['final_active_count'],0);self.assertEqual(s['step_summaries'][0]['entered_safety_ids'],[IDS[0]]);self.assertEqual(s['training_safety_budget'],.0018);self.assertEqual(s['original_response_budget'],.002)
 def test_registry_permutation_rejected(self):
  a=self.fixture([.0019]*9);a['frames'][3].reverse()
  with self.assertRaises(ValueError):self.run_diag(a)
 def test_changed_LP_activity_binding_rejected(self):
  a=self.fixture([.0019]*9);a['problems'][0]['active']=np.array([],dtype=np.int64)
  with self.assertRaises(ValueError):self.run_diag(a)
 def test_nonfinite_observation_rejected(self):
  a=self.fixture([.0019]*9);a['frames'][4][0]['observed'][0]=float('nan')
  with self.assertRaises(ValueError):self.run_diag(a)
 def test_negative_loss_rejected(self):
  a=self.fixture([.0019]*9);a['frames'][4][0]['objective_parts']['response_loss']=-1.
  with self.assertRaises(ValueError):self.run_diag(a)
if __name__=='__main__':unittest.main(verbosity=2)
