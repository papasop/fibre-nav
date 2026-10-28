"""Precision differences are recorded without becoming new native acceptance gates."""
import copy,tempfile,unittest
from core186 import *
from endpoint186 import precision_margins
class Checks(unittest.TestCase):
 def fixture(self):
  frames=[dict(state_id=sid,observed=[.001]*22) for sid in IDS];native={}
  for arm in ('initial','final'):
   for rnd in (0,1):native['native_'+arm+str(rnd)]=[dict(state_id=sid,metrics=dict(observed=[.001]*22),qualification=dict(complete_pass=True,failed_gates=[])) for sid in IDS]
  return native,copy.deepcopy(frames),copy.deepcopy(frames)
 def run_diag(self,n,i,f):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);s=precision_margins(n,i,f,t.name);return s,Path(t.name)
 def test_native_pass_and_FP64_boundary_failure_kept_distinct(self):
  n,i,f=self.fixture();f[0]['observed'][0]=.002001;s,out=self.run_diag(n,i,f);r=s['by_head']['final']['0'];self.assertEqual(r['native_full_passed'],84);self.assertEqual(r['FP64_response_only_passed'],83);self.assertEqual(r['response_gate_disagreements'],1);self.assertEqual(r['disagreement_state_ids'],[IDS[0]]);self.assertFalse(s['FP64_full_acceptance_evaluated']);self.assertTrue(s['native_full_acceptance_unchanged'])
 def test_native_failure_not_overridden_by_FP64_pass(self):
  n,i,f=self.fixture()
  for rnd in (0,1):n['native_final'+str(rnd)][0]['metrics']['observed'][0]=.00201;n['native_final'+str(rnd)][0]['qualification']=dict(complete_pass=False,failed_gates=['response'])
  s,out=self.run_diag(n,i,f);self.assertEqual(s['by_head']['final']['0']['native_full_passed'],83);self.assertEqual(s['by_head']['final']['0']['FP64_response_only_passed'],84);self.assertEqual(s['by_head']['final']['0']['response_gate_disagreements'],1)
 def test_other_native_gate_failure_not_counted_full_pass(self):
  n,i,f=self.fixture();n['native_final0'][0]['qualification']=dict(complete_pass=False,failed_gates=['KL']);s,out=self.run_diag(n,i,f);r=s['by_head']['final']['0'];self.assertEqual(r['native_response_only_passed'],84);self.assertEqual(r['native_full_passed'],83);self.assertEqual(r['native_full_pass_and_FP64_response_pass'],83)
 def test_exact_response_boundary_is_inclusive_no_new_margin(self):
  n,i,f=self.fixture();f[0]['observed'][0]=.002;n['native_final0'][0]['metrics']['observed'][0]=.002;s,out=self.run_diag(n,i,f);r=s['by_head']['final']['0'];self.assertEqual(r['FP64_response_only_passed'],84);self.assertEqual(r['response_gate_disagreements'],0);self.assertEqual(r['minimum_FP64_response_slack'],0.)
 def test_all336_rows_and_negative_slacks_exported(self):
  n,i,f=self.fixture();f[0]['observed'][0]=.003;s,out=self.run_diag(n,i,f);rows=clean_json((out/'ALL336_ENDPOINT_PRECISION_MARGINS.json').read_bytes());self.assertEqual(len(rows),336);r=next(r for r in rows if r['head']=='final' and r['state_id']==IDS[0]);self.assertLess(r['FP64_response_slack'],0);self.assertEqual(s['additional_model_forwards'],0);self.assertFalse(s['thresholds_changed'])
 def test_state_permutation_and_missing_state_block(self):
  n,i,f=self.fixture()
  with self.assertRaises(ValueError):self.run_diag(n,i,f[::-1])
  with self.assertRaises(ValueError):self.run_diag(n,i,f[:-1])
 def test_nonfinite_and_wrong_shape_block(self):
  n,i,f=self.fixture();f[0]['observed'][0]=float('nan')
  with self.assertRaises(ValueError):self.run_diag(n,i,f)
  n,i,f=self.fixture();f[0]['observed']=[.001]*21
  with self.assertRaises(ValueError):self.run_diag(n,i,f)
if __name__=='__main__':unittest.main(verbosity=2)
