"""Pure NumPy logging regressions: preserve failures and full arrays, never alter replay gates."""
import contextlib,io,tempfile,unittest
from core157 import *
from baseline157 import record_comparison
class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def record(self,a,b):
        with contextlib.redirect_stdout(io.StringIO()):return record_comparison(self.root,'TEST_BASELINE',a,b)
    def test_failed_baseline_is_saved_before_decision(self):
        a=np.zeros((12,6),np.float32);b=a.astype(float);b[2,4]=1e-5;r=self.record(a,b);self.assertFalse(r['passed_original_replay_tolerance']);self.assertEqual(r['maximum_absolute_error'],1e-5);self.assertEqual(r['worst_entries'][0]['index'],[2,4]);self.assertTrue((self.root/'TEST_BASELINE.json').is_file())
    def test_full_actual_and_archived_dtypes_and_values_preserved(self):
        a=np.arange(72,dtype=np.float32).reshape(12,6);b=a.astype(float)+1e-5;self.record(a,b)
        with np.load(self.root/'TEST_BASELINE.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['actual'],a);np.testing.assert_array_equal(z['archived'],b);self.assertEqual(z['actual'].dtype,np.float32);self.assertEqual(z['archived'].dtype,np.float64)
    def test_original_tolerance_not_replaced_by_response_budget(self):
        self.assertEqual(REPLAY_TOL,2e-9);r=self.record(np.array([1e-7],np.float32),np.zeros(1));self.assertFalse(r['passed_original_replay_tolerance']);self.assertFalse(r['threshold_changed']);self.assertFalse(r['references_replaced'])
    def test_equal_arrays_record_zero_error(self):
        a=np.zeros((12,6),np.float32);r=self.record(a,a.astype(float));self.assertTrue(r['passed_original_replay_tolerance']);self.assertEqual(r['number_of_different_entries'],0)
    def test_wrong_shapes_are_saved_and_not_broadcast(self):
        r=self.record(np.zeros((12,6)),np.zeros(6));self.assertFalse(r['shapes_equal']);self.assertFalse(r['passed_original_replay_tolerance']);self.assertIsNone(r['maximum_absolute_error'])
    def test_nonfinite_diagnostics_remain_failed_and_JSON_safe(self):
        r=self.record(np.array([np.nan]),np.array([0.]));self.assertFalse(r['all_values_finite']);self.assertFalse(r['passed_original_replay_tolerance']);self.assertIsNone(r['maximum_absolute_error']);self.assertIsNone(clean_json((self.root/'TEST_BASELINE.json').read_bytes())['maximum_absolute_error'])
if __name__=='__main__':unittest.main(verbosity=2)
