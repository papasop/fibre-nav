"""Check baseline comparison roles and persistence without model loading."""
import io,tempfile,unittest
from core158 import *
from baseline158 import record
class Checks(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_cross_archive_failure_remains_recorded_but_is_not_C158_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):r=record(self.root,'CROSS',np.ones(6,np.float32),np.zeros(6))
        self.assertFalse(r['passed_original_replay_tolerance']);self.assertFalse(r['blocks_C158_diagnostic']);self.assertFalse(r['C157_decision_reclassified']);self.assertTrue((self.root/'CROSS.npz').is_file())
    def test_within_run_role_is_explicitly_blocking(self):
        with contextlib.redirect_stdout(io.StringIO()):r=record(self.root,'REPEAT',np.ones(6),np.zeros(6),'within_run_repeat')
        self.assertTrue(r['blocks_C158_diagnostic']);self.assertFalse(r['passed_original_replay_tolerance']);self.assertEqual(r['original_replay_tolerance'],WITHIN_RUN_TOL)
    def test_original_arrays_and_dtypes_are_preserved(self):
        a=np.ones((12,6),np.float32);b=np.zeros((12,6),np.float64)
        with contextlib.redirect_stdout(io.StringIO()):record(self.root,'ARRAYS',a,b)
        with np.load(self.root/'ARRAYS.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['actual'],a);np.testing.assert_array_equal(z['archived'],b);self.assertEqual(z['actual'].dtype,np.float32)
if __name__=='__main__':unittest.main(verbosity=2)
