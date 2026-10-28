"""Observed worsening/guard activity remain visible despite positive affine witnesses."""
import copy,tempfile,unittest
from core193 import *
from diagnostics193 import coordinate_diagnostics,guard_diagnostics
from test_math193 import rows
class Checks(unittest.TestCase):
 def fixture(self):
  p,r=rows();pr=recorded_problem(r,p);d=-np.ones_like(p);cert=certificate(pr,d,p+.001*d);frames=[]
  for t in range(9):
   rr=copy.deepcopy(r)
   for row in rr:row['objective_parts']=dict(response_loss=0.,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=.001+t*.0001)
   frames.append(rr)
  return dict(heads=[p.copy() for _ in range(9)],updates=[dict(direction=d,before=p,after=p+.001*d) for _ in range(8)],problems=[pr for _ in range(8)],headcerts=[cert for _ in range(8)],frames=frames,repeat=copy.deepcopy(frames[-1]))
 def test_observed_cap_violation_not_hidden_by_positive_affine_margin(self):
  a=self.fixture();a['frames'][1][0]['observed'][0]=.0023
  with tempfile.TemporaryDirectory() as t:s=coordinate_diagnostics(a,t)
  first=s['all8_observed_coordinate_cap_summaries'][0];self.assertTrue(first['recorded_actual_head_strict_affine_margin']);self.assertFalse(first['observed_all504_training_caps_satisfied']);self.assertEqual(first['observed_cap_violation_coordinates'],1);self.assertEqual(first['observed_cap_violation_state_ids'],[IDS[0]]);self.assertEqual(s['response_coordinate_affine_rows'],4032)
 def test_guard_nonregularizer_activity_in_intermediate_frame_retained(self):
  a=self.fixture();a['frames'][3][0]['objective_parts']['read_loss']=.2;s=guard_diagnostics(a);self.assertFalse(s['all840_nonregularizer_guard_parts_zero']);self.assertFalse(s['observed_guard_change_from_regularizer_only']);self.assertEqual(s['guard_parts_by_frame']['read_loss']['nonzero_state_counts_by_frame'][3],1);self.assertAlmostEqual(s['mean_guard_change_recomposition_residual'],0.)
 def test_guard_regularizer_only_requires_zero_nonregularizer_everywhere(self):
  s=guard_diagnostics(self.fixture());self.assertTrue(s['all840_nonregularizer_guard_parts_zero']);self.assertTrue(s['observed_guard_change_from_regularizer_only']);self.assertAlmostEqual(s['initial_to_final_mean_guard_change'],.0008)
 def test_final_repeat_guard_activity_also_retained(self):
  a=self.fixture();a['repeat'][0]['objective_parts']['KL_loss']=.1;self.assertFalse(guard_diagnostics(a)['all840_nonregularizer_guard_parts_zero'])
if __name__=='__main__':unittest.main(verbosity=2)
