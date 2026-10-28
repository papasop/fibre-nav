"""Smooth objective, weights, gradients, schedule and numerical arithmetic tests."""
import unittest
from core163 import *
def rows(R):
 return [dict(state_id=sid,objective_parts=dict(response_loss=float(r),read_loss=.01,KL_loss=0.,radius_loss=0.,regularizer=.001),response_gradient=np.full((3,2,6),i/84).tolist(),guard_gradient=np.full((3,2,6),.02).tolist()) for i,(sid,r) in enumerate(zip(IDS,R))]
class Checks(unittest.TestCase):
 def test_zero_and_uniform_risk_has_uniform_weights(self):
  for value in (0.,.03):
   a=objective(rows(np.full(84,value)));self.assertAlmostEqual(a['smooth_response_risk'],value);np.testing.assert_allclose(a['response_weights'],1/84);self.assertAlmostEqual(a['objective'],value+.011)
 def test_smooth_risk_between_mean_and_max_and_old_score_retained(self):
  R=np.linspace(0,.1,84);a=objective(rows(R));self.assertGreaterEqual(a['smooth_response_risk'],R.mean());self.assertLessEqual(a['smooth_response_risk'],R.max());self.assertAlmostEqual(a['original_hardmax_objective'],.9*R.max()+.1*R.mean()+.011);self.assertAlmostEqual(sum(a['response_weights']),1.);self.assertGreater(a['response_weights'][-1],a['response_weights'][0])
 def test_stable_shifted_exponential_extreme_losses(self):
  a=objective(rows(np.r_[np.zeros(83),10000.]));self.assertTrue(np.isfinite(a['objective']));self.assertTrue(np.isfinite(a['response_weights']).all());self.assertAlmostEqual(a['response_weights'][-1],.9+.1/84)
 def test_weighted_coefficient_gradient_matches_independent_loop(self):
  rr=rows(np.linspace(0,.1,84));g,a=aggregate(rr,(3,2,6));np.testing.assert_allclose(g,independent_gradient(rr,(3,2,6)),rtol=1e-14,atol=1e-16);self.assertAlmostEqual(g[0,0,0],np.dot(a['response_weights'],np.arange(84)/84)+.02)
 def test_full_scalar_smooth_objective_directional_finite_difference(self):
  x=np.linspace(.01,.2,84);d=np.cos(np.arange(84));h=1e-6;a=objective(rows(x*x));pred=np.dot(a['response_weights'],2*x*d);fd=(objective(rows((x+h*d)**2))['objective']-objective(rows((x-h*d)**2))['objective'])/(2*h);self.assertAlmostEqual(fd,pred,delta=1e-9)
 def test_missing_or_permuted_state_blocks(self):
  rr=rows(np.zeros(84))
  with self.assertRaises(ValueError):objective(rr[:-1])
  rr[0],rr[1]=rr[1],rr[0]
  with self.assertRaises(ValueError):objective(rr)
 def test_nonfinite_or_negative_recorded_loss_blocks(self):
  for value in (float('nan'),-1.):
   rr=rows(np.zeros(84));rr[0]['objective_parts']['response_loss']=value
   with self.assertRaises(ValueError):objective(rr)
 def test_wrong_gradient_shape_and_nonfinite_blocks(self):
  for value in ([[1.]],np.full((3,2,6),float('nan')).tolist()):
   rr=rows(np.zeros(84));rr[0]['response_gradient']=value
   with self.assertRaises(ValueError):aggregate(rr,(3,2,6))
 def test_fixed8_registry_and_no_excluded_states(self):
  self.assertEqual(len(requests()),1092);self.assertEqual(STEPS,8);self.assertEqual(TAU,.01);self.assertEqual(RATE,1e-5);self.assertFalse(any('xor_10' in s for s in IDS));self.assertEqual(design()['counts']['actual_FP64_point_tensors'],756)
 def test_Adam_first_step_sign_scale_and_zero_moments(self):
  p=np.zeros((3,2,6));g=np.linspace(-3,3,36).reshape(p.shape);z=np.zeros_like(p);new,_,_,diag=update(p,g,z,z,1);gc=g/max(1.,np.linalg.norm(g));np.testing.assert_allclose(new,-1e-5*gc/(abs(gc)+1e-8),rtol=1e-14,atol=1e-18);self.assertLessEqual(diag['clipped_gradient_L2'],1.+1e-14)
 def test_Adam_fixed_box_and_step_range(self):
  p=np.full((3,2,6),.125);z=np.zeros_like(p);np.testing.assert_array_equal(update(p,-np.ones_like(p),z,z,1)[0],p)
  for step in (0,9,True):
   with self.assertRaises(ValueError):update(p,z,z,z,step)
 def test_hardmax_and_smooth_values_are_not_conflated(self):
  a=objective(rows(np.r_[np.zeros(83),.1]));self.assertLess(a['objective'],a['original_hardmax_objective']);self.assertEqual(a['hardmax_maximizing_state_ids'],[IDS[-1]])
if __name__=='__main__':unittest.main(verbosity=2)
