import unittest,inspect
import numpy as np,torch
from c29_math import *
from c29_coverage import coverage
class Tests(unittest.TestCase):
 def setUp(self):self.J=np.eye(6);self.e=np.ones(6)*.003
 def test_linear_exact(self):
  d=np.ones(6)*-.0015;r=paired(self.e,self.e+d,self.J,d);self.assertEqual(r['remainder_infinity'],0.);self.assertTrue(r['predicted_response_pass']);self.assertEqual(r['diagnostic_category'],'PREDICTED_AND_ACTUAL_IMPROVE')
 def test_wrong_direction(self):
  d=np.ones(6)*.001;r=paired(self.e,self.e+d,self.J,d);self.assertEqual(r['diagnostic_category'],'FIRST_ORDER_AND_ACTUAL_WORSEN');self.assertAlmostEqual(r['linear_alignment_with_needed_compensation'],-1.)
 def test_nonlinearity_can_reverse_prediction(self):
  d=np.ones(6)*-.001;r=paired(self.e,np.ones(6)*.004,self.J,d);self.assertEqual(r['diagnostic_category'],'PREDICTED_IMPROVES_ACTUAL_WORSENS');self.assertAlmostEqual(r['remainder_infinity'],.002)
 def test_predicted_pass_actual_fail(self):
  d=np.ones(6)*-.002;r=paired(self.e,np.ones(6)*.004,self.J,d);self.assertTrue(r['predicted_pass_actual_fail'])
 def test_quadratic_half_ratio_four(self):
  d=np.ones(6)*.01;r=paired(self.e,self.e+10*d*d,np.zeros((6,6)),d,{'residual':self.e+10*(d*.5)**2,'delta':d*.5});self.assertAlmostEqual(r['fixed_midpoint']['full_over_half_remainder_ratio'],4.,places=10)
 def test_small_half_remainder_ratio_suppressed(self):
  d=np.ones(6)*-.001;r=paired(self.e,self.e+d,self.J,d,{'residual':self.e+d*.5,'delta':d*.5});self.assertIsNone(r['fixed_midpoint']['full_over_half_remainder_ratio'])
 def test_actual_float32_displacement_not_intended_step(self):
  t=torch.full((6,),1e8,dtype=torch.float32);u=t+.1;d=(u.double()-t.double()).numpy();r=paired(self.e,self.e,self.J,d);self.assertEqual(r['actual_parameter_displacement_norm'],0.)
 def test_zero_alignment_undefined(self):self.assertIsNone(cosine(np.zeros(6),np.ones(6)))
 def test_compensation_alignment_positive(self):self.assertAlmostEqual(cosine(-self.e,-self.e),1.)
 def test_common_mode_ones(self):self.assertAlmostEqual(common_mode_energy(self.e),1.)
 def test_common_mode_contrast_zero(self):self.assertEqual(common_mode_energy([1,-1,0,0,0,0]),0.)
 def test_common_mode_zero_undefined(self):self.assertIsNone(common_mode_energy(np.zeros(6)))
 def test_shape_error_rejected(self):
  with self.assertRaises(ValueError):paired(self.e,self.e,np.zeros((5,6)),np.ones(6))
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):paired(self.e,np.ones(6)*float('nan'),self.J,np.ones(6))
 def test_invalid_budget_rejected(self):
  with self.assertRaises(ValueError):paired(self.e,self.e,self.J,np.zeros(6),budget=0)
 def test_invalid_midpoint_rejected(self):
  with self.assertRaises(ValueError):paired(self.e,self.e,self.J,np.zeros(6),{'residual':self.e,'delta':np.zeros(5)})
 def test_aggregate_retains_all_rows(self):
  a=paired(self.e,self.e-.001,self.J,np.ones(6)*-.001);b=paired(self.e,self.e+.001,self.J,np.ones(6)*.001);s=aggregate([a,b]);self.assertEqual(s['n'],2);self.assertEqual(sum(s['categories'].values()),2)
 def test_no_certificate_claim(self):self.assertFalse(paired(self.e,self.e,self.J,np.zeros(6))['finite_step_certificate'])
 def test_feature_coverage_no_fit(self):
  fm={'center':torch.zeros(10),'basis':torch.eye(10)[:,:8],'scale':torch.ones(8)};tr=np.zeros((19,10));de=np.zeros((2,10));de[0,0]=1;de[1,9]=1;r=coverage(tr,de,fm);self.assertEqual(r[0]['nearest_training_distance_in_frozen_8D'],1.);self.assertEqual(r[1]['outside_basis_fraction'],1.)
 def test_feature_coverage_nonfinite_rejected(self):
  fm={'center':torch.zeros(10),'basis':torch.eye(10)[:,:8],'scale':torch.ones(8)}
  with self.assertRaises(ValueError):coverage(np.zeros((19,10)),np.ones((1,10))*float('nan'),fm)
 def test_diagnostic_math_has_no_optimizer(self):
  code=inspect.getsource(paired)
  for term in ('optimiser','optimizer','backward(','svd('):self.assertNotIn(term,code)
 def test_midpoint_not_endpoint_acceptance(self):
  d=np.zeros(6);r=paired(self.e,self.e,self.J,d,{'residual':self.e,'delta':d});self.assertTrue(r['fixed_midpoint']['not_a_full_endpoint_acceptance_test'])
if __name__=='__main__':unittest.main()
