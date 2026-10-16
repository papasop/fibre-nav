import io,unittest
import numpy as np
from joint_calibration import prepare_system
from budget_calibration import interval_coefficients,reconstruct_direction,scale_direction
from radius_calibration import ridge_coefficients
from test_c18_math import Tests as JointAlgebraTests

class IntervalTests(unittest.TestCase):
 def system(self,e=.003,a=1.):return prepare_system(np.array([[[a]]]),np.array([[1.]]),np.array([[e]]),1.)
 def test_one_dimensional_solution_matches_analytic_positive(self):
  for ridge in (1e-6,1e-8,1e-10,.1):
   s=self.system();c,i=interval_coefficients(s,ridge,.0018);D=reconstruct_direction(s,c)
   self.assertAlmostEqual(D.item(),-(.003-.0018)/(1+ridge),places=12);self.assertTrue(i['converged']);self.assertLess(i['maximum_dual_KKT_residual'],1e-12)
 def test_one_dimensional_negative(self):
  s=self.system(-.003);c,i=interval_coefficients(s,1e-6,.0018);self.assertAlmostEqual(reconstruct_direction(s,c).item(),.0012/1.000001,places=12)
 def test_inside_interval_is_exact_noop(self):
  for e in (-.0018,-.001,0.,.001,.0018):
   c,i=interval_coefficients(self.system(e),1e-8,.0018);self.assertTrue(np.array_equal(c,np.zeros(1)));self.assertTrue(i['converged'])
 def test_interval_solution_smaller_than_zero_target(self):
  s=self.system();c,_=interval_coefficients(s,1e-8,.0018)
  self.assertLess(np.linalg.norm(reconstruct_direction(s,c)),np.linalg.norm(reconstruct_direction(s,ridge_coefficients(s,1e-8))))
 def test_row_normalisation_matches_scaled_analytic(self):
  for a in (.1,2.,100.):
   s=self.system(.003,a);c,i=interval_coefficients(s,1e-6,.0018)
   self.assertAlmostEqual(reconstruct_direction(s,c).item(),-.0012/(a*1.000001),places=11)
 def test_zero_derivative_reports_remaining_violation_no_certificate(self):
  c,i=interval_coefficients(self.system(.003,0.),1e-6,.0018,100)
  self.assertEqual(reconstruct_direction(self.system(.003,0.),c).item(),0.);self.assertGreater(i['unscaled_solver_linear_maximum_interval_excess'],0.);self.assertFalse(i['linear_feasibility_certified'])
 def test_incompatible_shared_constraints_no_feasibility_claim(self):
  s=prepare_system(np.ones((2,1,1)),np.ones((2,1)),np.array([[.003],[-.003]]),1.)
  c,i=interval_coefficients(s,1e-6,.0018,1000)
  self.assertGreater(i['unscaled_solver_linear_maximum_interval_excess'],.001);self.assertFalse(i['finite_acceptance_certified'])
 def test_general_convex_objective_reduces_and_stationarity(self):
  rng=np.random.default_rng(22);s=prepare_system(rng.normal(size=(4,2,6)),rng.normal(size=(4,3)),rng.normal(size=(4,2))*.004,.07)
  c,i=interval_coefficients(s,1e-3,.0018,8192,1e-9);self.assertLess(i['dual_objective'],0.);self.assertTrue(np.isfinite(c).all());self.assertLess(i['relative_dual_KKT_residual'],1e-7)
 def test_correlated_90_row_system_polish_reduces_stationarity_error(self):
  r=np.random.default_rng(2215);J=np.repeat(r.normal(size=(1,6,20)),15,axis=0)+r.normal(size=(15,6,20))*.01;H=np.repeat(r.normal(size=(1,5)),15,axis=0)+r.normal(size=(15,5))*.01;E=r.normal(size=(15,6))*.004;s=prepare_system(J,H,E,.07)
  for ridge in (1e-6,1e-8,1e-10):
   c,i=interval_coefficients(s,ridge,.0018);self.assertLess(i['relative_dual_KKT_residual'],1e-7);self.assertTrue(np.isfinite(c).all());self.assertFalse(i['finite_acceptance_certified'])
 def test_small_KKT_residual_is_not_an_interval_certificate(self):
  s=prepare_system(np.ones((2,1,1)),np.ones((2,1)),np.array([[.003],[-.003]]),1.)
  c,i=interval_coefficients(s,1e-6,.0018);self.assertLess(i['relative_dual_KKT_residual'],1e-9);self.assertGreater(i['unscaled_solver_linear_maximum_drift'],.002)
 def test_same_input_resume_is_deterministic_and_persistable(self):
  s=self.system();c,i=interval_coefficients(s,1e-8,.0018);d,j=interval_coefficients(s,1e-8,.0018);self.assertTrue(np.array_equal(c,d));self.assertEqual(i,j)
  b=io.BytesIO();np.save(b,c);b.seek(0);self.assertTrue(np.array_equal(np.load(b),c))
 def test_trust_caps_apply_to_interval_direction(self):
  s=self.system(.3);c,_=interval_coefficients(s,1e-8,.0018);base=reconstruct_direction(s,c)
  for radius in (.05,.1,.2):
   D,i=scale_direction(s,base,1e-8,.5,radius);self.assertLessEqual(np.linalg.norm(D),radius+1e-12)
 def test_iteration_limit_reported_honestly(self):
  r=np.random.default_rng(5);s=prepare_system(r.normal(size=(4,2,6)),r.normal(size=(4,3)),r.normal(size=(4,2))*.04,.07)
  c,i=interval_coefficients(s,1e-8,.0018,1,1e-20,0);self.assertEqual(i['iterations'],1);self.assertFalse(i['converged']);self.assertFalse(i['linear_feasibility_certified'])
 def test_invalid_solver_inputs_rejected(self):
  for ridge,budget,it,tol in [(0,.0018,10,1e-8),(1e-8,0,10,1e-8),(1e-8,.0018,0,1e-8),(1e-8,.0018,10,0)]:
   with self.assertRaises(ValueError):interval_coefficients(self.system(),ridge,budget,it,tol)

if __name__=='__main__':unittest.main()
