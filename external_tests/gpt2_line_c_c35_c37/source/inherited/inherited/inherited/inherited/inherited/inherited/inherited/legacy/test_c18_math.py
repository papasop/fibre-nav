import unittest
import numpy as np
from joint_calibration import prepare_system,solve_direction,response_prediction,linear_protection

class Tests(unittest.TestCase):
 def data(self):
  r=np.random.default_rng(18031);return r.normal(size=(3,2,7)),r.normal(size=(3,4)),r.normal(size=(3,2))*.005,.07
 def test_Kronecker_matches_explicit_ridge_solution(self):
  J,H,E,scale=self.data();system=prepare_system(J,H,E,scale);D,info=solve_direction(system,1e-6,1e9,1e9)
  A=scale*np.einsum('nkd,nh->nkdh',J,H).reshape(6,28);den=np.linalg.norm(A,axis=1);N=A/den[:,None]
  reference=N.T@np.linalg.solve(N@N.T+1e-6*np.eye(6),-E.reshape(-1)/den)
  np.testing.assert_allclose(D.reshape(-1),reference,atol=1e-10,rtol=1e-8)
 def test_Gram_matches_explicit_operator(self):
  J,H,E,scale=self.data();s=prepare_system(J,H,E,scale);A=scale*np.einsum('nkd,nh->nkdh',J,H).reshape(6,28);N=A/np.linalg.norm(A,axis=1)[:,None]
  reconstructed=s['vectors']@np.diag(s['values'])@s['vectors'].T;np.testing.assert_allclose(reconstructed,N@N.T,atol=1e-12)
 def test_linear_response_prediction_exact(self):
  J,H,E,scale=self.data();s=prepare_system(J,H,E,scale);D,i=solve_direction(s,1e-8,1e9,1e9);inc=scale*(D@H.T).T
  pred=response_prediction(J,E,inc);np.testing.assert_allclose(pred,E+np.einsum('npd,nd->np',J,inc),atol=1e-15)
 def test_independent_equations_reduce_residual(self):
  J,H,E,scale=self.data();D,i=solve_direction(prepare_system(J,H,E,scale),1e-10,1e9,1e9)
  self.assertLess(i['predicted_maximum_drift'],1e-8)
 def test_rank_deficiency_reported_not_certified(self):
  J=np.ones((3,2,4));H=np.ones((3,2));E=np.ones((3,2))*.003;s=prepare_system(J,H,E,.07)
  self.assertEqual(s['info']['numerical_Gram_rank'],1);self.assertFalse(s['info']['rank_certified'])
 def test_zero_J_does_not_invent_response_change(self):
  J=np.zeros((2,2,3));H=np.ones((2,2));E=np.ones((2,2))*.003;D,i=solve_direction(prepare_system(J,H,E,.07),1e-8,.5,.05)
  self.assertTrue(np.array_equal(D,np.zeros_like(D)));np.testing.assert_array_equal(i['predicted_full_step_residuals'],E);self.assertEqual(i['zero_derivative_rows'],4)
 def test_layer_trust_radius(self):
  J,H,E,scale=self.data();D,i=solve_direction(prepare_system(J,H,E,scale),1e-8,1e-5,1e9);self.assertLessEqual(np.linalg.norm(D),1.00001e-5)
 def test_increment_trust_radius(self):
  J,H,E,scale=self.data();D,i=solve_direction(prepare_system(J,H,E,scale),1e-8,1e9,1e-6);self.assertLessEqual(np.max(np.linalg.norm(scale*(D@H.T).T,axis=1)),1.00001e-6)
 def test_nearly_dependent_finite_ridge(self):
  J,H,E,s=self.data();J[1]=J[0]+1e-10;H[1]=H[0]+1e-10;D,i=solve_direction(prepare_system(J,H,E,s),1e-8,.5,.05);self.assertTrue(np.isfinite(D).all())
 def test_negative_ridge_rejected(self):
  with self.assertRaises(ValueError):solve_direction(prepare_system(*self.data()),-1,.5,.05)
 def test_nan_rejected(self):
  J,H,E,s=self.data();J[0,0,0]=np.nan
  with self.assertRaises(ValueError):prepare_system(J,H,E,s)
 def test_shape_mismatch_rejected(self):
  J,H,E,s=self.data()
  with self.assertRaises(ValueError):prepare_system(J,H[:2],E,s)
 def test_zero_scale_rejected(self):
  J,H,E,s=self.data()
  with self.assertRaises(ValueError):prepare_system(J,H,E,0)
 def test_invalid_normalisation_floor(self):
  with self.assertRaises(ValueError):prepare_system(*self.data(),normalisation_floor=0)
 def test_protected_row_cannot_be_sacrificed(self):
  pred=np.array([[.0021,0],[.0001,0]]);self.assertFalse(linear_protection(pred,[0],.002));self.assertTrue(linear_protection(pred,[1],.002))
 def test_actual_displacement_prediction_not_ideal(self):
  J=np.ones((1,1,1));E=np.zeros((1,1));ideal=np.array([[.0001]]);actual=np.zeros((1,1))
  self.assertNotEqual(float(response_prediction(J,E,ideal)[0,0]),float(response_prediction(J,E,actual)[0,0]))
 def test_prediction_shape_rejected(self):
  J,H,E,s=self.data()
  with self.assertRaises(ValueError):response_prediction(J,E,np.zeros((3,8)))
 def test_curved_response_needs_actual_backtracking(self):
  # Synthetic R(theta)=theta^2: local cancellation can overshoot across a curved response.
  theta=.1;E=np.array([[theta**2]]);J=np.array([[[2*theta]]]);H=np.ones((1,1));D,i=solve_direction(prepare_system(J,H,E,1.),1e-10,1e9,1e9)
  step=float(D[0,0]);self.assertLess(i['predicted_maximum_drift'],1e-8)
  actual=(theta+step)**2;self.assertGreater(actual,i['predicted_maximum_drift']);self.assertLess(actual,theta**2)

if __name__=='__main__':unittest.main()
