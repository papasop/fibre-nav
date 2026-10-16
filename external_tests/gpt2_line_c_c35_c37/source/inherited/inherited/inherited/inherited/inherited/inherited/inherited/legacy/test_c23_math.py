import unittest
import numpy as np
from radius_calibration import reconstruct_direction,scale_direction
from joint_calibration import response_prediction

class CachedDirectionTests(unittest.TestCase):
 def system(self):
  r=np.random.default_rng(23);J=r.normal(size=(15,6,4));H=r.normal(size=(15,3));E=r.normal(size=(15,6))*.003;scale=.07
  return {'J':J,'H':H,'E':E,'scale':scale,'j':J.reshape(90,4),'h':np.repeat(H,6,axis=0),'den':np.ones(90),'info':{'rank_certified':False,'finite_response_preservation_certified':False}}
 def test_cached_coefficients_rebuild_explicit_Kronecker_transpose(self):
  s=self.system();c=np.random.default_rng(24).normal(size=90);D=reconstruct_direction(s,c);expected=sum(s['scale']*c[k]*np.outer(s['j'][k],s['h'][k]) for k in range(90));self.assertTrue(np.allclose(D,expected,rtol=1e-13,atol=1e-13))
 def test_prediction_uses_actual_state_displacements(self):
  s=self.system();c=np.ones(90);D=reconstruct_direction(s,c);delta=s['scale']*(D@s['H'].T).T;actual=response_prediction(s['J'],s['E'],delta);expected=np.array([s['E'][i]+s['J'][i]@delta[i] for i in range(15)]);self.assertTrue(np.allclose(actual,expected))
 def test_all_three_radii_apply_without_new_solver(self):
  s=self.system();D=reconstruct_direction(s,np.ones(90))
  for radius in (.05,.1,.2):
   trial,info=scale_direction(s,D,1e-8,.5,radius);self.assertLessEqual(np.linalg.norm(trial),.5+1e-12);self.assertLessEqual(np.max(np.linalg.norm(s['scale']*(trial@s['H'].T).T,axis=1)),radius+1e-12)
 def test_scaling_does_not_mutate_cached_direction(self):
  s=self.system();D=reconstruct_direction(s,np.ones(90));before=D.copy();scale_direction(s,D,1e-8,.5,.05);self.assertTrue(np.array_equal(D,before))
 def test_same_cached_direction_reconstruction_is_exact(self):
  s=self.system();c=np.ones(90);self.assertTrue(np.array_equal(reconstruct_direction(s,c),reconstruct_direction(s,c)))
 def test_invalid_cached_coefficients_rejected(self):
  s=self.system()
  for c in (np.ones(89),np.full(90,np.nan)):
   with self.assertRaises(ValueError):reconstruct_direction(s,c)
 def test_zero_coefficients_produce_zero_direction(self):self.assertTrue(np.array_equal(reconstruct_direction(self.system(),np.zeros(90)),np.zeros((4,3))))
 def test_tiny_float32_layer_step_can_be_zero(self):
  old=np.ones((4,3),dtype=np.float32);step=np.full((4,3),1e-10,dtype=np.float64);self.assertTrue(np.array_equal((old.astype(np.float64)+step).astype(np.float32),old))
 def test_float32_actual_step_can_differ_from_double_ideal(self):
  old=np.ones((4,3),dtype=np.float32);step=np.full((4,3),1e-7,dtype=np.float64);actual=(old.astype(np.float64)+step).astype(np.float32).astype(np.float64)-old;self.assertGreater(np.max(np.abs(actual-step)),0.)
 def test_norm_and_prediction_are_not_finite_acceptance_certificates(self):
  s=self.system();_,info=scale_direction(s,reconstruct_direction(s,np.ones(90)),1e-8,.5,.05);self.assertFalse(info['rank_certified']);self.assertFalse(info['finite_response_preservation_certified'])

if __name__=='__main__':unittest.main()
