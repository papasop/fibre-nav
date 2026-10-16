import io,unittest
import numpy as np
from test_c18_math import Tests as JointAlgebraTests
from joint_calibration import prepare_system,solve_direction
from radius_calibration import ridge_coefficients,reconstruct_direction,scale_direction

class RadiusTests(unittest.TestCase):
 def data(self):
  r=np.random.default_rng(21031);return prepare_system(r.normal(size=(15,6,7)),r.normal(size=(15,4)),r.normal(size=(15,6))*.003,.07)
 def test_exact_coefficient_reconstruction_matches_original_solver(self):
  s=self.data()
  for ridge in (1e-6,1e-8,1e-10):
   c=ridge_coefficients(s,ridge);D=reconstruct_direction(s,c)
   expected,_=solve_direction(s,ridge,1e100,1e100);np.testing.assert_array_equal(D,expected)
 def test_every_radius_matches_original_solver(self):
  s=self.data()
  for ridge in (1e-6,1e-8,1e-10):
   base=reconstruct_direction(s,ridge_coefficients(s,ridge))
   for radius in (.05,.1,.2):
    D,i=scale_direction(s,base,ridge,.5,radius);expected,j=solve_direction(s,ridge,.5,radius)
    np.testing.assert_array_equal(D,expected);self.assertEqual(i['trust_clip_factor'],j['trust_clip_factor'])
 def test_both_radii_bounds(self):
  s=self.data();base=reconstruct_direction(s,ridge_coefficients(s,1e-8))
  for radius in (.05,.1,.2):
   D,i=scale_direction(s,base,1e-8,.5,radius);self.assertLessEqual(np.linalg.norm(D),.5+1e-12);self.assertLessEqual(np.max(np.linalg.norm(s['scale']*(D@s['H'].T).T,axis=1)),radius+1e-12)
 def test_scaling_does_not_mutate_base_direction(self):
  s=self.data();base=reconstruct_direction(s,ridge_coefficients(s,1e-8));old=base.copy();scale_direction(s,base,1e-8,.5,.1);np.testing.assert_array_equal(base,old)
 def test_persisted_coefficients_have_no_lossy_rounding(self):
  s=self.data();c=ridge_coefficients(s,1e-8);b=io.BytesIO();np.save(b,c);b.seek(0);again=np.load(b);np.testing.assert_array_equal(reconstruct_direction(s,c),reconstruct_direction(s,again))
 def test_bad_coefficient_shape_or_nonfinite_rejected(self):
  s=self.data()
  for c in (np.zeros(89),np.full(90,np.nan)):
   with self.assertRaises(ValueError):reconstruct_direction(s,c)
 def test_invalid_radii_rejected(self):
  s=self.data();D=reconstruct_direction(s,ridge_coefficients(s,1e-8))
  for radius in (0,-1,np.nan):
   with self.assertRaises(ValueError):scale_direction(s,D,1e-8,.5,radius)
 def test_unclipped_prediction_is_recorded_as_diagnostic(self):
  s=self.data();D=reconstruct_direction(s,ridge_coefficients(s,1e-8));_,i=scale_direction(s,D,1e-8,.5,.05)
  self.assertIn('unclipped_linear_predicted_R',i);self.assertFalse(i['finite_response_preservation_certified']);self.assertFalse(i['rank_certified'])
if __name__=='__main__':unittest.main()
