"""Independent NumPy coefficient-map and chain arithmetic, no model."""
import copy,tempfile,unittest
from core159 import *
from map159 import chain_arithmetic
from fixture159 import context
class Checks(unittest.TestCase):
 def test_explicit_cyclic_sum_matches_independent_coordinate_construction(self):
  c=constants(context()['data']);sid=FOCUS[0];got,p,e=numpy_map(c,sid,.5);u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);want=c[sid+'__raw'].copy()
  for i in range(len(u)):want+=c['V'][i]*(e[0,i]*u[i]+e[1,i]*u[(i-1)%len(u)])
  np.testing.assert_allclose(got,want,rtol=0,atol=1e-17)
 def test_original_commit_and_unrounded_map_remain_distinct(self):
  ctx=context();c=constants(ctx['data']);sid=FOCUS[0];p,_,_=numpy_map(c,sid,0.);old=ctx['data']['points']['zero__'+sid];self.assertGreater(max(abs(p-old.astype(float))),0.);self.assertEqual(old.dtype,np.float32);self.assertEqual(p.dtype,np.float64)
 def test_original_FP32_basis_is_cast_exactly_without_padding(self):
  ctx=context();ctx['data']['d']['V']=ctx['data']['d']['V'].astype(np.float32);c=constants(ctx['data']);np.testing.assert_array_equal(c['V'],ctx['data']['d']['V'].astype(np.float64));self.assertEqual(c['V'].shape,(6,DIM))
 def test_explicit_pullback_and_signed_direction_contraction(self):
  c=constants(context()['data']);sid=FOCUS[0];ge=np.arange(12.).reshape(2,6)-5;gp=c[sid+'__weights'][:,None,None]*ge;ga=float(np.sum(gp*c['deltaP']));r=chain_arithmetic(ga,ge,gp,c,sid);self.assertEqual(r['pullback_vs_direct_Linf'],0.);self.assertEqual(r['explicit_dot_minus_alpha'],0.);wrong=chain_arithmetic(-ga,ge,gp,c,sid);self.assertNotEqual(wrong['explicit_dot_minus_alpha'],0.)
 def test_gradient_shape_and_nonfinite_blocks(self):
  c=constants(context()['data']);sid=FOCUS[0]
  for ge in [np.zeros((6,2)),np.full((2,6),np.nan)]:
   with self.assertRaises(ValueError):chain_arithmetic(0.,ge,np.zeros((3,2,6)),c,sid)
 def test_registry_rejects_unregistered_alpha_and_boolean(self):
  c=constants(context()['data'])
  for a in [True,.125,float('nan')]:
   with self.assertRaises(ValueError):numpy_map(c,FOCUS[0],a)
 def test_nonconvex_route_and_zero_direction_are_rejected(self):
  data=context()['data'];data['d']['weights'][FOCUS[0]]=np.array([1.1,-.1,0.])
  with self.assertRaises(ValueError):constants(data)
  data=context()['data'];data['after']=data['before'].copy()
  with self.assertRaises(ValueError):constants(data)
 def test_capability_budgets_and_scope_are_explicit(self):
  d=design();self.assertFalse(d['LC2_acceptance_thresholds_changed']);self.assertEqual(d['LC2_acceptance_budgets']['response'],.002);self.assertEqual(d['counts']['total_model_forwards'],27);self.assertIn('NOT the exact FP32',d['rounding_boundary']);self.assertIn('fixed',d['routing'])
if __name__=='__main__':unittest.main(verbosity=2)
