"""Direction algebra, exact registry, bounded probes, and interpretation controls."""
import copy,unittest
from core165 import *
def rows():
 p=np.zeros((3,2,3));rr=[]
 for i,sid in enumerate(IDS):rr.append(dict(state_id=sid,objective_parts=dict(response_loss=0.,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_gradient=p.tolist(),guard_gradient=p.tolist()))
 return p,rr
class Checks(unittest.TestCase):
 def test_agreeing_active_gradients_common_descent(self):
  p,r=rows()
  for i in (0,1,2):r[i]['objective_parts']['response_loss']=1.;r[i]['response_gradient']=(np.ones_like(p)*(i+1)).tolist()
  d,s,h=direction(r,p);self.assertTrue(s['computed_strict_common_response_descent']);self.assertEqual(len(h),4096);np.testing.assert_array_equal(d,-np.ones_like(p));self.assertEqual(s['active_count'],3)
 def test_opposing_gradients_do_not_establish_global_infeasibility(self):
  p,r=rows()
  for i,sgn in [(0,1),(1,-1)]:r[i]['objective_parts']['response_loss']=1.;r[i]['response_gradient']=(sgn*np.ones_like(p)).tolist()
  d,s,h=direction(r,p);self.assertFalse(s['computed_strict_common_response_descent']);self.assertTrue(s['direction_zero']);self.assertTrue(s['no_infeasibility_or_finite_acceptance_claim']);self.assertEqual(len(h),4096)
 def test_zero_projected_gradient_active_row_retained(self):
  p,r=rows();r[0]['objective_parts']['response_loss']=1.;r[1]['objective_parts']['response_loss']=1.;r[1]['response_gradient']=np.ones_like(p).tolist();d,s,_=direction(r,p);self.assertIn(IDS[0],s['zero_projected_gradient_state_ids']);self.assertFalse(s['computed_strict_common_response_descent'])
 def test_no_active_response_loss_has_zero_registered_control(self):
  p,r=rows();d,s,h=direction(r,p);np.testing.assert_array_equal(d,0.);self.assertEqual(s['active_count'],0);self.assertEqual(h,[]);self.assertEqual(len(probe_heads(p,d)),6)
 def test_all84_positive_loss_rows_used_not_only_five_failures(self):
  p,r=rows()
  for x in r:x['objective_parts']['response_loss']=1.;x['response_gradient']=np.ones_like(p).tolist()
  _,s,_=direction(r,p);self.assertEqual(s['active_state_ids'],IDS)
 def test_near_box_boundary_coordinates_excluded_for_both_signs(self):
  p,r=rows();p[0,0,0]=.125;r[0]['objective_parts']['response_loss']=1.;r[0]['response_gradient']=np.ones_like(p).tolist();d,s,_=direction(r,p);self.assertEqual(d[0,0,0],0.);self.assertEqual(s['free_coefficient_coordinates'],p.size-1)
  for q in probe_heads(p,d).values():self.assertLessEqual(np.max(abs(q)),.125)
 def test_declared_three_symmetric_steps_exact_without_selection(self):
  self.assertEqual([a['alpha'] for a in ARMS],[-1e-6,1e-6,-5e-6,5e-6,-1e-5,1e-5]);self.assertFalse(design()['LC2_closed']);self.assertIn('No head/step selection',design()['scope'])
 def test_registry_full1848_order_and_denominator(self):
  q=requests();self.assertEqual(len(q),1848)
  for stage,count in [('anchor_gradient',84),('FP64_probe',504),('FP64_anchor_repeat',84),('native_probe0',504),('native_probe1',504)]:self.assertEqual(sum(r['stage']==stage for r in q),count)
  self.assertTrue(all(r['state_id'].split('__')[0] in ('xor_00','xor_01','xor_11') for r in q))
 def test_missing_and_permuted_state_rows_block(self):
  p,r=rows()
  with self.assertRaises(ValueError):direction(r[:-1],p)
  r[:2]=r[:2][::-1]
  with self.assertRaises(ValueError):direction(r,p)
 def test_nonfinite_gradient_and_loss_block(self):
  p,r=rows();r[0]['response_gradient'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):direction(r,p)
  p,r=rows();r[0]['objective_parts']['response_loss']=float('inf')
  with self.assertRaises(ValueError):direction(r,p)
 def test_wrong_head_shape_and_dtype_block(self):
  p,r=rows()
  with self.assertRaises(ValueError):direction(r,p.astype(np.float32))
  with self.assertRaises(ValueError):direction(r,np.zeros((3,6)))
 def test_probe_would_cross_box_rejected_without_clipping(self):
  p=np.full((3,2,3),.125)
  with self.assertRaises(ValueError):probe_heads(p,np.ones_like(p))
 def test_normalized_mixture_weights_and_signed_direction_dot(self):
  p,r=rows();g=np.ones_like(p);g[0,0,0]=2
  for i in range(3):r[i]['objective_parts']['response_loss']=.1;r[i]['response_gradient']=(g*(i+1)).tolist()
  d,s,_=direction(r,p);self.assertAlmostEqual(sum(s['normalization_weights']),1.);self.assertAlmostEqual(np.max(abs(d)),1.);self.assertAlmostEqual(s['all84_response_gradient_dot_direction'][0],float(np.sum(g*d)))
if __name__=='__main__':unittest.main(verbosity=2)
