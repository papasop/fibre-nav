"""Independent unit checks for omitted constraints, units, ties and binary64 endpoints."""
import copy,unittest
from core186 import *
SAFETY=source174.SAFETY
ETA=source174.ETA
chain_check=source174.loss_chain_check
certificate=exact_certificate
panel=source174.step_panel
H=source174.H
ARM_NAMES=source174.ARM_NAMES
def rows(active=True):
 p=np.zeros((3,2,6));rows=[]
 for i,sid in enumerate(IDS):
  r=np.full(6,.0022 if active and i<5 else .001);J=np.zeros((6,)+p.shape)
  for j in range(6):J[j].ravel()[j]=1.
  m=float(max(abs(r)));top=abs(r)==m;g=2*max(m/SAFETY-1,0)/SAFETY*np.mean(J[top]*np.sign(r[top]).reshape((-1,1,1,1)),axis=0)
  rows.append(dict(state_id=sid,observed=np.r_[r,np.zeros(16)].tolist(),response_coordinate_gradients=J.tolist(),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
 return p,rows
class Checks(unittest.TestCase):
 def test_all1008_signed_rows_include_inactive_states(self):
  p,r=rows();pr=problem(r,p);self.assertEqual(pr['A'].shape,(1008,36));self.assertEqual(len(set(map(tuple,pr['registry']))),1008);self.assertEqual(pr['registry'][-1].tolist(),[83,5,1]);self.assertEqual(pr['active'].tolist(),list(range(5)))
 def test_sign_scale_caps_not_native_gate(self):
  p,r=rows();pr=problem(r,p);self.assertEqual(pr['caps'][83],.0018);self.assertEqual(pr['caps'][0],.0022);self.assertAlmostEqual(pr['A'][1,0],ETA/SAFETY);self.assertAlmostEqual(pr['A'][0,0],-ETA/SAFETY);self.assertEqual(pr['rhs'][1],0.)
 def test_inactive_coordinate_rejects_direction_accepted_by_active_rows(self):
  p,r=rows();J=np.asarray(r[-1]['response_coordinate_gradients']);J[5].ravel()[0]=-100.;r[-1]['response_coordinate_gradients']=J.tolist();r[-1]['observed'][5]=.00179;pr=problem(r,p);c=certificate(pr,-np.ones_like(p));self.assertFalse(c['strict_recorded_active_margin']);bad=[x for x in c['all1008_signed_rows'] if int(x['normalized_recorded_cap_slack']['numerator'])<0];self.assertTrue(any(x['state_id']==IDS[-1] and x['coordinate']==5 for x in bad))
 def test_nonmax_coordinate_of_active_state_not_omitted(self):
  p,r=rows();J=np.asarray(r[0]['response_coordinate_gradients']);J[5].ravel()[0]=-100.;r[0]['response_coordinate_gradients']=J.tolist();r[0]['observed'][5]=.00219;r[0]['response_gradient']=(2*(.0022/SAFETY-1)/SAFETY*np.mean(J[:5],axis=0)).tolist();self.assertFalse(certificate(problem(r,p),-np.ones_like(p))['strict_recorded_active_margin'])
 def test_scalar_gradient_units_cannot_change(self):
  p,r=rows();r[0]['response_coordinate_gradients']=(np.asarray(r[0]['response_coordinate_gradients'])/SAFETY).tolist()
  with self.assertRaises(ValueError):problem(r,p)
 def test_signed_exact_max_ties_average_both_branches(self):
  p,r=rows();r[0]['observed'][:6]=[.0022,-.0022,0.,0.,0.,0.];J=np.asarray(r[0]['response_coordinate_gradients']);r[0]['response_gradient']=(2*(.0022/SAFETY-1)/SAFETY*(J[0]-J[1])/2).tolist();self.assertTrue(chain_check(r[0],p.shape)['within_declared_chain_tolerance']);problem(r,p)
 def test_binary64_endpoint_difference_not_rounded_subtraction(self):
  p,r=rows();p[:]=.01;pr=problem(r,p);d=-np.ones_like(p);q=p+ETA*d;c=certificate(pr,d,q);v=c['all1008_signed_rows'][1]['recorded_linear_signed_response'];want=rational(.0022)+(rational(q.ravel()[0])-rational(p.ravel()[0]));self.assertEqual(v,encoded(want))
 def test_full_panel_free_mask_not_step_specific(self):
  p,r=rows();p.ravel()[0]=.1245;pr=problem(r,p);self.assertFalse(pr['free'][0]);d=np.zeros_like(p);d.ravel()[0]=1.
  with self.assertRaises(ValueError):certificate(pr,d)
 def test_empty_active_has_zero_no_strict_descent(self):
  p,r=rows(False);pr=problem(r,p);c=certificate(pr,p.copy());self.assertFalse(c['strict_recorded_active_margin']);self.assertEqual(c['active_state_ids'],[])
  with self.assertRaises(ValueError):certificate(pr,np.ones_like(p))
 def test_wrong_shape_nonfinite_order_missing_rejected(self):
  p,r=rows()
  for rr in (r[::-1],r[:-1]):
   with self.assertRaises(ValueError):problem(rr,p)
  bad=copy.deepcopy(r);bad[0]['response_coordinate_gradients'][0][0][0][0]=float('nan')
  with self.assertRaises(ValueError):problem(bad,p)
  bad=copy.deepcopy(r);bad[0]['response_coordinate_gradients']=bad[0]['response_coordinate_gradients'][:5]
  with self.assertRaises(ValueError):problem(bad,p)
 def test_direction_dtype_and_bounds_rejected(self):
  p,r=rows();pr=problem(r,p)
  for d in (p.astype(np.float32),np.full_like(p,1.1),np.full_like(p,np.nan)):
   with self.assertRaises(ValueError):certificate(pr,d)
 def test_all_six_actual_heads_and_fractions_reconstruct(self):
  p,r=rows();pr=problem(r,p);d=-np.ones_like(p);hs,pa=panel(p,d);self.assertEqual([a['name'] for a in pa['arms']],ARM_NAMES)
  for n,q in hs.items():self.assertEqual(certificate(pr,d,q)['strict_recorded_active_margin'],n.startswith('plus'))
if __name__=='__main__':unittest.main(verbosity=2)
