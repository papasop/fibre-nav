"""Fixed schedule, independent arithmetic, denominator and tamper checks."""
import copy,tempfile,unittest
from core161 import *
from fixture161 import context
class Checks(unittest.TestCase):
 def test_exact1092_registry_and_all84_each_batch(self):
  rr=requests();self.assertEqual(len(rr),1092);self.assertEqual(len([r for r in rr if r['stage']=='training']),672)
  for t in range(8):self.assertEqual([r['state_id'] for r in rr if r['stage']=='training' and r['step']==t],[s['state_id'] for s in registry()])
  self.assertFalse(any('xor_10' in r['state_id'] for r in rr));self.assertEqual(STEPS,8);self.assertEqual(RATE,1e-5)
 def test_first_Adam_sign_scale_independent_formula(self):
  p=np.zeros((3,2,6));g=np.linspace(-2,2,36).reshape(p.shape);z=np.zeros_like(p);a,m,v,d=update(p,g,z,z,1);gc=g/max(1.,np.linalg.norm(g));want=-RATE*gc/(abs(gc)+1e-8);np.testing.assert_allclose(a,want,rtol=1e-14,atol=1e-18);self.assertLessEqual(d['clipped_gradient_L2'],1.+1e-14)
 def test_noop_gradient_and_inputs_not_mutated(self):
  p=np.full((3,2,6),.01);g=np.zeros_like(p);m=g.copy();v=g.copy();old=[a.copy() for a in (p,g,m,v)];a,_,_,_=update(p,g,m,v,1);np.testing.assert_array_equal(a,p)
  for x,y in zip((p,g,m,v),old):np.testing.assert_array_equal(x,y)
 def test_wrong_step_nonfinite_and_negative_moment_block(self):
  p=np.zeros((3,2,6))
  for step in (0,9,True):
   with self.assertRaises(ValueError):update(p,p,p,p,step)
  for g,v in [(p+np.nan,p),(p,p-1.)]:
   with self.assertRaises(ValueError):update(p,g,p,v,1)
 def test_box_projection_retained_in_history(self):
  p=np.full((3,2,6),.125);new,_,_,d=update(p,-np.ones_like(p),p*0,p*0,1);np.testing.assert_array_equal(new,p);self.assertEqual(d['projected_coordinates'],36)
 def test_exact_max_ties_full_denominator(self):
  rows=[dict(state_id=s['state_id'],objective_parts=dict(response_loss=2. if i<2 else 0.,regularizer=0.),response_gradient=np.ones((3,2,6)).tolist(),guard_gradient=np.zeros((3,2,6)).tolist()) for i,s in enumerate(registry())];g,a=aggregate(rows,(3,2,6));self.assertEqual(a['exact_max_tie_count'],2);self.assertAlmostEqual(a['response_weights'][0],.45+.1/84);np.testing.assert_allclose(g,1.)
  with self.assertRaises(ValueError):aggregate(rows[:-1],(3,2,6))
 def test_initial_head_is_exact_parent_before(self):
  ctx=context();c=previous.constants(ctx['data']);np.testing.assert_array_equal(c['P0'],ctx['data']['before']);self.assertTrue(np.any(c['P0']!=ctx['data']['after']))
 def test_frozen_head_and_full84_point_inventory(self):
  c=previous.constants(context()['data'])
  with tempfile.TemporaryDirectory() as t:
   pts,native,r=seal_head(c,c['P0'],Path(t)/'head','fixture',True);self.assertEqual(len(pts),84);self.assertEqual(len(native),84);self.assertEqual(r['head_digest'],parameter_hash(c['P0']));bad=c['P0'].copy();bad[-1,-1,-1]+=.001;self.assertNotEqual(parameter_hash(bad),r['head_digest'])
 def test_original_FP32_map_and_no_input_mutation(self):
  c=previous.constants(context()['data']);ids={n:array_digest(a) for n,a in c.items()};sid=registry()[0]['state_id'];pt,e=mapped(c,c['P0'],sid);q=arithmetic.proposal(c[sid+'__prepared'].astype(np.float32),c[sid+'__raw'].astype(np.float32),c['V'],e);np.testing.assert_array_equal(q,pt.astype(np.float32));self.assertEqual(ids,{n:array_digest(a) for n,a in c.items()})
 def test_protected_response_gate_not_relaxed(self):
  sid=registry()[0]['state_id'];pt=np.zeros(DIM,np.float32);case=sid.split('__')[0];x,y=int(case[-2]),int(case[-1]);o=np.r_[np.full(6,.002001),(2*np.tile([x,y,1,x^y],4)-1)*.1];m=archive155.full_metric(o,0.,pt,case);q=archive155.endpoint_gates(m,case,pt,pt,True);self.assertFalse(q['complete_pass']);self.assertIn('response',q['failed_gates'])
if __name__=='__main__':unittest.main(verbosity=2)
