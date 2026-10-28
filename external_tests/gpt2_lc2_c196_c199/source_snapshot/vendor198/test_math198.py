"""Frozen map, exact contracts and fresh/archived separation without models."""
import copy,tempfile,unittest
from core198 import *
from fixture198 import fixture,observed_metric
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.paths,cls.c,cls.ctx=fixture(Path(cls.tmp.name));cls.d=cls.ctx['data']['d'];cls.p=cls.ctx['C196_final_head'];cls.prep=np.zeros(DIM,np.float32);cls.capture=np.zeros(3147)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_request_registry_exact672_order(self):
  r=requests();self.assertEqual(len(r),672);self.assertEqual(len({(x['stage'],x['state_id']) for x in r}),672);self.assertEqual(r[0],dict(stage='prepare0',state_id=IDS[0]));self.assertEqual(r[-1],dict(stage='fresh1',state_id=IDS[-1]))
 def test_fixed43_hashes_include_C196_and_C197(self):
  c=contract();self.assertEqual(len(c['pins']),43);self.assertEqual(c['pins']['C197']['sha256'],'6a3cad33ca807b1564e483086d0beec6c709a85a8a2546ee35317dbd054b35b7');self.assertEqual(design()['counts']['model_forwards'],1513)
 def test_wrong_production_head_rejected(self):
  with self.assertRaises(ValueError):check_head(np.zeros((3,2,36)))
 def test_wrong_shape_head_rejected(self):
  with self.assertRaises(ValueError):check_head(np.zeros((3,36)),False)
 def test_nonfinite_head_rejected(self):
  q=self.p.copy();q.flat[0]=np.nan
  with self.assertRaises(ValueError):check_head(q,False)
 def test_out_of_box_head_rejected(self):
  q=self.p.copy();q.flat[0]=.126
  with self.assertRaises(ValueError):check_head(q,False)
 def test_capture_wrong_dimension_rejected(self):
  with self.assertRaises(ValueError):fresh_proposal(self.d,self.p,self.prep,np.zeros(3211))
 def test_nonfinite_capture_rejected(self):
  x=self.capture.copy();x[0]=np.nan
  with self.assertRaises(ValueError):fresh_proposal(self.d,self.p,self.prep,x)
 def test_full3211_tail_recomputed_and_prefix_retained(self):
  r=fresh_proposal(self.d,self.p,self.prep,self.capture);self.assertEqual(r['full'].shape,(3211,));np.testing.assert_array_equal(r['full'][:3147],self.capture);np.testing.assert_array_equal(r['full'][3147:],(r['raw'].astype(float)-self.prep.astype(float))@self.d['control']['delta_sketch'])
 def test_route_recomputed_ignores_archived_weights(self):
  r=fresh_proposal(self.d,self.p,self.prep,self.capture);d=dict(self.d,weights={sid:np.array([1.,0.,0.]) for sid in IDS});q=fresh_proposal(d,self.p,self.prep,self.capture);np.testing.assert_array_equal(r['weights'],q['weights']);np.testing.assert_array_equal(r['point'],q['point'])
 def test_capture_change_changes_raw_point(self):
  x=self.capture.copy();x[0]=1;r=fresh_proposal(self.d,self.p,self.prep,self.capture);q=fresh_proposal(self.d,self.p,self.prep,x);self.assertFalse(np.array_equal(r['raw'],q['raw']))
 def test_left_mix_and_positive_roll_formula(self):
  r=fresh_proposal(self.d,self.p,self.prep,self.capture);w=r['weights'];e=(w[0]*self.p[0]+w[1]*self.p[1])+w[2]*self.p[2];u=self.d['V']@(r['raw'].astype(float)-self.prep.astype(float));want=(r['raw'].astype(float)+self.d['V'].T@(e[0]*u+e[1]*np.r_[u[-1],u[:-1]])).astype(np.float32);np.testing.assert_array_equal(r['point'],want)
 def test_point_hash_change_does_not_pass_repeat(self):
  p=self.ctx['C196_final_points'][IDS[0]];a=observed_metric(IDS[0],p,True);b=copy.deepcopy(a);b['theta_sha256']='0'*64;self.assertFalse(pair_pass(replay_pair(a,b)))
 def test_gate_change_blocks_even_under_observable_tolerance(self):
  p=self.ctx['C196_final_points'][IDS[0]];a=observed_metric(IDS[0],p,True);b=copy.deepcopy(a);b['gates']['response']=False;self.assertFalse(pair_pass(replay_pair(a,b)))
 def test_response_gate_not_training_safety(self):
  p=self.ctx['C196_final_points'][IDS[0]];a=observed_metric(IDS[0],p,True);a=archive155.full_metric(np.r_[np.full(6,.00195),a['observed'][6:]],0.,p,IDS[0].split('__')[0]);self.assertTrue(metric_record(a,p,self.prep,IDS[0].split('__')[0],True)['qualification']['complete_pass'])
 def test_forged_response_gate_recomputed(self):
  p=self.ctx['C196_final_points'][IDS[0]];a=observed_metric(IDS[0],p,True);a['gates']['response']=False
  with self.assertRaises(ValueError):metric_record(a,p,self.prep,IDS[0].split('__')[0],True)
 def test_failed_prep_is_explicit_gate(self):
  p=self.ctx['C196_final_points'][IDS[0]];a=observed_metric(IDS[0],p,True);self.assertIn('preparation',metric_record(a,p,self.prep,IDS[0].split('__')[0],False)['qualification']['failed_gates'])
 def test_no_input_array_mutation(self):
  p=self.p.copy();x=self.capture.copy();a=self.prep.copy();fresh_proposal(self.d,p,a,x);np.testing.assert_array_equal(p,self.p);np.testing.assert_array_equal(x,self.capture);np.testing.assert_array_equal(a,self.prep)
 def test_no_solver_needed_by_map(self):
  with safe.no_solvers():r=fresh_proposal(self.d,self.p,self.prep,self.capture)
  self.assertEqual(r['point'].dtype,np.float32)
if __name__=='__main__':unittest.main(verbosity=2)
