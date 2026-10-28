"""Registered new-state geometry, novelty, fixed boundaries and contract tests."""
import copy,tempfile,unittest
from core199 import *
from novelty199 import collect
from fixture199 import fixture,history
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.paths,cls.c,cls.ctx=fixture(Path(cls.tmp.name));cls.d=cls.ctx['data']['d'];cls.h=history(cls.paths,cls.ctx,Path(cls.tmp.name));cls.points,cls.bindings=generate(cls.d,cls.h)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_registry48_three_classes16(self):self.assertEqual(len(registry()),48);self.assertEqual(Counter(r['case'] for r in registry()),dict(xor_00=16,xor_01=16,xor_11=16))
 def test_exact12_centers_two_norms_two_directions(self):self.assertEqual(len(centers()),12);self.assertEqual(NORMS,(.125,.375));self.assertEqual(len({r['seed'] for r in registry()}),48)
 def test_requested624_and1429_forwards(self):self.assertEqual(len(requests()),624);self.assertEqual(len({(r['stage'],r['state_id']) for r in requests()}),624);self.assertEqual(design()['counts']['model_forwards'],1429)
 def test_fixed44_originals_include_C198(self):self.assertEqual(len(contract()['pins']),44);self.assertEqual(contract()['pins']['C198']['sha256'],'b0b8d237d18e9005e4996a88c9b6fb54a3cdc9ea94a4cec6003b84d14fd2f7ba')
 def test_head_and_original_thresholds_unchanged(self):self.assertEqual(design()['head_digest'],HEAD_SHA256);self.assertEqual(design()['acceptance_budgets'],parent.design()['acceptance_budgets'])
 def test_generation_repeat_exact_bytes(self):
  p,b=generate(self.d,self.h);self.assertEqual(b,self.bindings)
  for sid in p:np.testing.assert_array_equal(p[sid],self.points[sid])
 def test_actual_radius_and_dtype(self):
  for r in self.bindings:self.assertLessEqual(abs(r['actual_distance']-r['norm']),1e-5);self.assertEqual(self.points[r['state_id']].dtype,np.float32)
 def test_ideal_orthogonal_component(self):self.assertLess(max(r['ideal_shared_row_component_Linf'] for r in self.bindings),1e-10)
 def test_no_input_or_history_mutation(self):
  h=copy.deepcopy(self.h);V=self.d['V'].copy();generate(self.d,h);self.assertEqual(h,self.h);np.testing.assert_array_equal(V,self.d['V'])
 def test_historical_coordinate_collision_blocks_no_resample(self):
  h=copy.deepcopy(self.h);h['hashes'].add(self.bindings[0]['point_sha256'])
  with self.assertRaisesRegex(ValueError,'collision'):generate(self.d,h)
 def test_historical_seed_collision_blocks(self):
  h=copy.deepcopy(self.h);h['seeds'].add(registry()[0]['seed'])
  with self.assertRaises(ValueError):generate(self.d,h)
 def test_historical_ID_collision_blocks(self):
  h=copy.deepcopy(self.h);h['state_ids'].add(registry()[0]['state_id'])
  with self.assertRaises(ValueError):generate(self.d,h)
 def test_reordered_registry_blocks(self):
  with self.assertRaises(ValueError):generate(self.d,self.h,registry()[::-1])
 def test_changed_norm_registry_blocks(self):
  r=registry();r[0]['norm']=.01
  with self.assertRaises(ValueError):generate(self.d,self.h,r)
 def test_duplicate_registry_blocks(self):
  r=registry();r[-1]=r[0]
  with self.assertRaises(ValueError):generate(self.d,self.h,r)
 def test_nonorthonormal_basis_blocks_without_repair(self):
  d=dict(self.d,V=self.d['V']*2)
  with self.assertRaisesRegex(ValueError,'orthonormal'):generate(d,self.h)
 def test_nonfinite_basis_blocks(self):
  V=self.d['V'].copy();V[0,0]=np.nan
  with self.assertRaises(ValueError):generate(dict(self.d,V=V),self.h)
 def test_digest_ID_seed_collection(self):
  h=dict(hashes=set(),state_ids=set(),seeds=set());collect(dict(state_id='x',seed=123,point_sha256='1'*64,children=[dict(theta_sha256='2'*64)]),h);self.assertEqual(h['state_ids'],{'x'});self.assertEqual(h['seeds'],{123});self.assertEqual(h['hashes'],{'1'*64,'2'*64})
 def test_plain_unlabelled_hex_not_treated_as_coordinate(self):
  h=dict(hashes=set(),state_ids=set(),seeds=set());collect(dict(description='1'*64),h);self.assertFalse(h['hashes'])
 def test_generation_needs_no_solver(self):
  with safe.no_solvers():p,b=generate(self.d,self.h)
  self.assertEqual(len(p),48)
if __name__=='__main__':unittest.main(verbosity=2)
