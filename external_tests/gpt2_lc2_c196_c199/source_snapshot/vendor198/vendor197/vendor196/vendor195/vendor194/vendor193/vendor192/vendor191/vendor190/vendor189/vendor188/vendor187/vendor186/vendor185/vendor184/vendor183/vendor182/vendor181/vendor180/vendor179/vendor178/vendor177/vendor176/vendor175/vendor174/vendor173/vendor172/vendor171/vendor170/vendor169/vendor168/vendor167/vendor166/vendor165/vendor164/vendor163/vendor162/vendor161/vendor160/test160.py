"""Pure objective accounting and protocol invariants; no model."""
import copy,tempfile,unittest
from core160 import *
from fixture160 import context,inputs
class Checks(unittest.TestCase):
 def rows(self):
  shape=(3,2,6);return [dict(state_id=s['state_id'],objective_parts=dict(response_loss=2. if i<2 else 1.,read_loss=.5,KL_loss=.1,radius_loss=.2,regularizer=.01),response_gradient=np.full(shape,float(i+1)).tolist(),guard_gradient=np.full(shape,.5).tolist()) for i,s in enumerate(registry())]
 def test_full84_exact_max_ties_and_gradient_weights(self):
  r=self.rows();g,a=aggregate(r,(3,2,6));self.assertEqual(a['exact_max_tie_count'],2);self.assertAlmostEqual(sum(a['response_weights']),1.);self.assertAlmostEqual(a['response_weights'][0],.1/84+.45);want=sum(w*(i+1)+.5/84 for i,w in enumerate(a['response_weights']));np.testing.assert_allclose(g,want,atol=1e-13,rtol=0)
 def test_missing_failed_state_never_reduces_denominator(self):
  r=self.rows();r.pop()
  with self.assertRaises(ValueError):objective(r)
 def test_permuted_or_duplicate_state_blocks(self):
  for op in ('swap','duplicate'):
   r=self.rows()
   if op=='swap':r[0],r[1]=r[1],r[0]
   else:r[1]['state_id']=r[0]['state_id']
   with self.assertRaises(ValueError):objective(r)
 def test_nonfinite_gradients_rejected(self):
  r=self.rows();r[0]['guard_gradient'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):aggregate(r,(3,2,6))
 def test_negative_losses_rejected(self):
  r=self.rows();r[0]['objective_parts']['response_loss']=-1.
  with self.assertRaises(ValueError):objective(r)
 def test_registry_requests_exact1008_and_five_archives(self):
  rr=requests();self.assertEqual(len(rr),1008);self.assertEqual(Counter(r['stage'] for r in rr),dict(native0=84,AD=84,finite=588,repeat=168,native1=84));self.assertEqual(len(INPUT_KEYS),5)
 def test_frozen84_no_training_and_same_original_gates(self):
  d=design();self.assertEqual(d['new_training_steps'],0);self.assertEqual(d['acceptance_budgets']['response'],.002);self.assertEqual(d['counts']['total_model_forwards'],2019);self.assertEqual(d['counts']['scalar_reverse_calls'],168);self.assertFalse(d['LC2_closed']);self.assertTrue(d['no_FP64_acceptance_score'])
 def test_captured_all84_constants_include_explicit_order(self):
  d=context()['data'];c=constants(d);self.assertEqual(len(c),3+3*84);self.assertEqual(c['P0'].shape,(3,2,6));self.assertTrue(all(s['state_id']+'__weights' in c for s in registry()))
 def test_wrong_whole_hash_blocks_before_payload(self):
  from inputs160 import load
  with tempfile.TemporaryDirectory() as t:
   p,c=inputs(t);c['pins']['C159']['sha256']='0'*64
   with self.assertRaises(ValueError):load(p,Path(t)/'out',c)
if __name__=='__main__':unittest.main(verbosity=2)
