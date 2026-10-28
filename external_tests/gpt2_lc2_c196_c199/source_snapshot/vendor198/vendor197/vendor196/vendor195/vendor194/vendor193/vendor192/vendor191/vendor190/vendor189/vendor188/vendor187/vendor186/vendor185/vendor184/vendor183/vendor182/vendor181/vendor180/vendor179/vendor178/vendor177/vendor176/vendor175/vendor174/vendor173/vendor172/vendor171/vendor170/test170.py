"""Direction/step/protocol tests on synthetic arrays only."""
import copy,tempfile,unittest,types,contextlib,io
from core170 import *
from solver170 import solve
from worker170 import run
def rows(nactive=9):
 p=np.zeros((3,2,6));rr=[]
 for i,sid in enumerate(IDS):
  g=np.zeros_like(p)
  if i<nactive:g.ravel()[i%p.size]=1.
  rr.append(dict(state_id=sid,objective_parts=dict(response_loss=1. if i<nactive else 0.,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_gradient=g.tolist(),guard_gradient=np.zeros_like(p).tolist()))
 return p,rr
class Checks(unittest.TestCase):
 def test_all84_registry_and_all1092_requests_fixed(self):
  r=requests();self.assertEqual(len(r),1092);self.assertEqual(sum(x['stage']=='training' for x in r),672);self.assertEqual(sum(x['stage'].startswith('native') for x in r),336);self.assertEqual(len(set(tuple(x.items()) for x in r)),1092)
 def test_problem_retains_every_active_row_not_only_native_failures(self):
  p,rr=rows();a=problem(rr,p);self.assertEqual(a['active'].tolist(),list(range(9)));self.assertEqual(a['raw'].shape,(84,36));self.assertEqual(a['guards'].shape,(84,36))
 def test_actual_SciPy_one_primal_direction_has_positive_exact_margin(self):
  p,rr=rows();a=problem(rr,p);r=solve(a);v=verify_direction(a,r['direction']);self.assertEqual(r['calls'],1);self.assertTrue(v['strict_recorded_common_descent']);self.assertGreater(int(v['normalized_margin_lower']['numerator']),0);self.assertEqual(len(v['all84_exact_contractions']),84)
 def test_zero_active_has_no_LP_and_no_arbitrary_direction(self):
  p,rr=rows(0);a=problem(rr,p);r=solve(a,lambda *a,**k:(_ for _ in ()).throw(AssertionError('No LP on empty active set')));v=verify_direction(a,r['direction']);self.assertEqual(r['calls'],0);self.assertFalse(np.any(r['direction']));self.assertIsNone(v['normalized_margin_lower'])
 def test_zero_gradient_active_is_not_certified(self):
  p,rr=rows();rr[0]['response_gradient']=np.zeros_like(p).tolist();a=problem(rr,p);v=verify_direction(a,-np.ones(p.size));self.assertFalse(v['strict_recorded_common_descent'])
 def test_opposing_active_rows_not_falsely_certified(self):
  p,rr=rows(2);rr[1]['response_gradient']=(-np.asarray(rr[0]['response_gradient'])).tolist();a=problem(rr,p);r=solve(a);self.assertFalse(verify_direction(a,r['direction'])['strict_recorded_common_descent'])
 def test_exact_contraction_not_solver_status_decides_sign(self):
  p,rr=rows();a=problem(rr,p);self.assertFalse(verify_direction(a,np.ones(p.size))['strict_recorded_common_descent']);self.assertTrue(verify_direction(a,-np.ones(p.size))['strict_recorded_common_descent'])
 def test_guard_ascent_is_reported_not_mislabeled_preserved(self):
  p,rr=rows();rr[0]['guard_gradient']=(-np.ones_like(p)).tolist();a=problem(rr,p);v=verify_direction(a,-np.ones(p.size));self.assertTrue(v['strict_recorded_common_descent']);self.assertGreater(v['all84_exact_contractions'][0]['guard_dot']['approximate'],0)
 def test_nonfinite_gradient_blocks(self):
  p,rr=rows();rr[0]['response_gradient'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):problem(rr,p)
 def test_missing_state_blocks(self):
  p,rr=rows()
  with self.assertRaises(ValueError):problem(rr[:-1],p)
 def test_state_order_changed_blocks(self):
  p,rr=rows()
  with self.assertRaises(ValueError):problem(rr[::-1],p)
 def test_fixed_free_mask_excludes_near_box_boundary(self):
  p,rr=rows();p.ravel()[0]=.125;pr=problem(rr,p);self.assertFalse(pr['free'][0]);d=np.zeros(p.size);d[0]=-1
  with self.assertRaises(ValueError):verify_direction(pr,d)
 def test_direction_outside_bounds_blocks(self):
  p,rr=rows()
  with self.assertRaises(ValueError):verify_direction(problem(rr,p),np.full(p.size,1.01))
 def test_nonfinite_direction_blocks(self):
  p,rr=rows()
  with self.assertRaises(ValueError):verify_direction(problem(rr,p),np.full(p.size,np.nan))
 def test_fixed_step_sign_scale_no_projection(self):
  p=np.zeros((3,2,6));d=np.linspace(-1,1,p.size).reshape(p.shape);new,r=update(p,d);np.testing.assert_array_equal(new,p+1e-4*d);self.assertEqual(r['step_size'],1e-4);self.assertFalse(r['head_projection_performed'])
 def test_head_box_violation_blocks_without_halving(self):
  p=np.full((3,2,6),.125)
  with self.assertRaises(ValueError):update(p,np.ones_like(p))
 def test_solver_missing_vector_has_no_fallback(self):
  p,rr=rows();calls=[]
  class Result(dict):
   def __getattr__(self,k):return self[k]
  def back(*a,**k):calls.append(1);return Result(x=None,status=1,success=False)
  with self.assertRaises(ValueError):solve(problem(rr,p),back)
  self.assertEqual(len(calls),1)
 def test_solver_uses_fixed_highs_method_and_time_limit(self):
  p,rr=rows();pr=problem(rr,p);seen=[]
  def backend(c,**kw):seen.append(kw);return dict(x=np.r_[-np.ones(p.size),1.],success=True)
  class Result(dict):
   def __getattr__(self,k):return self[k]
  def wrapped(c,**kw):return Result(backend(c,**kw))
  solve(pr,wrapped);self.assertEqual(len(seen),1);self.assertEqual(seen[0]['method'],'highs-ds');self.assertEqual(seen[0]['options']['time_limit'],60.)
 def test_wrong_whole_input_blocks_before_model(self):
  with tempfile.TemporaryDirectory() as t:
   t=Path(t);paths={}
   for k in INPUT_KEYS:q=t/(k+'.zip');q.write_bytes(b'synthetic');paths[k]=q
   c=contract();c.update(production=False,pins={k:dict(path=str(q),sha256=sha(q)) for k,q in paths.items()});c['pins']['C169']['sha256']='0'*64
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,t/'out',c,lambda *a:(_ for _ in ()).throw(AssertionError('Model must not load')))
   self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['LC2_closed'])
if __name__=='__main__':unittest.main(verbosity=2)
