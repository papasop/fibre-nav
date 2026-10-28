"""LP, exact-witness, registry, solver-failure and no-model checks."""
import unittest,tempfile,subprocess,sys,json,copy,contextlib,io
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core167 import *
from solver167 import solve
from certificate167 import verify,verify_saved,dec

def vectors(raw,active=None,free=None,guards=None):
 raw=np.asarray(raw,dtype=np.float64);n,m=raw.shape
 return dict(raw=raw,guards=np.zeros_like(raw) if guards is None else np.asarray(guards,dtype=np.float64),free=np.ones(m,dtype=bool) if free is None else np.array(free,dtype=bool),active=np.arange(n,dtype=np.int64) if active is None else np.array(active,dtype=np.int64),norms=np.linalg.norm(raw if free is None else raw*np.asarray(free),axis=1),ids=['s'+str(i) for i in range(n)])
def do(raw,**kw):
 a=vectors(raw,**kw);A=np.array([a['raw'][i]*a['free']/a['norms'][i] if a['norms'][i]>0 else np.zeros(a['raw'].shape[1]) for i in a['active']]).reshape(len(a['active']),a['raw'].shape[1]);r=solve(A,a['free']);return verify(**a,direction=r['direction'],dual=r['dual']),a,r
class Checks(unittest.TestCase):
 def test_feasible_independent_exact_lower_and_upper(self):
  c,a,r=do([[1,0],[0,1]]);self.assertEqual(c['classification'],'RECORDED_GRADIENT_COMMON_DESCENT_CERTIFIED');self.assertEqual(dec(c['optimal_margin_lower']),1);self.assertEqual(dec(c['optimal_margin_upper']),1);self.assertEqual(r['calls'],2)
 def test_exact_opposition_proves_only_recorded_subspace_impossibility(self):
  c,a,r=do([[1,2],[-1,-2]]);self.assertTrue(c['no_strict_recorded_raw_common_descent_proved']);self.assertEqual(dec(c['optimal_margin_upper']),0)
 def test_near_opposition_never_rounded_to_infeasible(self):
  c,a,r=do([[1,0],[-1,1e-7]]);self.assertTrue(c['strict_recorded_raw_common_descent']);self.assertGreater(dec(c['optimal_margin_upper']),0)
 def test_subsolver_tolerance_small_margin_is_unresolved_not_impossible(self):
  c,a,r=do([[1,0],[-1,1e-14]]);self.assertFalse(c['no_strict_recorded_raw_common_descent_proved']);self.assertGreater(dec(c['optimal_margin_upper']),0)
 def test_zero_active_gradient_retained_proves_impossible(self):
  c,a,r=do([[0,0],[1,1]]);self.assertTrue(c['no_strict_recorded_raw_common_descent_proved']);self.assertEqual(c['active_count'],2)
 def test_empty_active_does_not_claim_descent_or_close(self):
  c,a,r=do([[1,0]],active=[]);self.assertEqual(r['calls'],0);self.assertEqual(c['classification'],'NO_ACTIVE_RESPONSE_LOSS_NO_DIRECTION_REQUIRED');self.assertFalse(c['strict_recorded_raw_common_descent'])
 def test_fixed_subspace_not_full_space_claim(self):
  c,a,r=do([[1,1],[-1,1]],free=[True,False]);self.assertTrue(c['no_strict_recorded_raw_common_descent_proved']);self.assertEqual(r['direction'][1],0)
 def test_guard_uphill_recorded_not_hidden(self):
  c,a,r=do([[1,0]],guards=[[-1,0]]);self.assertTrue(c['strict_recorded_raw_common_descent']);self.assertGreater(dec(c['rows'][0]['guard_dot']),0)
 def test_inactive_row_uphill_retained(self):
  c,a,r=do([[1,0],[-1,0]],active=[0]);self.assertEqual(len(c['rows']),2);self.assertGreater(dec(c['rows'][1]['raw_response_dot']),0)
 def test_exact_nonunit_dual_mass_normalized(self):
  a=vectors([[1,0],[0,1]]);c=verify(**a,direction=np.array([-1.,-1.]),dual=np.array([2.,2.]));self.assertEqual(sum(dec(v) for v in c['exact_dual_weights']),1)
 def test_negative_dual_blocks(self):
  a=vectors([[1,0]])
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,0.]),dual=np.array([-1.]))
 def test_zero_dual_blocks(self):
  a=vectors([[1,0]])
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,0.]),dual=np.array([0.]))
 def test_out_of_box_direction_blocks(self):
  a=vectors([[1,0]])
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.01,0.]),dual=np.array([1.]))
 def test_excluded_coordinate_change_blocks(self):
  a=vectors([[1,1]],free=[True,False])
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,.01]),dual=np.array([1.]))
 def test_nonfinite_gradient_blocks(self):
  a=vectors([[1,0]]);a['raw'][0,1]=np.nan
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,0.]),dual=np.array([1.]))
 def test_forged_zero_norm_blocks(self):
  a=vectors([[1,0]]);a['norms'][0]=0
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,0.]),dual=np.array([1.]))
 def test_permuted_active_registry_blocks(self):
  a=vectors([[1,0],[0,1]]);a['active']=a['active'][::-1]
  with self.assertRaises(ValueError):verify(**a,direction=np.array([-1.,-1.]),dual=np.array([.5,.5]))
 def test_forged_saved_certificate_blocks(self):
  c,a,r=do([[1,0]]);arrays={k:v for k,v in a.items() if k!='ids'};arrays.update(direction=r['direction'],dual=r['dual']);bad=copy.deepcopy(c);bad['optimal_margin_lower']['numerator']='999'
  with self.assertRaises(ValueError):verify_saved(arrays,bad,a['ids'])
 def test_exact_serialize_reopen(self):
  c,a,r=do([[1,2],[3,1]]);arrays={k:v for k,v in a.items() if k!='ids'};arrays.update(direction=r['direction'],dual=r['dual'])
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);np.savez(p/'a.npz',**arrays);(p/'c.json').write_text(json.dumps(c))
   with np.load(p/'a.npz',allow_pickle=False) as z:verify_saved({k:z[k] for k in z.files},json.loads((p/'c.json').read_text()),a['ids'])
 def test_unbounded_residual_metadata_serializes_strict_JSON(self):
  c,a,r=do([[1,0],[0,1]])
  text=json.dumps(r['results'],allow_nan=False);self.assertIn('Infinity',text)
 def test_partial_solver_failure_counts_only_actual_attempt(self):
  calls=[]
  def fail(*a,**kw):raise RuntimeError('first solver call failed')
  with self.assertRaises(RuntimeError):solve(np.eye(2),np.ones(2,dtype=bool),fail,on_attempt=lambda i:calls.append(i))
  self.assertEqual(calls,[1])
 def test_solver_failure_no_arbitrary_direction(self):
  from scipy.optimize import OptimizeResult
  r=solve(np.eye(2),np.ones(2,dtype=bool),lambda *a,**k:OptimizeResult(x=None,status=1,success=False,message='fixed budget',fun=None));self.assertIsNone(r['direction']);self.assertEqual(r['calls'],2)
 def test_fake_solver_success_cannot_certify_wrong_signs(self):
  from scipy.optimize import OptimizeResult
  calls=[]
  def fake(*a,**k):
   calls.append(1);return OptimizeResult(x=np.array([1.,1.,100.]) if len(calls)==1 else np.array([.5,.5,100.,100.]),status=0,success=True,fun=-100.,message='forged')
  a=vectors([[1,0],[0,1]]);r=solve(np.eye(2),a['free'],fake);c=verify(**a,direction=r['direction'],dual=r['dual']);self.assertFalse(c['strict_recorded_raw_common_descent']);self.assertFalse(c['no_strict_recorded_raw_common_descent_proved']);self.assertEqual(dec(c['optimal_margin_lower']),-1)
 def test_216_coordinate_problem_exact_witness(self):
  raw=np.zeros((9,216));raw[:,:9]=np.eye(9);c,a,r=do(raw);self.assertTrue(c['strict_recorded_raw_common_descent']);self.assertEqual(c['active_count'],9)
 def test_full84_registry_and_all_positive_loss_rows(self):
  p=np.zeros((3,2,36));rows=[]
  for i,sid in enumerate(IDS):rows.append(dict(state_id=sid,objective_parts=dict(response_loss=1. if i<9 else 0.),response_gradient=np.full_like(p,float(i)).tolist(),guard_gradient=np.zeros_like(p).tolist()))
  a=problem(rows,p);self.assertEqual(a['active'].tolist(),list(range(9)));self.assertEqual(a['A'].shape,(9,216));self.assertEqual(a['norms'][0],0)
  rows[0],rows[1]=rows[1],rows[0]
  with self.assertRaises(ValueError):problem(rows,p)
 def test_whole_hash_rejects_before_parent_loader(self):
  from worker167 import run
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);z=p/'wrong.zip';z.write_bytes(b'wrong');paths={k:z for k in INPUT_KEYS}
   with contextlib.redirect_stdout(io.StringIO()):s=run(paths,p/'out',contract(),lambda *a:(_ for _ in ()).throw(AssertionError('Payload forbidden')))
   self.assertFalse(s['complete']);self.assertEqual(s['linear_program_calls_attempted'],0);self.assertFalse(s['LC2_closed'])
 def test_exact_verifier_imports_no_scipy_or_model(self):
  code="import sys;sys.path.insert(0,sys.argv[1]);import certificate167;assert not any(n in sys.modules for n in ('scipy','torch','transformers'));print('VERIFIER_IMPORT_OK')"
  r=subprocess.run([sys.executable,'-I','-c',code,str(ROOT)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_no_model_guard(self):
  with archive155.prior.no_models():
   with self.assertRaises(ValueError):__import__('torch')
if __name__=='__main__':unittest.main(verbosity=2)
