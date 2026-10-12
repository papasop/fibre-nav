import ast,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from c7_protocol import *
from geometry import ridge_correction
ROOT=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_functional_gate_accepts_nine_valid_episodes(self):self.assertTrue(functional_fit_gate([{'episode_pass':True}]*9))
 def test_gate_rejects_short_or_failed_record(self):
  self.assertFalse(functional_fit_gate([{'episode_pass':True}]*8));r=[{'episode_pass':True}]*8+[{'episode_pass':False}];self.assertFalse(functional_fit_gate(r))
 def test_MSE_not_used_as_eligibility(self):
  self.assertNotIn('warm_mse_gate',SETTINGS)
  f=ast.parse((ROOT/'c7_protocol.py').read_text());gate=next(n for n in ast.walk(f) if isinstance(n,ast.FunctionDef) and n.name=='functional_fit_gate')
  self.assertNotIn('mse',ast.unparse(gate))
 def test_fixed_acceptance_thresholds(self):self.assertEqual((SETTINGS['margin_gate'],SETTINGS['response_budget'],SETTINGS['kl_budget']),(.05,.002,.005))
 def test_all_variants_split_together(self):
  for f in CASES:
   tr=set(fold_states(f));te=set(fold_states(f,False));self.assertEqual((len(tr),len(te)),(9,3));self.assertFalse(tr&te)
 def test_arm_order_balanced(self):self.assertEqual(sum(arm_order(i)[0]=='functional' for i in range(4)),2)
 def endpoint(self):return {'bits':[0,1,1,1],'evaluation':{'exact_margins':[.1]*4,'heldout_margins':[.1]*12},'drift':.001,'kl':.003}
 def test_good_endpoint_passes(self):self.assertTrue(episode_gate([0,1,1,0],self.endpoint()))
 def test_no_flip_rejected(self):self.assertFalse(episode_gate([0,1,1,1],self.endpoint()))
 def test_unselected_damage_rejected(self):
  v=self.endpoint();v['bits'][2]=0;self.assertFalse(episode_gate([0,1,1,0],v))
 def test_response_KL_margin_failures_rejected(self):
  for k,val in [('drift',.00201),('kl',.00501),('kl',float('nan'))]:
   v=self.endpoint();v[k]=val;self.assertFalse(episode_gate([0,1,1,0],v))
  v=self.endpoint();v['evaluation']['heldout_margins'][0]=.049;self.assertFalse(episode_gate([0,1,1,0],v))
 def test_subset_denominators_explicit(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertIn("'paired_original_denominator':len(eligible)",s)
  self.assertIn("'paired_refinement_not_run_on_folds'",s);self.assertEqual(matched_counts([])['n'],0)
 def test_full_baseline_records_all_four_heads(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertIn('for fold in CASES:',s);self.assertIn("for sid in c['test_ids']:",s)
  self.assertIn("'frozen_baseline_original_four_inputs'",s)
 def test_eligibility_written_before_excluded_baseline(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertLess(s.index('dump(gate_path,gate)'),s.index("for sid in c['test_ids']:"))
 def test_no_warm_retraining(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertNotIn('def fit_warm',s);self.assertIn("'new_warm_training_steps':0",s)
 def test_C6_decision_is_retained(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertIn("dump(out/'PARENT_C6_DECISION.json',json.loads(c6.read('SUMMARY.json')))",s)
  self.assertFalse(SETTINGS['C6_decision_relabelled'])
 def test_training_only_data_in_refinement(self):
  tree=ast.parse((ROOT/'worker_c7.py').read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='refine')
  self.assertNotIn('test_ids',ast.unparse(f));self.assertIn('pred=head(Xn)',ast.unparse(f).replace(' ',''))
 def test_differentiable_candidate_not_copied(self):
  tree=ast.parse((ROOT/'worker_c7.py').read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='refine')
  calls=[ast.unparse(n.func) for n in ast.walk(f) if isinstance(n,ast.Call)]
  self.assertIn('last_logits',calls);self.assertIn('loss.backward',calls);self.assertNotIn('w.assign_flat',calls)
 def test_matched_heads_and_optimisers(self):
  s=(ROOT/'worker_c7.py').read_text();self.assertIn('if head_hash(head)!=warm_sha',s);self.assertIn("'optimiser_reset_at_stage_boundary':True",s)
 def test_no_teacher_fallback(self):self.assertIn("'teacher_fallback_used':False",(ROOT/'worker_c7.py').read_text())
 def test_loss_scale_not_budget(self):self.assertEqual(SETTINGS['response_surrogate_scale'],.1);self.assertEqual(SETTINGS['response_budget'],.002)
 def test_ridge_linear_identity(self):
  J=np.array([[1.,2.,0.],[0.,0.,.2]]);e=np.array([.3,.2]);l=.01;d,_=ridge_correction(J,e,l)
  np.testing.assert_allclose(e+J@d,l*np.linalg.solve(J@J.T+l*np.eye(2),e),atol=1e-12)
 def zip_bytes(self,entries):
  entries=dict(entries);entries['MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in entries.items()}).encode();buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w') as z:
   for n,b in entries.items():z.writestr(n,b)
  return buf.getvalue()
 def test_source_zip_rejected(self):
  buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w') as z:z.writestr('launcher.py','source')
  with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
   with self.assertRaises(ValueError):verify_open_zip(z)
 def test_path_traversal_rejected(self):
  b=self.zip_bytes({'identity.json':b'{}','../escape':b'x'})
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   with self.assertRaises(ValueError):verify_open_zip(z)
 def test_manifest_hash_corruption_rejected(self):
  b=self.zip_bytes({'identity.json':json.dumps({'protocol':PROTOCOL}).encode(),'data':b'abc'})
  with zipfile.ZipFile(io.BytesIO(b)) as z:entries={n:z.read(n) for n in z.namelist()}
  entries['data']=b'corrupt';buf=io.BytesIO()
  with zipfile.ZipFile(buf,'w') as z:
   for n,b in entries.items():z.writestr(n,b)
  with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
   with self.assertRaises(ValueError):verify_open_zip(z)
 def test_snapshot_ignores_partial_files(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'session/run';out.mkdir(parents=True);(out/'identity.json').write_text(json.dumps({'protocol':PROTOCOL}))
   (out/'state.pt').write_bytes(b'weights');(out/'state.tmp').write_bytes(b'partial');p=snapshot(out)
   with zipfile.ZipFile(p) as z:self.assertIn('state.pt',z.namelist());self.assertNotIn('state.tmp',z.namelist())
 def test_isolated_hf_and_retained_cuda_torch(self):
  s=(ROOT/'launch_c7.py').read_text();self.assertIn('--no-deps',s);self.assertNotIn('torch==',s);self.assertIn('test_c7_torch.py',s)
 def test_dependency_regression_prevented(self):
  for n in ['proposer.py','functional_writer.py','c6_protocol.py','c5_protocol.py','parent_c4_protocol.py']:self.assertTrue((ROOT/n).exists())
if __name__=='__main__':unittest.main()
