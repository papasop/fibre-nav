"""Pure NumPy complete workflow, actual-point tamper and independent arithmetic tests."""
import copy,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core164 import *
from audit164 import audit
from worker164 import run
from fixture164 import fixture,changed,json_change,array_change
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c,cls.ctx=fixture(cls.root/'fixture')
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def reject(self,name,edit,rebind=True):
  z=changed(self.paths['C163'],self.root/(self._testMethodName+'.zip'),name,edit,rebind)
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((ValueError,RuntimeError,KeyError)):audit(z,copy.deepcopy(self.ctx),self.root/(self._testMethodName+'_out'),self.c)
 def rows(self,fn):
  def edit(raw):r=[clean_json(x) for x in raw.splitlines()];fn(r);return ('\n'.join(json.dumps(x,allow_nan=True) for x in r)+'\n').encode()
  return edit
 def test_full1092_worker_actual756_points_no_Torch_or_training(self):
  with contextlib.redirect_stdout(io.StringIO()),archive155.prior.no_models(),safe.no_solvers():s=run(self.paths,self.root/'complete',self.c,lambda *a:copy.deepcopy(self.ctx))
  self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['actual_Torch_FP64_points_rehashed'],756);self.assertEqual(len(clean_json((self.root/'complete/C163_arithmetic/ALL756_ACTUAL_FP64_POINT_AUDIT_ROWS.json').read_bytes())),756);self.assertTrue(s['Torch_FP64_forward_point_bytes_independently_rehashed']);self.assertTrue(s['reported_map_errors_independently_recomputed']);self.assertEqual(s['training_records_audited'],672);self.assertEqual(s['training_updates_audited'],8);self.assertEqual(s['native_endpoint_records_audited'],336);self.assertFalse(s['LC2_closed']);self.assertFalse(s['gradients_independently_recomputed']);self.assertFalse(s['chronological_seal_order_independently_proved']);self.assertFalse(s['all_original_parent_archives_independently_bound']);self.assertFalse(any(n in sys.modules for n in ('torch','transformers','peft')));self.assertTrue(s['diagnostics']['all756_response_margin_rows_retained']);self.assertEqual(s['new_model_forwards'],0)
 def test_wrong_whole_hash_before_any_payload(self):
  cc=copy.deepcopy(self.c);cc['pins']['C163']['sha256']='0'*64
  with contextlib.redirect_stdout(io.StringIO()),patch('worker164.audit',side_effect=AssertionError('No payload')):s=run(self.paths,self.root/'wrongwhole',cc,lambda *a:(_ for _ in ()).throw(AssertionError('No loader')))
  self.assertFalse(s['complete'])
 def test_source_change_rebound_manifest(self):self.reject('source/core163.py',lambda b:b+b'\n# changed\n')
 def test_temperature_change_blocks(self):self.reject('DESIGN_FROZEN_BEFORE_MODEL.json',json_change(lambda a:a['optimizer'].__setitem__('rate',1e-4)))
 def test_final_head_change_blocks(self):self.reject('FIXED_FINAL_HEAD_STEP8.npz',array_change(lambda a:a['P'].__setitem__((0,0,0),.1)))
 def test_initial_moments_not_zero_blocks(self):self.reject('INITIAL_C161_STEP8_AND_FRESH_MOMENTS.npz',array_change(lambda a:a['m'].__setitem__((0,0,0),.1)))
 def test_initial_restart_head_change_blocks(self):self.reject('INITIAL_C161_STEP8_AND_FRESH_MOMENTS.npz',array_change(lambda a:a['P'].__setitem__((0,0,0),.1)))
 def test_actual_before_head_change_rebound_seal(self):self.reject('step_01_before/HEAD.npz',array_change(lambda a:a['P'].__setitem__((0,0,0),.1)))
 def test_actual_FP64_point_changed_even_rebound_all_receipts(self):self.reject('step_01_before/ACTUAL_TORCH_FP64_POINTS.npz',array_change(lambda a:a[IDS[0]].__setitem__(0,a[IDS[0]][0]+.001)))
 def test_actual_FP64_point_wrong_dtype_blocks(self):self.reject('step_01_before/ACTUAL_TORCH_FP64_POINTS.npz',array_change(lambda a:a.__setitem__(IDS[0],a[IDS[0]].astype(np.float32))))
 def test_actual_point_binding_forged_blocks(self):self.reject('step_01_before/ACTUAL_TORCH_FP64_POINT_BINDINGS.json',json_change(lambda a:a[0].__setitem__('actual_Torch_FP64_sha256','0'*64)))
 def test_actual_point_counter_forged_blocks(self):self.reject('step_01_before/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',json_change(lambda a:a.__setitem__('model_forwards_so_far',172)))
 def test_actual_point_seal_order_flag_blocks(self):self.reject('step_02_before/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',json_change(lambda a:a.__setitem__('sealed_before_this_batch',False)))
 def test_recorded_actual_point_hash_forgery_blocks(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[84].__setitem__('FP64_point_sha256','0'*64)))
 def test_reported_map_error_independently_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[84].__setitem__('map_Linf',1e-13)))
 def test_actual_FP32_point_change_blocks(self):self.reject('final_step8/PHYSICAL_FP32_POINTS.npz',array_change(lambda a:a[IDS[0]].__setitem__(0,.1)))
 def test_gradient_change_blocks(self):self.reject('UPDATE_STEP_03.npz',array_change(lambda a:a['gradient'].__setitem__((0,0,0),.1)))
 def test_moment_change_blocks(self):self.reject('UPDATE_STEP_04.npz',array_change(lambda a:a['m_after'].__setitem__((0,0,0),.1)))
 def test_update_change_blocks(self):self.reject('UPDATE_STEP_08.npz',array_change(lambda a:a['after'].__setitem__((0,0,0),.1)))
 def test_objective_weight_forgery_recomputed(self):self.reject('ALL8_UPDATE_HISTORY.json',json_change(lambda a:a[0]['aggregation']['response_weights'].__setitem__(0,.9)))
 def test_hardmax_summary_not_replaced_by_smooth(self):self.reject('SUMMARY.json',json_change(lambda a:a.__setitem__('original_hardmax_objective_final',a['final_FP64_objective']['objective'])))
 def test_missing_or_permuted_rows_block(self):
  self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a.pop(84)));self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a.__setitem__(slice(84,86),a[84:86][::-1])))
 def test_nonfinite_gradient_blocks(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[84]['response_gradient'][0][0].__setitem__(0,float('nan'))))
 def test_forged_loss_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[84]['objective_parts'].__setitem__('read_loss',.25)))
 def test_native_gate_forgery_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[0]['qualification'].__setitem__('complete_pass',True)))
 def test_replay_declaration_recomputed(self):self.reject('ALL1092_RECORDS.jsonl',self.rows(lambda a:a[84]['C161_final_FP64_replay'].__setitem__('observed_Linf',.1)))
 def test_fixed_final_cannot_be_intermediate(self):self.reject('FINAL_HEAD_FROZEN_BEFORE_EVALUATION.json',json_change(lambda a:a.__setitem__('step',3)))
 def test_repeat_missing_blocks(self):self.reject('ALL168_NATIVE_REPEAT_CHECKS.json',json_change(lambda a:a.pop()))
 def test_smooth_independent_aggregation_matches_parent_and_extremes(self):
  rows=[dict(state_id=sid,objective_parts=dict(response_loss=10000. if i==0 else .001*i,read_loss=0.,KL_loss=0.,radius_loss=0.,regularizer=0.),response_gradient=np.full((3,2,2),i).tolist(),guard_gradient=np.full((3,2,2),.1*i).tolist()) for i,sid in enumerate(IDS)];g,a=aggregate(rows,(3,2,2));pg,pa=parent.aggregate(rows,(3,2,2));compare(a,pa);close(g,pg,'Independent smooth weight arithmetic',1e-10);self.assertGreater(a['original_hardmax_objective'],a['objective']);self.assertAlmostEqual(sum(a['response_weights']),1.)
 def test_nonzero_saved_Torch_NumPy_difference_within_original_tolerance_is_accepted(self):
  from audit164 import actual_batch
  with contextlib.closing(Archive(self.paths['C163'])) as z:
   folder='step_01_before';head=z.arrays(folder+'/HEAD.npz')['P'];points={sid:mapping(parent.parent.previous.constants(self.ctx['data']),head,sid)[0] for sid in IDS};arrays=z.arrays(folder+'/ACTUAL_TORCH_FP64_POINTS.npz');arrays[IDS[0]][0]+=5e-14;err=float(np.max(abs(arrays[IDS[0]]-points[IDS[0]])));self.assertGreater(err,0.)
   bindings=z.json(folder+'/ACTUAL_TORCH_FP64_POINT_BINDINGS.json');bindings[0]['actual_Torch_FP64_sha256']=point_hash(arrays[IDS[0]]);bindings[0]['Torch_vs_NumPy_Linf']=err
   buf=io.BytesIO();np.savez_compressed(buf,**arrays);replacements={folder+'/ACTUAL_TORCH_FP64_POINTS.npz':buf.getvalue(),folder+'/ACTUAL_TORCH_FP64_POINT_BINDINGS.json':json.dumps(bindings).encode()};receipt=z.json(folder+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json')
   for n in receipt['artifact_seals']:receipt['artifact_seals'][n]=hashlib.sha256(replacements.get(folder+'/'+n,z.raw(folder+'/'+n))).hexdigest()
   replacements[folder+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json']=json.dumps(receipt).encode()
   class Wrapped:
    def raw(self,n):return replacements[n] if n in replacements else z.raw(n)
    def json(self,n):return clean_json(self.raw(n))
    def arrays(self,n):
     with np.load(io.BytesIO(self.raw(n)),allow_pickle=False) as f:return {k:f[k] for k in f.files}
   saved,_=actual_batch(Wrapped(),folder,head,points,'training_before_local_step1',171,0);self.assertEqual(saved[IDS[0]][0],arrays[IDS[0]][0])
 def test_independent_vector_Adam_sign_projection_and_scalar_agreement(self):
  p=np.zeros((3,2,6));g=np.linspace(-2,2,p.size).reshape(p.shape);m=p.copy();v=p.copy();q,mm,vv,d=adam(p,g,m,v,1);clipped=g/max(1.,np.linalg.norm(g));np.testing.assert_allclose(q,-1e-5*clipped/(abs(clipped)+1e-8),rtol=1e-14,atol=1e-18)
  for step in range(1,9):
   expected=old.adam(p,g,m,v,step);actual=adam(p,g,m,v,step)
   for a,b in zip(actual[:3],expected[:3]):np.testing.assert_allclose(a,b,atol=2e-16,rtol=1e-12)
   p,m,v=actual[:3]
  p=np.full_like(p,.125);z=np.zeros_like(p);np.testing.assert_array_equal(adam(p,-np.ones_like(p),z,z,1)[0],p)
  for step in (0,9,True):
   with self.assertRaises(ValueError):adam(p,z,z,z,step)
 def test_guards_no_models_or_solvers(self):
  with archive155.prior.no_models(),safe.no_solvers():
   with self.assertRaises(ValueError):__import__('torch')
   with self.assertRaises(RuntimeError):np.linalg.solve(np.eye(2),np.ones(2))
 def test_unsafe_and_duplicate_paths_block(self):
  for name,names in [('unsafe',['../bad']),('duplicate',['same','same'])]:
   p=self.root/(name+'.zip')
   with zipfile.ZipFile(p,'w') as z:
    for n in names:z.writestr(n,b'x')
   with self.assertRaises(ValueError):Archive(p)
if __name__=='__main__':unittest.main(verbosity=2)
