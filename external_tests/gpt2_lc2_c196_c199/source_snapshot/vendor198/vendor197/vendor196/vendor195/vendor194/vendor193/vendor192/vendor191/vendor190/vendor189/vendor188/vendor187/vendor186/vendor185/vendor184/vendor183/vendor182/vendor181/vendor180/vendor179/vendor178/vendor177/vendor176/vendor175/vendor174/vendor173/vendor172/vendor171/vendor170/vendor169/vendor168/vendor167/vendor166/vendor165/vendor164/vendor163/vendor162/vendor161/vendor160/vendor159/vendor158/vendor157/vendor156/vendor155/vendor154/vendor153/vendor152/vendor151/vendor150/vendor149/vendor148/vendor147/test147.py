"""Independent training arithmetic and adversarial archive tests; no Torch imports."""
import contextlib,copy,io,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core147 import *
from audit147 import audit
from worker147 import run
from fixture147 import make_fixture,rebound,npz_bytes

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.fixture=None;cls.full=None;cls.bound_context=None
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def original(self):
        if self.fixture is None:
            with contextlib.redirect_stdout(io.StringIO()):self.__class__.fixture=make_fixture(self.fresh()/'archives')
        return self.fixture
    def execute(self):
        if self.full is None:
            paths,c=self.original();out=self.fresh()
            with contextlib.redirect_stdout(io.StringIO()):s=run(paths,out,c)
            self.__class__.full=(s,out)
        return self.full
    def bad146(self,changes,pattern=None):
        paths,c=self.original();bad=rebound(paths['C146'],self.fresh()/'changed.zip',changes);cc=copy.deepcopy(c);cc['pins']['C146']['sha256']=sha(bad)
        # C143/C145 bytes do not change in these C146-only mutation tests.
        # Their complete reader is exercised separately by the full audit test.
        import audit147
        if self.bound_context is None:
            with contextlib.redirect_stdout(io.StringIO()),no_models(),safe.no_solvers():self.__class__.bound_context=audit147.bind_originals(paths,self.fresh(),c)
        def same_bound_originals(pp,out,ct):
            for k,p in pp.items():compare(sha(p),ct['pins'][k]['sha256'],'Mutation test input/'+k)
            return copy.deepcopy(self.bound_context)
        with patch.object(audit147,'bind_originals',same_bound_originals),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError) as ctx:
            with no_models(),safe.no_solvers():audit147.audit(dict(paths,C146=bad),self.fresh(),cc)
        if pattern:self.assertIn(pattern,str(ctx.exception))
    def read(self,name):
        paths,_=self.original()
        with contextlib.closing(Archive(paths['C146'])) as z:return z.json(name)
    def arrays(self,name):
        paths,_=self.original()
        with contextlib.closing(Archive(paths['C146'])) as z:return z.arrays(name)
    def rows(self):return [dict(state_id=sp['state_id'],objective_parts=dict(response_loss=float(i/83),read_loss=.2),response_gradient=[2.,-3.],guard_gradient=[.4,.2]) for i,sp in enumerate(registry())]
    def test_full_CPU_archive_without_model_fit_solve_or_SVD(self):
        before=set(sys.modules);s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['training_observations_checked'],1008);self.assertEqual(s['training_steps_audited'],12);self.assertEqual(s['sealed_final_proposals_recomputed'],252);self.assertEqual(s['native_gate_records_recomputed'],504)
        for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','linear_systems_solved'):self.assertEqual(s[k],0)
        self.assertFalse(any(n.split('.')[0] in {'torch','transformers','peft'} for n in set(sys.modules)-before));self.assertEqual(len(clean_json((out/'ALL1008_STATE_TRAINING_DIAGNOSTICS.json').read_bytes())),1008);self.assertEqual(len(clean_json((out/'ALL84_STATE_HISTORIES_AND_FINAL_GATES.json').read_bytes())),84)
    def test_failed_parent_retained_and_no_LC2_or_schedule_selected(self):
        s,_=self.execute();self.assertEqual(s['parent_decision'],'C146_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE');self.assertFalse(s['parent_fixed_final_training_qualification_passed'])
        for k in ('LC2_closed','LC2_single_fold_passed','candidate_selected','writer_deployed','parent_decision_reclassified','new_schedule_or_checkpoint_selected','causal_failure_explanation_established','gradients_independently_recomputed'):self.assertFalse(s[k])
    def test_functional_and_final_native_sources_are_distinct(self):
        _,out=self.execute();rows=clean_json((out/'ALL12_RECORDED_TRANSITIONS.json').read_bytes());self.assertEqual(len(rows),12);self.assertTrue(rows[-1]['native_functional_sources_differ']);self.assertEqual(rows[-1]['to_observation'],'final_native_round0');self.assertFalse(any(r['native_functional_sources_differ'] for r in rows[:-1]))
    def test_independent_objective_sign_units_and_heldout_block(self):
        point=np.zeros(DIM,np.float32);a=np.zeros(2);o=np.r_[[-.0036,.001,0,0,0,0],(2*np.tile([0,0,1,0],4)-1)*.1];parts=loss(o,0.,point,point,a,'xor_00');self.assertEqual(parts['response_loss'],1.);self.assertEqual(parts['read_loss'],0.)
        with self.assertRaises(ValueError):loss(o,0.,point,point,a,'xor_10')
    def test_independent_exact_max_and_guard_gradient_algebra(self):
        g,r=aggregate(self.rows(),2);np.testing.assert_allclose(g,[2.4,-2.8],atol=1e-13,rtol=1e-12);self.assertAlmostEqual(r['objective'],1.15);self.assertAlmostEqual(r['response_weights'][-1],.9+.1/84)
    def test_ties_and_zero_gradients_have_no_invented_direction(self):
        rows=self.rows()
        for r in rows:r.update(objective_parts=dict(response_loss=1.,read_loss=0.),response_gradient=[0.,0.],guard_gradient=[0.,0.])
        g,a=aggregate(rows,2);np.testing.assert_array_equal(g,np.zeros(2));self.assertEqual(a['exact_max_tie_count'],84);np.testing.assert_allclose(a['response_weights'],np.full(84,1/84),atol=1e-15)
    def test_independent_update_sign_scale_clip_and_projection(self):
        a=np.array([.1249,-.1249]);g=np.array([-3.,4.]);new,m,v,info=update(a,g,np.zeros(2),np.zeros(2),1);expected=np.clip(a-.0005*(g/5)/(abs(g/5)+1e-8),-.125,.125);np.testing.assert_allclose(new,expected,atol=1e-15,rtol=1e-12);self.assertEqual(info['projected_coordinates'],2)
        with self.assertRaises(ValueError):update(a,g,np.zeros(2),np.zeros(2),13)
    def test_scalar_dot_is_descriptive_not_error_bound(self):
        q=dot_diagnostic(np.array([2.,-3.]),np.array([.001,.002]));self.assertAlmostEqual(q['dot'],-.004);self.assertNotIn('accuracy_bound',q)
    def test_no_model_and_no_solver_guards(self):
        with no_models(),self.assertRaises(ValueError):__import__('torch')
        with safe.no_solvers(),self.assertRaises(Exception):np.linalg.solve(np.eye(2),np.ones(2))
    def test_wrong_input_hash_blocks_before_archive(self):
        paths,c=self.original();cc=copy.deepcopy(c);cc['pins']['C146']['sha256']='0'*64
        with self.assertRaises(ValueError):audit(paths,self.fresh(),cc)
    def test_archive_tampering_blocks(self):
        paths,c=self.original();p=self.fresh()/'bad.zip';p.write_bytes(paths['C146'].read_bytes()+b'tamper')
        with self.assertRaises(ValueError):audit(dict(paths,C146=p),self.fresh(),c)
    def test_unsafe_archive_paths_rejected(self):
        p=self.fresh()/'unsafe.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('../escape',b'no')
        with self.assertRaises(ValueError):Archive(p)
    def test_actual_initial_head_change_blocks_even_rebound(self):
        a=self.arrays('INITIAL_C143_STEP12_CONTROL.npz');a['a'][0]+=.001;self.bad146({'INITIAL_C143_STEP12_CONTROL.npz':npz_bytes(**a)},'Exact C143 final initialization')
    def test_actual_final_head_change_blocks_even_rebound(self):
        a=self.arrays('FINAL_SHARED_CONTROL_STEP12.npz');a['a'][0]+=.001;self.bad146({'FINAL_SHARED_CONTROL_STEP12.npz':npz_bytes(**a)},'Actual final tensor')
    def test_fixed_threshold_change_blocks(self):
        r=self.read('DESIGN_FROZEN_BEFORE_MODEL.json');r['acceptance_budgets']['response']=.003;self.bad146({'DESIGN_FROZEN_BEFORE_MODEL.json':json.dumps(r).encode()},'Original fixed C146 protocol')
    def test_source_change_blocks_even_rebound(self):self.bad146({'source/worker146.py':b"raise AssertionError('never execute archived source')\n"},'C146 trusted source')
    def test_missing_training_step_blocks(self):
        r=self.read('TRAINING_PROGRESS.json');self.bad146({'TRAINING_PROGRESS.json':json.dumps(r[:-1]).encode()},'All12 updates')
    def test_permuted_training_rows_block(self):
        r=self.read('training/step01/OBSERVATIONS.json');r[0],r[1]=r[1],r[0];self.bad146({'training/step01/OBSERVATIONS.json':json.dumps(r).encode()},'Gradient duplicate')
    def test_nonfinite_archived_gradient_blocks(self):
        r=self.read('training/step01/OBSERVATIONS.json');r[0]['response_gradient'][0]=float('nan');self.bad146({'training/step01/OBSERVATIONS.json':json.dumps(r).encode()},'Nonfinite JSON')
    def test_wrong_max_weight_and_contribution_block(self):
        r=self.read('training/step01/AGGREGATION.json');r['response_weights'][0]+=.1;self.bad146({'training/step01/AGGREGATION.json':json.dumps(r).encode()},'Independent max/mean aggregation')
    def test_rebound_update_change_blocks(self):
        r=self.arrays('training/step01/UPDATE.npz');r['after_a'][0]+=.001;self.bad146({'training/step01/UPDATE.npz':npz_bytes(**r)},'Independent Adam after_a')
    def test_missing_repeat_and_forged_gate_block(self):
        r=self.read('ALL504_FINAL_TRAINING_ENDPOINTS.json');raw=json.dumps(r[:-1]).encode();self.bad146({'ALL504_FINAL_TRAINING_ENDPOINTS.json':raw,'FINAL_ENDPOINT_PROGRESS.json':raw},'length differs')
    def test_summary_statistic_change_blocks(self):
        r=self.read('SUMMARY.json');r['final_training_summaries']['all84']['0']['C146_fixed_step12']['passed']+=1;self.bad146({'SUMMARY.json':json.dumps(r).encode()},'passed')
    def test_scalar_native_measurement_forgery_recomputed(self):
        r=self.read('ALL504_FINAL_TRAINING_ENDPOINTS.json');r[0]['metrics']['response_drift']+=.001;raw=json.dumps(r).encode();self.bad146({'ALL504_FINAL_TRAINING_ENDPOINTS.json':raw,'FINAL_ENDPOINT_PROGRESS.json':raw,'final_measurements/record000.json':json.dumps(r[0]).encode()},'scalar differs')
    def test_forged_gate_recomputed_after_all_duplicates_rebound(self):
        r=self.read('ALL504_FINAL_TRAINING_ENDPOINTS.json');r[0]['gates']['complete_pass']=not r[0]['gates']['complete_pass'];raw=json.dumps(r).encode();self.bad146({'ALL504_FINAL_TRAINING_ENDPOINTS.json':raw,'FINAL_ENDPOINT_PROGRESS.json':raw,'final_measurements/record000.json':json.dumps(r[0]).encode()},'Recomputed native gate')
    def test_changed_frozen_point_blocks_even_with_rebound_seals(self):
        a=self.arrays('ALL252_FROZEN_FINAL_POINTS.npz');key=next(iter(a));a[key][0]+=.001;raw=npz_bytes(**a);seal=self.read('STEP12_FREEZE_BEFORE_EVALUATION.json');seal['artifact_seals']['ALL252_FROZEN_FINAL_POINTS.npz']=hashlib.sha256(raw).hexdigest();self.bad146({'ALL252_FROZEN_FINAL_POINTS.npz':raw,'STEP12_FREEZE_BEFORE_EVALUATION.json':json.dumps(seal).encode()},'Recomputed final physical point')
    def test_launcher_failure_single_capsule_and_kernel_GPU_mask_preserved(self):
        import launch147
        old=dict(os.environ);self.assertEqual(launch147.child_environment()['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(dict(os.environ),old);root=self.fresh();dump(root/'INPUT_PINS.json',contract()['pins']);dump(root/'SOURCE_MANIFEST.json',{'INPUT_PINS.json':sha(root/'INPUT_PINS.json')});base=self.fresh()
        with patch.object(launch147,'ROOT',root),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):launch147.start(False,{},False,base)
        files=list(base.rglob('*_results.zip'));self.assertEqual(len(files),1)
        with contextlib.closing(Archive(files[0])) as z:z.verify_manifest();s=z.json('SUMMARY.json');self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['new_model_forwards'],0)

if __name__=='__main__':unittest.main(verbosity=2)
