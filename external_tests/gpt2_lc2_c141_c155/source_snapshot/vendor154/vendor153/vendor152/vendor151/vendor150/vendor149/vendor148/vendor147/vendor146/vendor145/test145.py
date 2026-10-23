"""NumPy protocol tests. Actual Torch graph tests are mandatory in a separate child."""
import contextlib,copy,io,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core145 import *
from fixture145 import data145,dummy_inputs,emit,Toy,Analytic,archive_inputs
from inputs145 import load
from worker145 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.data=data145();cls.paths,cls.c=dummy_inputs(cls.root/'dummy');cls.cache={};cls.originals=None
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'run'
            def loader(paths,out,c):d=copy.deepcopy(self.data);emit(out,d,paths);return d
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda out,d,ph,calls:Toy(out,d,ph,calls,mode),lambda rt,d,ph,calls:Analytic(rt,d,ph,calls,mode),loader)
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def five_originals(self):
        if self.originals is None:self.__class__.originals=archive_inputs(self.fresh()/'archives')
        return self.originals
    def test_full84_two_directions168_backwards1512_endpoints4033_forwards(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_backward_calls'],168);self.assertEqual(s['endpoint_records'],1512);self.assertEqual(s['new_model_forwards'],4033);self.assertEqual(s['gradient_state_records'],84);self.assertEqual(s['preflight_records'],336);self.assertEqual(s['diagnostic_control_vectors_generated'],8)
        with np.load(out/'ALL756_FROZEN_DIAGNOSTIC_POINTS.npz',allow_pickle=False) as f:self.assertEqual(len(f.files),756)
    def test_no_selection_or_LC2_even_if_registered_probes_pass_all84(self):
        s,_=self.execute();self.assertTrue(s['diagnostic_configurations_with_all84_passed']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['writer_deployed']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_head_fits'],0);self.assertEqual(s['optimizer_updates'],0)
    def test_registry_all84_exposed_training_three_seen_classes(self):
        rr=registry();self.assertEqual(len(rr),84);self.assertEqual(Counter(r['case'] for r in rr),dict(xor_00=28,xor_01=28,xor_11=28));self.assertTrue(all(r['used_for_training'] and not r['independent_validation_eligible'] and r['dataset_role']=='training' for r in rr))
    def test_fixed_configuration_registry_both_signs_both_steps(self):
        cfg=configurations();self.assertEqual(len(cfg),9);self.assertEqual(len({r['config_id'] for r in cfg}),9);self.assertEqual({(r['direction'],r['step'],r['sign']) for r in cfg[1:]},{(d,h,s) for d in DIRECTIONS for h in (.0005,.0015) for s in (1,-1)})
    def test_preflight_drift_blocks_before_any_backward(self):
        s,_=self.execute('preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['diagnostic_control_vectors_generated'],0)
    def test_all84_parity_before_any_backward(self):
        s,_=self.execute('parity_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['endpoint_records'],0)
    def test_gradient_graph_forward_drift_blocks_before_backward(self):
        s,_=self.execute('gradient_forward_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_nonfinite_gradient_blocks_before_directions_or_probes(self):
        s,_=self.execute('nonfinite_gradient');self.assertFalse(s['complete']);self.assertEqual(s['diagnostic_control_vectors_generated'],0);self.assertEqual(s['endpoint_records'],0)
    def test_wrong_frozen_control_digest_blocks(self):
        s,_=self.execute('wrong_digest');self.assertFalse(s['complete']);self.assertEqual(s['diagnostic_control_vectors_generated'],0)
    def test_basis_mutation_during_native_probes_blocks(self):
        s,_=self.execute('mutate_basis');self.assertFalse(s['complete']);self.assertFalse(s['candidate_selected'])
    def test_repeat_drift_retained_no_direction_interpretation(self):
        s,_=self.execute('repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['baseline_replay_consistent']);self.assertIsNone(s['diagnostic_interpretation']);self.assertEqual(s['diagnostic_configurations_with_all84_passed'],[]);self.assertFalse(s['candidate_selected'])
    def test_actual_probe_seals_and_baseline_control_unchanged(self):
        s,out=self.execute();f=clean_json((out/'ALL_PROBES_SEALED_BEFORE_MEASUREMENTS.json').read_bytes());self.assertEqual(f['native_probe_endpoint_measurements_so_far'],0);self.assertEqual(f['points'],756)
        for n,h in f['artifact_seals'].items():self.assertEqual(sha(out/n),h)
        with np.load(out/'ALL9_FROZEN_CONTROLS.npz',allow_pickle=False) as a:exact(a['C143_baseline'],self.data['a'],'Original final C143 retained')
    def test_both_objective_and_response_direction_rules_use_all84(self):
        s,out=self.execute();r=clean_json((out/'DIRECTION_RULES_AND_WEIGHTS.json').read_bytes());self.assertEqual(set(r),set(DIRECTIONS))
        for v in r.values():self.assertEqual(len(v['state_weights']),84);self.assertAlmostEqual(sum(v['state_weights']),1.);self.assertEqual(len(v['negative_gradient_Linf_normalized']),6)
    def test_zero_gradients_have_no_fabricated_direction(self):
        rr=[dict(state_id=r['state_id'],original_gradient=[0.,0.],response_gradient=[0.,0.],original_objective=0.,response_ratio=0.) for r in registry()];d,info=directions_from_rows(rr,2)
        for k in DIRECTIONS:self.assertFalse(info[k]['available']);exact(d[k],np.zeros(2),'Explicit zero direction')
        a=np.array([.02,-.01]);controls,rows=probe_controls(a,d)
        for v in controls.values():exact(v,a,'Degenerate probes retained at baseline')
    def test_negative_gradient_normalization_and_units(self):
        rr=[dict(state_id=r['state_id'],original_gradient=[3.,-4.],response_gradient=[-2.,1.],original_objective=.1,response_ratio=.7) for r in registry()];d,info=directions_from_rows(rr,2);np.testing.assert_allclose(d[DIRECTIONS[0]],[-.75,1]);np.testing.assert_allclose(d[DIRECTIONS[1]],[1,-.5]);self.assertEqual(qvalue([-.003,.001,0,0,0,0]+[.1]*16),1.5)
    def test_both_aggregate_directions_match_finite_difference_of_declared_goals(self):
        rng=np.random.default_rng(145);M=rng.normal(size=(84,3));P=rng.normal(size=(84,3));b=np.linspace(.01,.2,84);a=np.array([.01,-.02,.03])
        def function(x):
            u=P@x+b;L=.1+u*u;q=.8+M@x;rows=[dict(state_id=r['state_id'],original_gradient=(2*u[i]*P[i]).tolist(),response_gradient=M[i].tolist(),original_objective=float(L[i]),response_ratio=float(q[i])) for i,r in enumerate(registry())]
            v0=auditcore.aggregate(L,np.zeros((84,1)))[1]['objective'];shift=q.max();v1=shift+np.log(np.exp(32*(q-shift)).mean())/32
            return rows,np.array([v0,v1])
        rows,_=function(a);directions,_=directions_from_rows(rows,3);fd=np.array([(function(a+np.eye(3)[j]*1e-6)[1]-function(a-np.eye(3)[j]*1e-6)[1])/2e-6 for j in range(3)]).T
        for i,name in enumerate(DIRECTIONS):np.testing.assert_allclose(directions[name],-fd[i]/max(abs(fd[i])),atol=2e-8,rtol=2e-7)
    def test_box_projection_recorded_with_actual_coefficient_displacement(self):
        a=np.array([.1249,-.1249]);d={k:np.array([1.,-1.]) for k in DIRECTIONS};cs,rows=probe_controls(a,d);r=next(r for r in rows if r['direction']==DIRECTIONS[0] and r['sign']==1);self.assertEqual(r['projected_coordinates'],2);self.assertTrue(np.all(abs(cs[r['config_id']])<=.125));np.testing.assert_allclose(r['coefficient_change'],cs[r['config_id']]-a)
    def test_missing_or_permuted_gradient_registry_blocks(self):
        rr=[dict(state_id=r['state_id'],original_gradient=[1.],response_gradient=[1.],original_objective=.1,response_ratio=.5) for r in registry()]
        with self.assertRaises(ValueError):directions_from_rows(rr[:-1],1)
        rr[0],rr[1]=rr[1],rr[0]
        with self.assertRaises(ValueError):directions_from_rows(rr,1)
    def test_nonfinite_direction_and_unregistered_direction_block(self):
        a=np.zeros(2)
        with self.assertRaises(ValueError):probe_controls(a,{k:np.array([np.nan,0]) for k in DIRECTIONS})
        with self.assertRaises(ValueError):probe_controls(a,{'foreign':np.zeros(2)})
    def test_phase_prevents_probes_before_full84_and_unregistered_points(self):
        ph=Phase(self.data)
        with self.assertRaises(ValueError):ph.freeze({})
        with self.assertRaises(ValueError):ph.begin_gradients()
        sp=registry()[0];p=self.data['points'][sp['state_id']]['prepared'].copy();p[0]+=.01
        with self.assertRaises(ValueError):ph.authorize((sp['state_id'],'prepared'),p,sp['case'],False)
    def test_five_original_hashes_block_before_model(self):
        for k in INPUT_KEYS:
            c=copy.deepcopy(self.c);c['pins'][k]['sha256']='0'*64
            with self.assertRaises(ValueError):load(self.paths,self.fresh(),c)
    def test_five_original_archive_reader_rebinds_exact_C143_final84(self):
        paths,c=self.five_originals()
        with contextlib.redirect_stdout(io.StringIO()):d=load(paths,self.fresh(),c)
        self.assertEqual(len(d['points']),84);self.assertEqual(d['C143_summary']['decision'],'C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
        with contextlib.closing(Archive(paths['C143'])) as z:exact(d['a'],z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],'Original actual C143 final')
    def test_rebound_C144_source_change_blocks(self):
        from fixture144 import rebound
        paths,c=self.five_originals();bad=rebound(paths['C144'],self.fresh()/'bad.zip',{'source/worker144.py':b"raise AssertionError('NEVER EXECUTE ARCHIVED SOURCE')\n"});cc=copy.deepcopy(c);cc['pins']['C144']['sha256']=sha(bad)
        with self.assertRaisesRegex(ValueError,'Trusted C144 source'):load(dict(paths,C144=bad),self.fresh(),cc)
    def test_production_contract_rejects_synthetic_five_originals(self):
        with self.assertRaises(ValueError):load(self.paths,self.fresh(),contract())
    def test_parent_summaries_and_repeats_not_independent_evidence(self):
        s,_=self.execute();self.assertEqual(s['training_states'],84);self.assertEqual(s['confirmation_states_evaluated'],0);self.assertEqual(s['excluded_input_states_evaluated'],0);self.assertFalse(s['parent_decision_reclassified'])
    def test_launcher_CPU_child_mask_only_and_mandatory_smoke(self):
        import inspect,launch145
        old=dict(os.environ);env=launch145.child_environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(old,dict(os.environ));source=inspect.getsource(launch145.start);self.assertIn("ROOT/'worker145.py'",source);self.assertIn("ROOT/'torch_smoke145.py'",source);self.assertIn('dependencies(deps,env,log)',source)
    def test_launcher_failure_packages_one_capsule_with_unqualified_state(self):
        import launch145
        root=self.fresh();dump(root/'INPUT_PINS.json',contract()['pins']);dump(root/'SOURCE_MANIFEST.json',{'INPUT_PINS.json':sha(root/'INPUT_PINS.json')});base=self.fresh()
        with patch.object(launch145,'ROOT',root),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):launch145.start(False,{},False,base)
        capsules=list(base.rglob('*_results.zip'));self.assertEqual(len(capsules),1)
        with contextlib.closing(Archive(capsules[0])) as z:z.verify_manifest();s=z.json('SUMMARY.json');self.assertFalse(s['complete']);self.assertEqual(s['requested_endpoint_records'],1512);self.assertFalse(s['LC2_closed'])

if __name__=='__main__':unittest.main(verbosity=2)
