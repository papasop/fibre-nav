"""Full84 synthetic integration, independent algebra and adversarial archival checks."""
import contextlib,copy,io,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core146 import *
from fixture146 import data146,dummy_inputs,bindings,Native,Analytic,archive_inputs
from inputs146 import load,audit145
from worker146 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.data=data146();cls.paths,cls.c=dummy_inputs(cls.root/'dummy');cls.cache={};cls.originals=None;cls.bound=None
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'run'
            def loader(paths,out,c):d=copy.deepcopy(self.data);bindings(out,d,paths);return d
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda out,d,ph,calls:Native(out,d,ph,calls,mode),lambda rt,d,ph,calls:Analytic(rt,d,ph,calls,mode),loader)
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def original_fixture(self):
        if self.originals is None:
            with contextlib.redirect_stdout(io.StringIO()):self.__class__.originals=archive_inputs(self.fresh()/'archives')
        return self.originals
    def bound_fixture(self):
        if self.bound is None:
            paths,c=self.original_fixture()
            with contextlib.redirect_stdout(io.StringIO()):self.__class__.bound=load(paths,self.fresh(),c)
        return self.bound
    def rows(self,R=None,gr=None,gg=None,G=None):
        R=np.linspace(0.,1.,N) if R is None else np.asarray(R);gr=np.tile([2.,-3.],(N,1)) if gr is None else np.asarray(gr);gg=np.tile([.4,.2],(N,1)) if gg is None else np.asarray(gg);G=np.full(N,.2) if G is None else np.asarray(G)
        return [dict(state_id=s['state_id'],objective_parts=dict(response_loss=float(R[i]),read_loss=float(G[i])),response_gradient=gr[i].tolist(),guard_gradient=gg[i].tolist()) for i,s in enumerate(registry())]
    def test_full84_fixed12_2016_backwards_504_endpoints_4033_forwards(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],12);self.assertEqual(s['new_backward_calls'],2016);self.assertEqual(s['new_model_forwards'],4033);self.assertEqual(s['preflight_records'],336);self.assertEqual(s['zero_parity_records'],84);self.assertEqual(s['warm_parity_records'],84);self.assertEqual(s['final_endpoint_records'],504);self.assertTrue(s['step12_frozen_before_final_evaluation'])
        with np.load(out/'ALL252_FROZEN_FINAL_POINTS.npz',allow_pickle=False) as f:self.assertEqual(len(f.files),252)
    def test_complete_training_never_closes_LC2_or_selects_probe(self):
        s,_=self.execute();self.assertTrue(s['fixed_final_training_qualification_passed']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertFalse(s['C145_probe_selected']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['writer_deployed']);self.assertEqual(s['excluded_input_states_evaluated'],0)
    def test_final_head_is_exact_step12_not_best_or_recovery(self):
        s,out=self.execute();last=clean_json((out/'TRAINING_PROGRESS.json').read_bytes())[-1]
        with np.load(out/last['update_file'],allow_pickle=False) as u,np.load(out/'FINAL_SHARED_CONTROL_STEP12.npz',allow_pickle=False) as f:exact(f['a'],u['after_a'],'Fixed step12 only')
        self.assertFalse(s['best_checkpoint_selected']);self.assertFalse(s['recovery_checkpoint_selected'])
    def test_freeze_precedes_endpoints_and_C143_initialization_unchanged(self):
        _,out=self.execute();seal=clean_json((out/'STEP12_FREEZE_BEFORE_EVALUATION.json').read_bytes());self.assertEqual(seal['final_endpoint_measurements_so_far'],0)
        for n,h in seal['artifact_seals'].items():self.assertEqual(sha(out/n),h)
        with np.load(out/'INITIAL_C143_STEP12_CONTROL.npz',allow_pickle=False) as f:exact(f['a'],self.data['a'],'Never a selected C145 probe')
    def test_failed_final_retains_all84_and_both_repeats(self):
        s,_=self.execute('failed_final');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['fixed_final_training_qualification_passed']);self.assertEqual(s['decision'],'C146_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE');self.assertEqual(s['final_endpoint_records'],504)
        for rnd in ('0','1'):self.assertEqual(s['final_training_summaries']['all84'][rnd][EVAL_ARMS[2]]['passed'],83)
    def test_repeat_drift_blocks_qualification(self):
        s,_=self.execute('repeat146');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['fixed_final_training_qualification_passed'])
    def test_preflight_drift_before_any_backward(self):
        s,_=self.execute('preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0)
    def test_zero_and_warm_parity_before_any_backward(self):
        for mode in ('zero_drift','warm_drift'):
            s,_=self.execute(mode);self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0)
    def test_missing_training_row_nonfinite_and_wrong_weights_block(self):
        for mode in ('missing_state','nonfinite','wrong_weight','wrong_digest'):
            s,_=self.execute(mode);self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['fixed_final_training_qualification_passed'])
    def test_interruption_retains_partial_but_never_promotes_it(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['new_training_steps'],1);self.assertFalse((out/'FINAL_SHARED_CONTROL_STEP12.npz').exists());self.assertFalse(s['fixed_final_training_qualification_passed'])
    def test_basis_mutation_blocks(self):
        s,_=self.execute('mutate_basis');self.assertFalse(s['complete']);self.assertFalse(s['fixed_final_training_qualification_passed'])
    def test_max_weight_plus_all84_guard_mean_independent_algebra(self):
        g,a=aggregate(self.rows(),2);np.testing.assert_allclose(g,[2.4,-2.8],atol=1e-13,rtol=1e-12);self.assertAlmostEqual(sum(a['response_weights']),1.);self.assertAlmostEqual(a['response_weights'][-1],.9+.1/84);self.assertAlmostEqual(a['objective'],.9+.1*.5+.2)
    def test_exact_ties_are_averaged(self):
        R=np.zeros(N);R[:2]=1;g,a=aggregate(self.rows(R=R),2);self.assertEqual(a['exact_max_tie_count'],2);self.assertAlmostEqual(a['response_weights'][0],.45+.1/84);self.assertEqual(a['response_weights'][0],a['response_weights'][1]);self.assertTrue(all(w>0 for w in a['response_weights']))
    def test_aggregate_gradient_matches_finite_difference_of_declared_goal(self):
        rng=np.random.default_rng(146);M=rng.normal(size=(84,3));P=rng.normal(size=(84,3));b=np.linspace(.01,.3,84);a=np.array([.03,-.02,.01])
        def fun(x):
            r=(M@x+b)**2;g=(P@x+.02)**2;rows=self.rows(R=r,gr=2*(M@x+b)[:,None]*M,gg=2*(P@x+.02)[:,None]*P,G=g);return aggregate(rows,3)
        g,agg=fun(a);fd=np.array([(fun(a+np.eye(3)[j]*1e-6)[1]['objective']-fun(a-np.eye(3)[j]*1e-6)[1]['objective'])/2e-6 for j in range(3)]);np.testing.assert_allclose(g,fd,atol=1e-8,rtol=1e-7)
    def test_negative_nonfinite_and_permuted_rows_block(self):
        rows=self.rows();rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):aggregate(rows,2)
        rows=self.rows();rows[0]['response_gradient'][0]=float('nan')
        with self.assertRaises(ValueError):aggregate(rows,2)
        rows=self.rows();rows[0]['objective_parts']['response_loss']=-1.
        with self.assertRaises(ValueError):aggregate(rows,2)
    def test_objective_units_safety_and_heldout_forbidden(self):
        p=np.zeros(DIM,np.float32);a=np.zeros(3);o=np.r_[[.0036,-.001,0,0,0,0],np.array(safe.target_bits('xor_00',True)*4)*.2-.1];parts=objective(o,0.,p,p,a,'xor_00');self.assertAlmostEqual(parts['response_loss'],1.);self.assertEqual(parts['read_loss'],0.)
        with self.assertRaises(ValueError):objective(o,0.,p,p,a,'xor_10')
    def test_independent_Adam_direction_scale_and_projection(self):
        a=np.array([.1249,-.1249]);g=np.array([-3.,4.]);m=np.zeros(2);v=np.zeros(2);new,mm,vv,info=update(a,g,m,v,1);clipped=g/5;expected=np.clip(a-.0005*clipped/(abs(clipped)+1e-8),-.125,.125);np.testing.assert_allclose(new,expected,atol=1e-15,rtol=1e-13);self.assertEqual(info['projected_coordinates'],2)
        with self.assertRaises(ValueError):update(a,g,m,v,13)
    def test_registry_no_confirmation_or_excluded_relabel(self):
        rr=registry();self.assertEqual(len(rr),84);self.assertEqual(Counter(r['case'] for r in rr),dict(xor_00=28,xor_01=28,xor_11=28));self.assertTrue(all(r['used_for_training'] and r['C146_membership']=='training' and not r['independent_validation_eligible'] for r in rr));self.assertEqual(design()['acceptance_budgets']['response'],.002)
    def test_phase_no_freeze_or_training_before_full84(self):
        ph=Phase(self.data)
        with self.assertRaises(ValueError):ph.begin_training()
        with self.assertRaises(ValueError):ph.freeze({})
        with self.assertRaises(ValueError):ph.begin_batch(0,self.data['a'])
    def test_six_fixed_hashes_block_before_parent_or_model(self):
        for k in INPUT_KEYS:
            c=copy.deepcopy(self.c);c['pins'][k]['sha256']='0'*64
            with self.assertRaises(ValueError):load(self.paths,self.fresh(),c)
    def test_production_contract_rejects_fixture(self):
        with self.assertRaises(ValueError):load(self.paths,self.fresh(),contract())
    def test_six_original_reader_full_C145_recomputation(self):
        d=self.bound_fixture();self.assertEqual(len(d['points']),84);self.assertEqual(d['C145_summary']['new_backward_calls'],168);paths,c=self.original_fixture()
        with contextlib.closing(Archive(paths['C143'])) as z:exact(d['a'],z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],'Original C143 final, not C145 probe')
    def test_forged_C145_gate_is_recomputed_even_rebound(self):
        from fixture144 import rebound
        paths,c=self.original_fixture();d=self.bound_fixture()
        with contextlib.closing(Archive(paths['C145'])) as z:rows=z.json('ALL1512_DIAGNOSTIC_ENDPOINTS.json')
        rows[0]['gates']['complete_pass']=not rows[0]['gates']['complete_pass'];bad=rebound(paths['C145'],self.fresh()/'bad.zip',{'ALL1512_DIAGNOSTIC_ENDPOINTS.json':json.dumps(rows).encode()})
        with contextlib.closing(Archive(bad)) as z,self.assertRaises(ValueError):audit145(z,d,c)
    def test_rebound_C145_source_change_blocks(self):
        from fixture144 import rebound
        paths,c=self.original_fixture();d=self.bound_fixture();bad=rebound(paths['C145'],self.fresh()/'bad.zip',{'source/worker145.py':b"raise AssertionError('never execute archived code')\n"})
        with contextlib.closing(Archive(bad)) as z,self.assertRaisesRegex(ValueError,'Trusted C145 source'):audit145(z,d,c)
    def test_rebound_baseline_control_change_blocks(self):
        from fixture144 import rebound
        paths,c=self.original_fixture();d=self.bound_fixture();b=io.BytesIO();np.savez_compressed(b,a=d['a']+.001,V=d['V'],C140_a=d['C140_a']);bad=rebound(paths['C145'],self.fresh()/'bad.zip',{'BASELINE_C143_CONTROL.npz':b.getvalue()})
        with contextlib.closing(Archive(bad)) as z,self.assertRaisesRegex(ValueError,'Original baseline'):audit145(z,d,c)
    def test_launcher_failure_single_capsule_and_CPU_mask(self):
        import launch146
        env=dict(os.environ);self.assertEqual(launch146.child_environment()['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(dict(os.environ),env);root=self.fresh();dump(root/'INPUT_PINS.json',contract()['pins']);dump(root/'SOURCE_MANIFEST.json',{'INPUT_PINS.json':sha(root/'INPUT_PINS.json')});base=self.fresh()
        with patch.object(launch146,'ROOT',root),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):launch146.start(False,{},False,base)
        capsules=list(base.rglob('*_results.zip'));self.assertEqual(len(capsules),1)
        with contextlib.closing(Archive(capsules[0])) as z:z.verify_manifest();s=z.json('SUMMARY.json');self.assertFalse(s['complete']);self.assertFalse(s['fixed_final_training_qualification_passed']);self.assertFalse(s['LC2_closed'])

if __name__=='__main__':unittest.main(verbosity=2)
