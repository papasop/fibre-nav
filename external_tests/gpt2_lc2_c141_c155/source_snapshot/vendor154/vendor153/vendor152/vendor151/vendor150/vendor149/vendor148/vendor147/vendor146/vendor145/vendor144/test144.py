"""Synthetic end-to-end audits plus deliberate rebound archive corruptions."""
import contextlib,copy,io,os,tempfile,unittest,zipfile
from unittest.mock import patch
from core144 import *
from audit144 import audit
from fixture144 import make_fixture,rebound

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.paths,cls.c,cls.holder,cls.fixture_folder=make_fixture(cls.root/'fixture')
        with contextlib.redirect_stdout(io.StringIO()),safe.no_solvers():cls.complete=audit(cls.paths,cls.root/'full_audit',cls.c)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def fast_loader(self,paths,out,c):
        out.mkdir(parents=True,exist_ok=True);dump(out/'C142_BOUND_RECEIPT.json',self.holder['receipt']);return self.holder['data']
    def original_json(self,name):
        with zipfile.ZipFile(self.paths['C143']) as z:return clean_json(z.read(name))
    def encode(self,obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
    def bad_audit(self,changes,pattern=None):
        folder=self.fresh();bad=rebound(self.paths['C143'],folder/'bad.zip',changes);paths=dict(self.paths,C143=bad);c=copy.deepcopy(self.c);c['pins']['C143']['sha256']=sha(bad)
        error=self.assertRaisesRegex(ValueError,pattern) if pattern else self.assertRaises(ValueError)
        with patch('audit144.load_bound_inputs',self.fast_loader),contextlib.redirect_stdout(io.StringIO()),error:audit(paths,folder/'audit',c)
    def test_full_CPU_four_original_audit_12_updates1008_rows504_endpoints(self):
        s=self.complete;self.assertEqual(s['parent_decision'],'C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE');self.assertEqual(s['training_observations_audited'],1008);self.assertEqual(s['endpoint_records_audited'],504);self.assertEqual(s['training_steps_audited'],12);self.assertFalse(s['parent_training_qualification_passed']);self.assertFalse(s['gradients_independently_recomputed']);self.assertTrue(s['actual_final_head_is_update12'])
    def test_failed_training_and_newly_failed_state_retained(self):
        s=self.complete;self.assertEqual(s['final_training_summaries']['all84']['0'][ARMS[2]]['passed'],83);self.assertEqual(len(s['C143_failed_ids']),1);self.assertEqual(s['newly_failed_vs_C140'],s['C143_failed_ids']);self.assertEqual(len(s['rescued_vs_C140']),6)
    def test_full3211_tail_change_blocks_even_rebound_manifest(self):
        with zipfile.ZipFile(self.paths['C143']) as z:
            b=z.read('ALL84_TRAINING_INPUTS.npz')
        with np.load(io.BytesIO(b),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        k=next(k for k in arrays if k.endswith('__features'));arrays[k][-1]+=.1;buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        self.bad_audit({'ALL84_TRAINING_INPUTS.npz':buf.getvalue()},'full3211')
    def test_delivered_source_change_blocks_even_rebound_manifest(self):
        self.bad_audit({'source/worker143.py':b"raise RuntimeError('NEVER EXECUTE ARCHIVED SOURCE')\n"},'Trusted C143 source')
    def test_changed_acceptance_threshold_blocks(self):
        d=self.original_json('DESIGN_FROZEN_BEFORE_MODEL.json');d['acceptance_budgets']['response']=.003;self.bad_audit({'DESIGN_FROZEN_BEFORE_MODEL.json':self.encode(d)},'design/budgets')
    def test_permuted_training_registry_blocks(self):
        d=self.original_json('ALL84_TRAINING_REGISTRY_FROZEN.json');d[0],d[1]=d[1],d[0];self.bad_audit({'ALL84_TRAINING_REGISTRY_FROZEN.json':self.encode(d)},'registry')
    def test_missing_training_observation_blocks(self):
        d=self.original_json('training/step01/OBSERVATIONS.json')[:-1];self.bad_audit({'training/step01/OBSERVATIONS.json':self.encode(d),'training/step01/GRADIENT_PROGRESS.json':self.encode(d)},'Missing gradient state')
    def test_wrong_shared_coefficient_digest_blocks(self):
        d=self.original_json('training/step01/OBSERVATIONS.json');d[0]['coefficient_digest']='0'*64;self.bad_audit({n:self.encode(d) for n in ('training/step01/OBSERVATIONS.json','training/step01/GRADIENT_PROGRESS.json')},'shared vector')
    def test_forged_objective_parts_block(self):
        d=self.original_json('training/step01/OBSERVATIONS.json');d[0]['objective_parts']['response_loss']+=.01;self.bad_audit({n:self.encode(d) for n in ('training/step01/OBSERVATIONS.json','training/step01/GRADIENT_PROGRESS.json')},'state objective')
    def test_changed_gradient_contribution_blocks(self):
        d=self.original_json('training/step01/OBSERVATIONS.json');d[0]['gradient_contribution'][0]+=.01;self.bad_audit({n:self.encode(d) for n in ('training/step01/OBSERVATIONS.json','training/step01/GRADIENT_PROGRESS.json')},'Weighted state contribution')
    def test_update_change_blocks_even_with_rebound_update_digest(self):
        with zipfile.ZipFile(self.paths['C143']) as z:
            with np.load(io.BytesIO(z.read('training/step01/UPDATE.npz')),allow_pickle=False) as f:u={k:f[k] for k in f.files}
        u['after_a']=u['after_a'].copy();u['after_a'][0]+=.001;b=io.BytesIO();np.savez_compressed(b,**u);raw=b.getvalue();progress=self.original_json('TRAINING_PROGRESS.json');progress[0]['update_sha256']=hashlib.sha256(raw).hexdigest();progress[0]['after_digest']=row_digest(u['after_a'])
        self.bad_audit({'training/step01/UPDATE.npz':raw,'TRAINING_PROGRESS.json':self.encode(progress)},'Independent Adam')
    def test_final_head_cannot_be_replaced_even_with_rebound_seal(self):
        with zipfile.ZipFile(self.paths['C143']) as z:
            with np.load(io.BytesIO(z.read('FINAL_SHARED_CONTROL_STEP12.npz')),allow_pickle=False) as f:a=f['a']
        a=a.copy();a[0]+=.001;b=io.BytesIO();np.savez_compressed(b,a=a);raw=b.getvalue();freeze=self.original_json('STEP12_FREEZE_BEFORE_EVALUATION.json');freeze['control_sha256']=hashlib.sha256(raw).hexdigest();freeze['control_digest']=row_digest(a);freeze['proposal_seals']['FINAL_SHARED_CONTROL_STEP12.npz']=hashlib.sha256(raw).hexdigest()
        self.bad_audit({'FINAL_SHARED_CONTROL_STEP12.npz':raw,'STEP12_FREEZE_BEFORE_EVALUATION.json':self.encode(freeze)},'actual fixed step12')
    def test_missing_repeat_endpoint_blocks(self):
        d=self.original_json('ALL504_FINAL_TRAINING_ENDPOINTS.json')[:-1];self.bad_audit({'ALL504_FINAL_TRAINING_ENDPOINTS.json':self.encode(d),'FINAL_ENDPOINT_PROGRESS.json':self.encode(d)},'All504')
    def test_forged_gate_is_independently_recomputed(self):
        d=self.original_json('ALL504_FINAL_TRAINING_ENDPOINTS.json');d[2]['recomputed_gates']['complete_pass']=True;d[2]['recomputed_gates']['failed_gates']=[]
        self.bad_audit({'ALL504_FINAL_TRAINING_ENDPOINTS.json':self.encode(d),'FINAL_ENDPOINT_PROGRESS.json':self.encode(d),'final_measurements/record002.json':self.encode(d[2])},'native endpoint gates')
    def test_forged_summary_count_blocks(self):
        d=self.original_json('SUMMARY.json');d['final_training_summaries']['all84']['0'][ARMS[2]]['passed']=84;self.bad_audit({'SUMMARY.json':self.encode(d)})
    def test_fixed_original_hash_blocks_before_payload(self):
        c=copy.deepcopy(self.c);c['pins']['C143']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA256'):audit(self.paths,self.fresh(),c)
    def test_production_pins_reject_fixture(self):
        with self.assertRaises(ValueError):audit(self.paths,self.fresh(),contract())
    def test_manifest_tampering_blocks(self):
        bad=self.fresh()/'bad.zip'
        with zipfile.ZipFile(self.paths['C143']) as z,zipfile.ZipFile(bad,'w') as w:
            for n in z.namelist():w.writestr(n,b'{}' if n=='SUMMARY.json' else z.read(n))
        paths=dict(self.paths,C143=bad);c=copy.deepcopy(self.c);c['pins']['C143']['sha256']=sha(bad)
        with patch('audit144.load_bound_inputs',self.fast_loader),self.assertRaisesRegex(ValueError,'member hash'):audit(paths,self.fresh(),c)
    def test_unsafe_archive_path_blocks(self):
        p=self.fresh()/'unsafe.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('../bad','bad')
        with self.assertRaises(ValueError):Archive(p)
    def test_independent_aggregate_matches_analytic_and_finite_difference(self):
        rng=np.random.default_rng(144);M=rng.normal(size=(84,3));b=np.linspace(.01,.4,84);a=np.array([.02,-.01,.03])
        def f(a):r=M@a+b;return aggregate(.01+r*r,2*r[:,None]*M)
        g,record=f(a);fd=np.array([(f(a+np.eye(3)[j]*1e-6)[1]['objective']-f(a-np.eye(3)[j]*1e-6)[1]['objective'])/2e-6 for j in range(3)]);np.testing.assert_allclose(g,fd,atol=1e-10,rtol=1e-8);self.assertAlmostEqual(sum(record['weights']),1)
    def test_clipped_Adam_sign_scale_and_noop(self):
        a=np.array([.02,-.03]);g=np.array([3.,-4.]);new,_,_,info=update(a,g,np.zeros(2),np.zeros(2),1);q=g/5;np.testing.assert_allclose(new,a-.0005*q/(abs(q)+1e-8),atol=1e-15);self.assertAlmostEqual(info['gradient_clip_factor'],.2);same,_,_,_=update(a,np.zeros(2),np.zeros(2),np.zeros(2),1);exact(a,same,'Zero gradient noop')
    def test_nonfinite_or_missing_rows_block(self):
        with self.assertRaises(ValueError):aggregate(np.r_[np.nan,np.zeros(83)],np.zeros((84,2)))
        with self.assertRaises(ValueError):aggregate(np.zeros(83),np.zeros((83,2)))
    def test_signed_coordinate_slack_preserves_both_signs(self):
        d=response_diagnostic([.003,-.003,0,0,0,0],[.0021,-.0022,0,0,0,0]);np.testing.assert_allclose(d['measured_response_change'][:2],[-.0009,.0008]);np.testing.assert_allclose(d['C143_coordinate_budget_slack'][:2],[-.0001,-.0002])
    def test_two_rounds_do_not_increase_independent_training_denominator(self):
        self.assertEqual(self.complete['training_states_audited'],84);self.assertEqual(self.complete['endpoint_records_audited'],504);self.assertTrue(self.complete['all84_remain_training'])
    def test_no_torch_HF_imports_or_solver_during_full_audit(self):
        self.assertFalse(any(k=='torch' or k.startswith('torch.') or k=='transformers' or k.startswith('transformers.') for k in sys.modules))
    def test_worker_retains_parent_failure_with_zero_new_compute(self):
        from worker144 import run
        out=self.fresh()
        with patch('audit144.load_bound_inputs',self.fast_loader),contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c)
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['decision'],'C144_READONLY_JOINT84_AUDIT_COMPLETE_PARENT_DECISION_RETAINED');self.assertFalse(s['parent_training_qualification_passed']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['LC2_closed']);self.assertTrue((out/'REPORT_ZH.md').is_file());self.assertTrue(all(s['original_inputs_unchanged'].values()))
    def test_launcher_child_mask_only_and_worker144(self):
        import launch144,inspect
        old=dict(os.environ);env=launch144.child_environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(old,dict(os.environ));source=inspect.getsource(launch144.start);self.assertIn("ROOT/'worker144.py'",source);self.assertNotIn('dependencies(',source)
    def test_launcher_failure_still_packages_one_result(self):
        import launch144
        root=self.fresh();dump(root/'INPUT_PINS.json',contract()['pins']);dump(root/'SOURCE_MANIFEST.json',{'INPUT_PINS.json':sha(root/'INPUT_PINS.json')});base=self.fresh()
        with patch.object(launch144,'ROOT',root),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):launch144.start(False,{},False,base)
        capsules=list(base.rglob('*_results.zip'));self.assertEqual(len(capsules),1)
        with contextlib.closing(Archive(capsules[0])) as z:
            z.verify_manifest();s=z.json('SUMMARY.json');self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['decision'],'C144_INCOMPLETE_AUDIT_NOT_ACCEPTED')

if __name__=='__main__':unittest.main(verbosity=2)
