import contextlib,copy,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from core139 import *
from fixture139 import *
from inputs139 import audit138
from worker139 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.paths,cls.c=dummy_inputs(cls.root);cls.z138,cls.olddata,cls.oldc=archive138(cls.root/'previous');cls.cache={}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'out'
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *a:Toy(*a,mode=mode),lambda paths,out,c:data139())
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def audit(self,path=None):
        out=self.fresh()
        with contextlib.closing(Archive(path or self.z138)) as z:audit138(z,copy.deepcopy(self.olddata),out,self.oldc)
        return clean_json((out/'C138_PREPARATION_AND_ENDPOINT_AUDIT.json').read_bytes())
    def modified(self,name,fn):
        folder=self.fresh()
        with zipfile.ZipFile(self.z138) as z:z.extractall(folder)
        p=folder/name
        if p.suffix=='.json':dump(p,fn(clean_json(p.read_bytes())))
        else:
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            np.savez_compressed(p,**fn(a))
        # Rebind outer manifest: semantic checks must still reject forged contents.
        return pack(folder,self.fresh()/'changed.zip')
    def test_full_worker517_forwards48_preparations96_endpoints(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],517);self.assertEqual(s['kernel_preparation_records'],48);self.assertEqual(s['final_endpoint_records'],96);self.assertEqual(s['kernel_feature_records'],24);self.assertEqual(s['C138_preparation_replays'],24);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['LC2_closed'])
    def test_all24_exposed_directions_not_unseen(self):
        r=registry();self.assertEqual(len(r),24);self.assertEqual(len({x['source_state_id'] for x in r}),24);self.assertEqual(Counter(x['case'] for x in r),dict(xor_00=8,xor_01=8,xor_11=8));self.assertTrue(all(x['dataset_role']=='development' and not x['independent_validation_eligible'] for x in r));self.assertEqual(design()['independent_confirmation_states'],0)
    def test_coordinate_projection_norm_and_row_component(self):
        d=data139();pts,rr=freeze_kernel_coordinates(d)
        for r in rr:self.assertAlmostEqual(r['actual_distance'],r['norm'],5);self.assertLess(r['ideal_shared_row_component_Linf'],1e-12);self.assertLess(r['committed_shared_row_component_Linf'],1e-6)
        self.assertEqual(len(pts),24)
    def test_coordinate_reproducibility_no_head_mutation(self):
        d=data139();h=head_digests(d);a,r=freeze_kernel_coordinates(d);b,t=freeze_kernel_coordinates(d);compare(r,t);compare(head_digests(d),h)
        for sid in a:exact(a[sid],b[sid],'Reproducible coordinates')
    def test_registry_seed_norm_excluded_or_role_change_blocks(self):
        for key,v in (('seed',1),('norm',.01),('case','xor_10'),('dataset_role','confirmation')):
            rr=copy.deepcopy(registry());rr[0][key]=v
            with self.assertRaises(ValueError):freeze_kernel_coordinates(data139(),rr)
    def test_nonorthonormal_basis_blocks_without_refit(self):
        d=data139();d['V'][0]*=2
        with self.assertRaisesRegex(ValueError,'orthonormal'):freeze_kernel_coordinates(d)
    def test_nonfinite_basis_blocks(self):
        d=data139();d['V'][0,0]=np.nan
        with self.assertRaises(ValueError):freeze_kernel_coordinates(d)
    def test_collapsed_kernel_blocks_no_resampling(self):
        d=data139()
        with patch('core139.np.random.default_rng') as p:
            p.return_value.normal.return_value=np.r_[np.ones(len(d['V'])),np.zeros(DIM-len(d['V']))]
            with self.assertRaisesRegex(ValueError,'Collapsed'):freeze_kernel_coordinates(d)
    def test_coordinate_collision_blocks_no_replacement(self):
        d=data139();p,_=freeze_kernel_coordinates(d);d['historical_hashes'].add(safe.tensor_hash(next(iter(p.values()))))
        with self.assertRaisesRegex(ValueError,'collision'):freeze_kernel_coordinates(d)
    def test_C138_archive_reconstructs24_preparations96_gates(self):
        r=self.audit();self.assertEqual(r['all24_preparation_gates']['n_evaluated'],24);self.assertEqual(sum(x['n_states'] for x in r['descriptive_strata'].values()),24);self.assertFalse(r['parent_decision_reclassified']);self.assertFalse(r['subset_requalification_performed'])
    def test_C138_preparation_flag_tamper_blocks_even_rebound(self):
        def f(a):a[0]['preparation_valid']=not a[0]['preparation_valid'];return a
        with self.assertRaises(ValueError):self.audit(self.modified('PREWRITE_PROGRESS.json',f))
    def test_C138_coordinate_tamper_blocks_even_rebound(self):
        def f(a):a[next(iter(a))][0]+=.01;return a
        with self.assertRaises(ValueError):self.audit(self.modified('ALL24_PREPARED_COORDINATES_BEFORE_MODEL.npz',f))
    def test_C138_source_tamper_blocks_even_rebound(self):
        p=self.modified('source/PROTOCOL.json',lambda a:dict(a,new_bad_field=True))
        with self.assertRaises(ValueError):self.audit(p)
    def test_C138_gate_tamper_blocks_even_rebound(self):
        def f(a):a[0]['recomputed_gates']['complete_pass']=not a[0]['recomputed_gates']['complete_pass'];return a
        with self.assertRaises(ValueError):self.audit(self.modified('ALL96_CONFIRMATION_ENDPOINTS.json',f))
    def test_C138_head_tamper_blocks_even_rebound(self):
        def f(a):a['a'][0]+=.01;return a
        with self.assertRaises(ValueError):self.audit(self.modified('FROZEN_C137_CONTROL.npz',f))
    def test_invalid_preparation_retained_in_all96_endpoint_requests(self):
        s,out=self.execute('prep_invalid');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['preparation_valid_both_rounds'],0);r=clean_json((out/'ALL96_KERNEL_ENDPOINTS.json').read_bytes());self.assertEqual(len(r),96)
        for x in r:
            if not x['preparation_valid_both_rounds']:self.assertIn('preparation',x['recomputed_gates']['failed_gates']);self.assertFalse(x['recomputed_gates']['complete_pass'])
    def test_preparation_repeat_drift_retained_blocks_positive(self):
        s,_=self.execute('prep_repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['kernel_preparations_all24_passed']);self.assertEqual(s['final_endpoint_records'],96)
    def test_endpoint_repeat_drift_blocks_positive(self):
        s,_=self.execute('repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['frozen_writer_all24_development_gates_passed'])
    def test_feature_drift_stops_before_kernel_measurements(self):
        s,_=self.execute('feature_drift');self.assertFalse(s['complete']);self.assertEqual(s['kernel_preparation_records'],0)
    def test_known_drift_stops_before_kernel_measurements(self):
        s,_=self.execute('control_drift');self.assertFalse(s['complete']);self.assertEqual(s['kernel_preparation_records'],0)
    def test_C138_full_replay_drift_stops_before_kernel_measurements(self):
        s,_=self.execute('full_drift');self.assertFalse(s['complete']);self.assertEqual(s['kernel_preparation_records'],0);self.assertEqual(s['C138_preparation_replays'],0)
    def test_interruption_keeps_requested_denominator_and_sealed48(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['requested_endpoint_records'],96);self.assertEqual(s['final_endpoint_records'],5);self.assertTrue((out/'ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz').exists())
    def test_candidate_mutation_blocks_before_endpoints(self):
        s,_=self.execute('mutate_candidate');self.assertFalse(s['complete']);self.assertEqual(s['final_endpoint_records'],0)
    def test_no_solver_svd_fit_or_new_gradient(self):
        with safe.no_solvers():s,_=self.execute('no_solvers')
        self.assertTrue(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_Jacobians'],0)
    def phase(self):
        d=data139();k,_=freeze_kernel_coordinates(d);return Phase(d,k),d,k
    def test_early_phase_transition_blocks(self):
        p,d,k=self.phase()
        with self.assertRaises(ValueError):p.advance('full_replay')
    def test_unauthorized_native_access_blocks(self):
        p,d,k=self.phase()
        with self.assertRaises(ValueError):p.consume(next(iter(k.values())),'xor_00',True)
    def test_early_kernel_features_blocks(self):
        p,d,k=self.phase();r=registry()[0]
        with self.assertRaises(ValueError):p.feature(r['state_id'],k[r['state_id']],0)
    def test_early_endpoint_freeze_blocks(self):
        p,d,k=self.phase()
        with self.assertRaises(ValueError):p.freeze({})
    def test_wrong_original_hash_blocks_before_model(self):
        c=copy.deepcopy(self.c);c['pins']['C138']['sha256']='0'*64
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.fresh()/'out',c,Toy)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_response_scalar_forgery_recomputed(self):
        d=data139();sp=registry()[0];theta=d['C138_fresh'][sp['source_state_id']];m=metric(theta,sp['case'],False);m['response_drift']+=.001
        with self.assertRaises(ValueError):preparation_gates(m,sp['case'],theta)
    def test_invalid_preparation_never_endpoint_success(self):
        d=data139();sp=parent.registry()[0];p=d['points'][sp['state_id']];m=metric(p['raw'],sp['case'],True);g=previous.gate_record(m,sp['case'],p['raw'],p['prepared'],False);self.assertFalse(g['complete_pass']);self.assertIn('preparation',g['failed_gates'])
    def test_no_unseen_or_LC2_claim_even_on_full_synthetic_success(self):
        s,_=self.execute();self.assertFalse(s['candidate_selected']);self.assertFalse(s['writer_deployed']);self.assertEqual(s['independent_confirmation_states_evaluated'],0);self.assertEqual(s['excluded_input_states_evaluated'],0);self.assertFalse(s['LC2_single_fold_passed'])

if __name__=='__main__':unittest.main(verbosity=2)
