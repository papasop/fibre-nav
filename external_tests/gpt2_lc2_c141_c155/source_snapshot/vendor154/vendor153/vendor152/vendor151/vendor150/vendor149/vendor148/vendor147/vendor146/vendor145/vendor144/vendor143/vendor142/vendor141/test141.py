import contextlib,copy,io,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from core141 import *
from inputs141 import audit140,explicit_coordinate_hashes
from fixture141 import *
from worker141 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.paths,cls.z140,cls.initial,cls.c=archive140(cls.root/'parent');cls.base=copy.deepcopy(cls.initial);audit140(Archive(cls.z140),cls.base,cls.root,cls.c);cls.base['historical_hashes']|={safe.tensor_hash(p[k]) for p in cls.base['points'].values() for k in ('prepared','raw')};cls.base['historical_hashes']|={safe.tensor_hash(p) for p in cls.base['C140_points'].values()};cls.cache={}
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'out'
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *a:Toy(*a,mode),lambda paths,out,c:copy.deepcopy(self.base))
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def changed(self,n,fn):
        folder=self.fresh()
        with zipfile.ZipFile(self.z140) as z:z.extractall(folder)
        p=folder/n
        if p.suffix=='.json':dump(p,fn(clean_json(p.read_bytes())))
        else:
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            np.savez_compressed(p,**fn(a))
        return pack(folder,self.fresh()/'changed.zip')
    def audit(self,path):
        with contextlib.closing(Archive(path)) as z:audit140(z,copy.deepcopy(self.initial),self.fresh(),self.c)
    def test_full_worker48_states96_preps144_proposals288_endpoints1309_forwards(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['local_confirmation_passed']);self.assertEqual(s['preparation_records'],96);self.assertEqual(s['new_feature_records'],48);self.assertEqual(s['final_endpoint_records'],288);self.assertEqual(s['new_model_forwards'],1309);self.assertEqual(len(np.load(out/'ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz').files),144)
    def test_C140_source_and_arithmetic_reconstruct12_updates432_rows216_gates(self):
        self.audit(self.z140);self.assertEqual(len(self.base['C140_controls']),216);self.assertEqual(len(self.base['C140_points']),108)
    def test_C140_rebound_update_tamper_blocks(self):
        def f(a):a['after_a'][0]+=.001;return a
        with self.assertRaises(ValueError):self.audit(self.changed('training/step12/UPDATE.npz',f))
    def test_C140_rebound_final_head_tamper_blocks(self):
        def f(a):a['a'][0]+=.001;return a
        with self.assertRaises(ValueError):self.audit(self.changed('FINAL_SHARED_CONTROL_STEP12.npz',f))
    def test_C140_rebound_gradient_weight_tamper_blocks(self):
        def f(a):a[0]['aggregate_weight']+=.1;return a
        with self.assertRaises(ValueError):self.audit(self.changed('training/step01/OBSERVATIONS.json',f))
    def test_C140_rebound_endpoint_gate_tamper_blocks(self):
        def f(a):a[0]['recomputed_gates']['complete_pass']=False;return a
        with self.assertRaises(ValueError):self.audit(self.changed('ALL216_FINAL_TRAINING_ENDPOINTS.json',f))
    def test_C140_rebound_source_tamper_blocks(self):
        with self.assertRaises(ValueError):self.audit(self.changed('source/PROTOCOL.json',lambda d:dict(d,bad=True)))
    def test_registry48_unique_seeds_balanced_known_classes(self):
        rr=registry();self.assertEqual(len(rr),48);self.assertEqual(len({r['seed'] for r in rr}),48);self.assertEqual(Counter(r['case'] for r in rr),dict(xor_00=16,xor_01=16,xor_11=16));self.assertTrue(all(r['dataset_role']=='confirmation' and not r['used_for_training'] for r in rr));self.assertEqual(len({r['parent_id'] for r in rr}),12)
    def test_projection_reproducible_norm_and_heads_unchanged(self):
        d=copy.deepcopy(self.base);h=head_digests(d);a,ra=fresh_coordinates(d);b,rb=fresh_coordinates(d);compare(ra,rb);compare(head_digests(d),h)
        for r in ra:self.assertAlmostEqual(r['actual_distance'],r['norm'],5);self.assertLess(r['ideal_shared_row_component_Linf'],1e-12);exact(a[r['state_id']],b[r['state_id']],'Reproducible fresh state')
    def test_registry_seed_norm_case_or_role_change_blocks(self):
        for k,x in (('seed',1),('norm',.01),('case','xor_10'),('dataset_role','training')):
            rr=copy.deepcopy(registry());rr[0][k]=x
            with self.assertRaises(ValueError):fresh_coordinates(self.base,rr)
    def test_historical_coordinate_collision_blocks_without_resampling(self):
        d=copy.deepcopy(self.base);pts,_=fresh_coordinates(d);d['historical_hashes'].add(safe.tensor_hash(next(iter(pts.values()))))
        with self.assertRaisesRegex(ValueError,'collision'):fresh_coordinates(d)
    def test_nonorthonormal_or_nonfinite_basis_blocks(self):
        for val in (2.,np.nan):
            d=copy.deepcopy(self.base);d['V'][0,0]=val
            with self.assertRaises(ValueError):fresh_coordinates(d)
    def test_collapsed_direction_blocks_without_replacement(self):
        d=self.base
        with patch('core141.np.random.default_rng') as p:
            p.return_value.normal.return_value=np.r_[np.ones(len(d['V'])),np.zeros(DIM-len(d['V']))]
            with self.assertRaisesRegex(ValueError,'Collapsed'):fresh_coordinates(d)
    def test_known_feature_drift_blocks_before_new_preparations(self):
        s,_=self.execute('feature_drift');self.assertFalse(s['complete']);self.assertEqual(s['preparation_records'],0)
    def test_known_native_drift_blocks_before_new_preparations(self):
        s,_=self.execute('control_drift');self.assertFalse(s['complete']);self.assertEqual(s['preparation_records'],0)
    def test_mutated_candidate_blocks_before_new_endpoints(self):
        s,_=self.execute('mutate_candidate');self.assertFalse(s['complete']);self.assertEqual(s['final_endpoint_records'],0)
    def test_invalid_preparations_remain_all288_writer_requests(self):
        s,out=self.execute('invalid_preparation');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['preparation_states_valid_both_rounds'],0);self.assertEqual(s['final_endpoint_records'],288);rr=clean_json((out/'ALL288_CONFIRMATION_ENDPOINTS.json').read_bytes());self.assertTrue(all('preparation' in r['recomputed_gates']['failed_gates'] and not r['recomputed_gates']['complete_pass'] for r in rr));self.assertFalse(s['local_confirmation_passed'])
    def test_preparation_repeat_drift_retained_blocks_confirmation(self):
        s,_=self.execute('prep_repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['local_confirmation_passed']);self.assertEqual(s['final_endpoint_records'],288)
    def test_endpoint_repeat_drift_retained_blocks_confirmation(self):
        s,_=self.execute('repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['local_confirmation_passed'])
    def test_raw_or_C137_success_cannot_substitute_for_C140_failure(self):
        s,_=self.execute('candidate_failure');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['local_confirmation_passed']);q=s['confirmation_summaries']['all48']['0'];self.assertEqual(q[ARMS[0]]['passed'],48);self.assertEqual(q[ARMS[1]]['passed'],48);self.assertEqual(q[ARMS[2]]['passed'],47)
    def test_interruption_retains288_denominator_and_presealed144(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['final_endpoint_records'],5);self.assertEqual(s['requested_endpoint_records'],288);self.assertTrue((out/'ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz').exists());self.assertFalse(s['LC2_closed'])
    def test_no_solvers_fitting_or_new_derivatives(self):
        with safe.no_solvers():s,_=self.execute('no_solvers')
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0);self.assertEqual(s['new_Jacobians'],0)
    def phase(self):d=copy.deepcopy(self.base);fresh,_=fresh_coordinates(d);return Phase(d,fresh),d,fresh
    def test_early_preparations_prewrite_and_endpoint_freeze_block(self):
        p,d,f=self.phase()
        with self.assertRaises(ValueError):p.begin_preparations()
        with self.assertRaises(ValueError):p.begin_prewrite()
        with self.assertRaises(ValueError):p.freeze({})
    def test_unauthorized_native_and_postwrite_features_block(self):
        p,d,f=self.phase();r=registry()[0]
        with self.assertRaises(ValueError):p.consume(f[r['state_id']],r['case'],False)
        with self.assertRaises(ValueError):p.feature(r['state_id'],f[r['state_id']],0)
    def test_duplicate_known_native_access_blocks(self):
        p,d,f=self.phase();r=parent.registry()[0];pt=d['points'][r['state_id']]['prepared'];p.metric((r['state_id'],'prepared'),pt,r['case'],False);p.consume(pt,r['case'],False)
        with self.assertRaises(ValueError):p.metric((r['state_id'],'prepared'),pt,r['case'],False)
    def test_wrong_original_hash_blocks_before_model(self):
        c=copy.deepcopy(self.c);c['pins']['C140']['sha256']='0'*64
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.fresh()/'out',c,Toy)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_archived_explicit_hash_inventory_is_descriptive(self):
        with contextlib.closing(Archive(self.z140)) as z:hh=explicit_coordinate_hashes(z)
        self.assertIn(next(iter(self.base['C140_controls'].values()))['metrics']['theta_sha256'],hh);self.assertIn('No exhaustive',design()['novelty'])
    def test_original_optional_geometry_fields_preserved_in_head_digest(self):
        d=copy.deepcopy(self.base);h=head_digests(d);d['writer']['response_offset']=np.zeros(6);hh=head_digests(d);self.assertNotEqual(h['writer'],hh['writer']);d['writer']['response_offset'][0]=.01;self.assertNotEqual(hh['writer'],head_digests(d)['writer'])
    def test_success_is_local_confirmation_not_LC2_or_new_centers(self):
        s,_=self.execute();self.assertTrue(s['local_confirmation_passed']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertEqual(s['excluded_input_states_evaluated'],0);self.assertFalse(s['candidate_selected']);self.assertFalse(s['real_model_execution_claim']);self.assertEqual(design()['new_parent_centers'],0)

if __name__=='__main__':unittest.main(verbosity=2)
