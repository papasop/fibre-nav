import contextlib,copy,io,tempfile,unittest
from unittest.mock import patch
from core138 import *
from worker138 import run
from fixture138 import worker_data,dummy_inputs,Toy,prior_archive_fixture
from inputs138 import audit137,validate_frozen_inputs

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c=dummy_inputs(cls.root);cls.counter=0
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def fresh(self):type(self).counter+=1;p=self.root/f'case{self.counter}';p.mkdir();return p
    def execute(self,mode='normal',data=None):
        data=worker_data() if data is None else data;out=self.fresh()/'out'
        def loader(paths,out,c):
            dump(out/'FROZEN_WRITER_SCHEMA_VALIDATION.json',validate_frozen_inputs(data['writer'],data['control']));return data
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *args:Toy(*args,mode=mode),loader)
        return s,out
    def ready(self):
        data=worker_data();fresh,_=fresh_coordinates(data);phase=Phase(data,fresh)
        for sp in parent.registry():
            sid=sp['state_id'];p=data['points'][sid];case=sp['case'];desired=int(case[-2])^int(case[-1]);phase.feature(sid,p['prepared'],desired);phase.consume_feature(p['prepared'],desired)
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('candidate',data['C137_points'][sid],True)):phase.metric((sid,kind),point,case,write);phase.consume(point,case,write)
        phase.begin_prewrite();return data,fresh,phase
    def test_full24_prewrite_and96_native_endpoint_records(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],421);self.assertEqual(s['final_endpoint_records'],96);self.assertEqual(s['confirmation_states_evaluated'],24);self.assertEqual(s['new_backward_calls'],0);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertFalse(s['real_model_execution_claim']);self.assertNotIn('torch',sys.modules)
        rows=clean_json((out/'ALL96_CONFIRMATION_ENDPOINTS.json').read_bytes());self.assertEqual(len(rows),96);self.assertEqual(len({r['state_id'] for r in rows}),24);self.assertTrue(s['all_proposals_frozen_before_new_endpoint_measurements'])
        with np.load(out/'FROZEN_C137_CONTROL.npz',allow_pickle=False) as f:exact(f['a'],worker_data()['a'],'Candidate changed')
        for n in ('new_training_steps','new_head_fits','new_Jacobians','new_SVD_calls','linear_systems_solved'):self.assertEqual(s[n],0)
    def test_invalid_preparations_remain_in_both_rounds_and_denominators(self):
        s,out=self.execute();self.assertTrue(s['complete']);rows=clean_json((out/'ALL96_CONFIRMATION_ENDPOINTS.json').read_bytes());bad=[r for r in rows if not r['preparation_valid']];self.assertGreater(len(bad),0)
        self.assertTrue(all(r['status']=='EVALUATED' and 'preparation' in r['recomputed_gates']['failed_gates'] and not r['recomputed_gates']['complete_pass'] for r in bad));self.assertFalse(s['local_confirmation_passed']);self.assertEqual(s['preparation_states_passed']+s['preparation_states_failed'],24)
        self.assertEqual(len(bad),4*s['preparation_states_failed'])
    def test_registry_has24_new_ids_seeds_balanced_seen_classes(self):
        rows=registry();self.assertEqual(len(rows),24);self.assertEqual(len({r['seed'] for r in rows}),24);self.assertEqual(Counter(r['case'] for r in rows),dict(xor_00=8,xor_01=8,xor_11=8));self.assertTrue(all(r['dataset_role']=='confirmation' for r in rows))
    def test_fresh_coordinates_reproducible_and_heads_unchanged(self):
        d=worker_data();h=head_digests(d);a,ra=fresh_coordinates(d);b,rb=fresh_coordinates(d);compare(ra,rb)
        for sid in a:exact(a[sid],b[sid],'Nonrepeatable generation')
        compare(head_digests(d),h);self.assertEqual(len({safe.tensor_hash(v) for v in a.values()}),24)
    def test_committed_noise_has_declared_distance(self):
        _,rows=fresh_coordinates(worker_data())
        for row in rows:self.assertAlmostEqual(row['actual_distance'],row['norm'],5)
    def test_seed_norm_or_case_registry_change_rejected(self):
        for key,value in (('seed',123),('norm',.25),('case','xor_10')):
            rows=copy.deepcopy(registry());rows[0][key]=value
            with self.assertRaises(ValueError):fresh_coordinates(worker_data(),rows)
    def test_historical_coordinate_collision_blocks_without_resampling(self):
        d=worker_data();p,_=fresh_coordinates(d);d['historical_hashes'].add(safe.tensor_hash(next(iter(p.values()))))
        with self.assertRaisesRegex(ValueError,'collision'):fresh_coordinates(d)
    def test_nonfinite_parent_blocks(self):
        d=worker_data();next(iter(d['points'].values()))['prepared'][0]=np.nan
        with self.assertRaises(ValueError):fresh_coordinates(d)
    def test_prewrite_forbidden_before_all_known_controls(self):
        d=worker_data();fresh,_=fresh_coordinates(d);phase=Phase(d,fresh)
        with self.assertRaises(ValueError):phase.begin_prewrite()
    def test_unauthorized_native_metric_blocks(self):
        d=worker_data();fresh,_=fresh_coordinates(d);phase=Phase(d,fresh)
        with self.assertRaises(ValueError):phase.consume(next(iter(fresh.values())),'xor_00',True)
    def test_fresh_feature_forbidden_during_controls(self):
        d=worker_data();fresh,_=fresh_coordinates(d);phase=Phase(d,fresh);r=registry()[0]
        with self.assertRaises(ValueError):phase.feature(r['state_id'],fresh[r['state_id']],0)
    def test_wrong_desired_bit_blocks(self):
        d,fresh,p=self.ready();r=registry()[0]
        with self.assertRaises(ValueError):p.feature(r['state_id'],fresh[r['state_id']],1)
    def test_feature_point_substitution_blocks(self):
        d,fresh,p=self.ready();r=registry()[0];wrong=fresh[r['state_id']].copy();wrong[0]+=.001
        with self.assertRaises(ValueError):p.feature(r['state_id'],wrong,0)
    def test_pending_feature_must_be_consumed(self):
        d,fresh,p=self.ready();r=registry()[0];theta=fresh[r['state_id']];p.feature(r['state_id'],theta,0)
        with self.assertRaises(ValueError):p.metric(r['state_id'],theta,r['case'],False)
    def test_duplicate_feature_request_blocks(self):
        d,fresh,p=self.ready();r=registry()[0];theta=fresh[r['state_id']];p.feature(r['state_id'],theta,0);p.consume_feature(theta,0)
        with self.assertRaises(ValueError):p.feature(r['state_id'],theta,0)
    def test_early_freeze_blocks_missing_prewrite_records(self):
        d,fresh,p=self.ready()
        with self.assertRaises(ValueError):p.freeze({})
    def test_endpoint_access_before_freeze_blocks(self):
        d,fresh,p=self.ready();r=registry()[0]
        with self.assertRaises((ValueError,StopIteration,KeyError)):p.metric((0,r['state_id'],ARMS[1]),fresh[r['state_id']],r['case'],True)
    def test_feature_drift_stops_before_new_measurements(self):
        s,_=self.execute('feature_drift');self.assertFalse(s['complete']);self.assertEqual(s['prewrite_records'],0);self.assertEqual(s['final_endpoint_records'],0)
    def test_known_native_drift_stops_before_new_measurements(self):
        s,_=self.execute('control_drift');self.assertFalse(s['complete']);self.assertEqual(s['prewrite_records'],0);self.assertEqual(s['new_backward_calls'],0)
    def test_interruption_retains96_requested_denominator(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['requested_endpoint_records'],96);self.assertEqual(s['final_endpoint_records'],5);self.assertFalse(s['LC2_closed']);self.assertTrue((out/'ALL48_PROPOSALS_BEFORE_ENDPOINTS.npz').is_file())
    def test_repeat_drift_blocks_qualification(self):
        s,_=self.execute('repeat_drift');self.assertTrue(s['complete']);self.assertFalse(s['repeatable']);self.assertFalse(s['local_confirmation_passed']);self.assertIn('REPEAT_DRIFT',s['decision'])
    def test_zero_control_reproduces_raw_and_frozen_nonzero_map(self):
        d=worker_data();p=next(iter(d['points'].values()));raw=p['raw'];zero=parent.proposal(p['prepared'],raw,d['V'],np.zeros(6))[0];exact(zero,raw,'Zero must be raw')
        modified=parent.proposal(p['prepared'],raw,d['V'],d['a'])[0];self.assertFalse(np.array_equal(modified,raw))
    def test_preparation_failure_cannot_be_relabelled_as_endpoint_success(self):
        d=worker_data();sp=parent.registry()[0];p=d['points'][sp['state_id']];m=d['C137_controls'][(sp['state_id'],parent.EVAL_ARMS[1],0)]['metrics'];g=gate_record(m,sp['case'],d['C137_points'][sp['state_id']],p['prepared'],False);self.assertFalse(g['complete_pass']);self.assertIn('preparation',g['failed_gates'])
    def test_full_worker_uses_no_new_solver_or_SVD(self):
        with patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')),patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')):s,_=self.execute()
        self.assertTrue(s['complete'],s['errors'])
    def test_novelty_scope_does_not_claim_exhaustive_programme_history(self):
        self.assertIn('not claimed',design()['novelty']);self.assertEqual(design()['excluded_input_states_evaluated'],0)
    def test_candidate_mutation_blocks_before_new_endpoints(self):
        s,_=self.execute('mutate_candidate');self.assertFalse(s['complete']);self.assertEqual(s['final_endpoint_records'],0);self.assertIn('Frozen candidate tensors',' '.join(s['errors']))
    def test_wrong_original_hash_blocks_before_model(self):
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C137']['sha256']='0'*64;out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,c,Toy)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('Wrong fixed original C137',' '.join(s['errors']))
    def test_invalid_scientific_zip_blocks_before_model(self):
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.fresh()/'out',self.c,Toy)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_sealed_endpoint_coordinate_change_and_postwrite_features_block(self):
        d,fresh,p=self.ready();points={}
        for sp in registry():
            sid=sp['state_id'];theta=fresh[sid];case=sp['case'];desired=int(case[-2])^int(case[-1]);p.metric(sid,theta,case,False);p.consume(theta,case,False);p.feature(sid,theta,desired);p.consume_feature(theta,desired)
            for arm in ARMS:points[(sid,arm)]=theta.copy()
        p.freeze(points);sp=registry()[0];point=fresh[sp['state_id']].copy();point[0]+=.001
        with self.assertRaises(ValueError):p.metric((0,sp['state_id'],ARMS[1]),point,sp['case'],True)
        with self.assertRaises(ValueError):p.feature(sp['state_id'],fresh[sp['state_id']],0)
    def test_missing_endpoint_cannot_finish(self):
        d,fresh,p=self.ready();p.phase='evaluation'
        with self.assertRaises(ValueError):p.finish()
    def test_new_coordinate_mutation_after_seal_blocks(self):
        d,fresh,p=self.ready();r=registry()[0];fresh[r['state_id']][0]+=.001
        with self.assertRaises(ValueError):p.metric(r['state_id'],fresh[r['state_id']],r['case'],False)

class WriterSchemaChecks(unittest.TestCase):
    def extended(self):
        d=worker_data();w=d['writer'];w.update(J_mean=np.zeros((22,DIM)),B_all=np.zeros((DIM,22)),B_response=np.zeros((DIM,6)),response_offset=np.zeros(6));return d
    def test_original_nine_field_schema_preserved_and_shape_checked(self):
        d=self.extended();before=safe.head_digest(d['writer']);report=validate_frozen_inputs(d['writer'],d['control']);self.assertEqual(len(report['actual_fields']),9);self.assertEqual(report['retained_geometry_fields'],['B_all','B_response','J_mean','response_offset']);self.assertEqual(before,safe.head_digest(d['writer']));self.assertFalse(report['geometry_extensions_used_in_raw_prediction'])
    def test_minimal_five_field_fixture_remains_supported(self):
        d=worker_data();self.assertEqual(validate_frozen_inputs(d['writer'],d['control'])['retained_geometry_fields'],[])
    def test_missing_core_field_reports_missing_and_actual_keys(self):
        d=worker_data();del d['writer']['mean']
        with self.assertRaisesRegex(ValueError,"missing=.*mean.*actual="):validate_frozen_inputs(d['writer'],d['control'])
    def test_unknown_extension_does_not_silently_enter_prediction(self):
        d=worker_data();d['writer']['unregistered']=np.zeros(1)
        with self.assertRaisesRegex(ValueError,'unknown=.*unregistered'):validate_frozen_inputs(d['writer'],d['control'])
    def test_wrong_geometry_extension_shapes_block(self):
        for name in ('J_mean','B_all','B_response','response_offset'):
            d=self.extended();d['writer'][name]=np.zeros(1)
            with self.assertRaises(ValueError):validate_frozen_inputs(d['writer'],d['control'])
    def test_nonfinite_in_unused_geometry_is_rejected(self):
        d=self.extended();d['writer']['B_all'][0,0]=np.nan
        with self.assertRaises(ValueError):validate_frozen_inputs(d['writer'],d['control'])
    def test_nonfloating_geometry_and_nonpositive_scale_block(self):
        d=self.extended();d['writer']['response_offset']=np.zeros(6,np.int64)
        with self.assertRaises(ValueError):validate_frozen_inputs(d['writer'],d['control'])
        d=self.extended();d['writer']['scale'][0]=0
        with self.assertRaises(ValueError):validate_frozen_inputs(d['writer'],d['control'])
    def test_unused_extension_values_cannot_change_raw_or_controlled_proposal(self):
        d=self.extended();sp=parent.registry()[0];p=d['points'][sp['state_id']];capture=p['features'][:3147];raw,full=safe.reconstruct_raw(d['writer'],d['control']['delta_sketch'],p['prepared'],capture);point=parent.proposal(p['prepared'],raw,d['V'],d['a'])[0]
        for name in ('J_mean','B_all','B_response','response_offset'):d['writer'][name].fill(123.)
        validate_frozen_inputs(d['writer'],d['control']);new,newfull=safe.reconstruct_raw(d['writer'],d['control']['delta_sketch'],p['prepared'],capture);newpoint=parent.proposal(p['prepared'],new,d['V'],d['a'])[0];exact(new,raw,'Extension changed raw');exact(newfull,full,'Extension changed features');exact(newpoint,point,'Extension changed candidate')
    def test_full_worker_retains_all_geometry_arrays_and_frozen_digest(self):
        d=self.extended()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,c=dummy_inputs(root);out=root/'out'
            def loader(paths,out,c):dump(out/'FROZEN_WRITER_SCHEMA_VALIDATION.json',validate_frozen_inputs(d['writer'],d['control']));return d
            with contextlib.redirect_stdout(io.StringIO()):s=run(paths,out,c,Toy,loader)
            self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['final_endpoint_records'],96)
            with np.load(out/'FROZEN_RAW_WRITER.npz',allow_pickle=False) as f:
                self.assertEqual(set(f.files),set(d['writer']))
                for key in f.files:exact(f[key],d['writer'][key],'Dropped or changed field/'+key)

class ArchiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c=prior_archive_fixture(cls.root/'fixture')
        from inputs137 import load as load137
        (cls.root/'old_bindings').mkdir()
        with contextlib.redirect_stdout(io.StringIO()):cls.data=load137({k:cls.paths[k] for k in parent.INPUT_KEYS},cls.root/'old_bindings',cls.c)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_original_C137_all24_updates_and48_endpoints_audited_without_model(self):
        with contextlib.closing(Archive(self.paths['C137'])) as z:audit137(z,copy.deepcopy(self.data),self.root/'receipt',self.c)
        receipt=clean_json((self.root/'receipt/C137_BOUND_RECEIPT.json').read_bytes());self.assertEqual(receipt['training_rows_checked'],288);self.assertEqual(receipt['final_endpoints_checked'],48);self.assertFalse(receipt['archived_gradients_independently_recomputed'])
    def test_C137_source_pin_change_blocks(self):
        c=dict(self.c,source137=dict(self.c['source137']));c['source137'][next(iter(c['source137']))]='0'*64
        with contextlib.closing(Archive(self.paths['C137'])) as z,self.assertRaises(ValueError):audit137(z,copy.deepcopy(self.data),self.root/'bad_source',c)
    def test_C137_requires_complete_training_qualification(self):
        import zipfile
        with zipfile.ZipFile(self.paths['C137']) as z:payload={n:z.read(n) for n in z.namelist()}
        s=json.loads(payload['SUMMARY.json']);s['fixed_final_training_qualification_passed']=False;payload['SUMMARY.json']=json.dumps(s).encode();payload['RESULT_MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items() if n!='RESULT_MANIFEST.json'}).encode();path=self.root/'invalid_parent.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for n,b in payload.items():z.writestr(n,b)
        with contextlib.closing(Archive(path)) as z,self.assertRaises(ValueError):audit137(z,copy.deepcopy(self.data),self.root/'bad_parent',self.c)
    def tampered(self,name,edit):
        import zipfile
        with zipfile.ZipFile(self.paths['C137']) as z:payload={n:z.read(n) for n in z.namelist()}
        payload[name]=edit(payload[name]);payload['RESULT_MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items() if n!='RESULT_MANIFEST.json'}).encode();path=self.root/(name.replace('/','_')+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for n,b in payload.items():z.writestr(n,b)
        with contextlib.closing(Archive(path)) as z,self.assertRaises(ValueError):audit137(z,copy.deepcopy(self.data),self.root/'tampered_receipt',self.c)
    def test_changed_C137_final_vector_blocks_even_rebound_manifest(self):
        def edit(b):
            with np.load(io.BytesIO(b),allow_pickle=False) as f:a=f['a'].copy()
            a[0]+=.001;out=io.BytesIO();np.savez_compressed(out,a=a);return out.getvalue()
        self.tampered('FINAL_SHARED_CONTROL_STEP24.npz',edit)
    def test_changed_C137_state_weight_blocks_even_rebound_manifest(self):
        def edit(b):
            rows=json.loads(b);rows[0]['aggregate_weight']+=.01;return json.dumps(rows).encode()
        self.tampered('training/step01/OBSERVATIONS.json',edit)
    def test_changed_C137_final_gate_blocks_even_rebound_manifest(self):
        def edit(b):
            rows=json.loads(b);rows[0]['recomputed_gates']['complete_pass']=not rows[0]['recomputed_gates']['complete_pass'];return json.dumps(rows).encode()
        self.tampered('ALL48_FINAL_TRAINING_ENDPOINTS.json',edit)

if __name__=='__main__':unittest.main(verbosity=2)
