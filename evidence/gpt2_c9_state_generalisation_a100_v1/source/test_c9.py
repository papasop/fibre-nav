import ast,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from c9_protocol import *
from geometry import null_project,ridge_correction
ROOT=Path(__file__).parent

class Tests(unittest.TestCase):
    def test_training_contains_all_four_classes(self):
        self.assertEqual({s.split('__')[0] for s in STATES},set(CASES));self.assertEqual(len(STATES),12)
    def test_new_state_split_fixed_and_disjoint(self):
        specs=test_specs();self.assertEqual(len(specs),8);self.assertEqual(len({s['id'] for s in specs}),8)
        self.assertFalse(set(STATES)&{s['id'] for s in specs})
    def test_test_seeds_unique(self):self.assertEqual(len({s['seed'] for s in test_specs()}),8)
    def test_two_predeclared_states_per_class(self):
        for c in CASES:self.assertEqual([s['norm'] for s in test_specs() if s['case']==c],[.375,.75])
    def test_thresholds_not_relaxed(self):self.assertEqual([SETTINGS[k] for k in ['margin_gate','response_budget','kl_budget']],[.05,.002,.005])
    def test_fixed_final_head(self):self.assertEqual(SETTINGS['warm_steps']+SETTINGS['functional_steps'],2240);self.assertIn('no test-based',SETTINGS['selection'])
    def test_requested_bit_explicit(self):self.assertTrue(SETTINGS['requested_output_bit_in_features']);self.assertFalse(SETTINGS['internal_parameter_update'])
    def test_zero_output_triviality_rejected(self):
        starts=[[0,0,1,1],[0,1,1,0],[1,0,1,0],[1,1,1,1]]
        for s in starts:self.assertNotEqual(target_from_reads(s)[3],s[3])
    def test_all_valid_states_pass_decision(self):
        ok={'state_usable':True,'corrected':{'episode_pass':True}}
        self.assertEqual(decision([ok]*12,[ok]*8),'C9_LOCAL_STATE_GENERALISATION_PASS')
    def test_invalid_reference_not_dropped(self):
        ok={'state_usable':True,'corrected':{'episode_pass':True}};bad={'state_usable':False}
        self.assertEqual(decision([ok]*12,[ok]*7+[bad]),'C9_NEW_STATE_REFERENCES_INCONCLUSIVE')
    def test_training_failure_separate_from_generalisation(self):
        ok={'state_usable':True,'corrected':{'episode_pass':True}};bad={'state_usable':True,'corrected':{'episode_pass':False}}
        self.assertEqual(decision([ok]*11+[bad],[ok]*8),'C9_TRAINING_FUNCTION_FAILURE')
    def test_new_failure_reported(self):
        ok={'state_usable':True,'corrected':{'episode_pass':True}};bad={'state_usable':True,'corrected':{'episode_pass':False}}
        self.assertEqual(decision([ok]*12,[ok]*7+[bad]),'C9_LOCAL_STATE_GENERALISATION_FAIL')
    def test_denominator_mismatch_rejected(self):
        with self.assertRaises(ValueError):decision([],[])
    def test_saved_jacobian_projection_nontrivial(self):
        J=np.array([[1.,2.,0.]]);p,info=null_project(J,[2,3,4]);self.assertLess(np.linalg.norm(J@p),1e-12);self.assertGreater(np.linalg.norm(p),0)
    def test_fixed_seed_reproduces_direction(self):
        s=test_specs()[0];a=np.random.default_rng(s['seed']).normal(size=50);b=np.random.default_rng(s['seed']).normal(size=50);self.assertTrue(np.array_equal(a,b))
    def test_response_correction_not_content_certification(self):
        d,info=ridge_correction([[1,0]],[1]);self.assertLess(abs(1+d[0]),1e-5);self.assertEqual(d[1],0);self.assertFalse(info['actual_nonlinear_improvement_guaranteed'])
    def test_generation_and_evaluation_after_freeze(self):
        t=(ROOT/'worker_c9.py').read_text();freeze=t.index("dump(out/'WRITER_FROZEN.json'")
        self.assertGreater(t.index('for si,t in enumerate(test_specs())'),freeze)
        self.assertGreater(t.index('training=[]'),freeze)
        self.assertIn("if not (out/'WRITER_FROZEN.json').exists()",t)
    def test_train_only_data_dependency(self):
        t=(ROOT/'worker_c9.py').read_text();fit=t.index("fit_stage('functional'");generation=t.index('for si,t in enumerate(test_specs())')
        self.assertLess(fit,generation);self.assertIn("X=torch.stack([r[2] for r in data])",t)
        self.assertIn('j=step%len(STATES)',t);self.assertIn('normalise(X,norm)',t)
    def test_no_teacher_fallback(self):
        t=(ROOT/'worker_c9.py').read_text();tree=ast.parse(t)
        f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate_writer')
        source=ast.get_source_segment(t,f);self.assertNotIn('teacher0',source);self.assertNotIn('teacher.pt',source)
        self.assertIn('one_forward_proposal',source)
    def test_no_test_retry_or_state_replacement(self):
        t=(ROOT/'worker_c9.py').read_text();self.assertIn("'generation_retries':0",t);self.assertIn("'state_replacement_used':False",t)
        self.assertNotIn('while ',t)
    def test_checkpoints_include_optimizer_and_norm(self):
        t=(ROOT/'worker_c9.py').read_text();self.assertIn("'optimiser':optimiser.state_dict()",t);self.assertIn("'normalisation':",t);self.assertIn("'start_head_sha256':start_hash",t)
    def test_input_wrong_digest_rejected_before_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'not parent')
            with self.assertRaises(ValueError):validate_c8(p)
    def test_archive_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('identity.json','{}');z.writestr('MANIFEST.json',json.dumps({'identity.json':'0'*64}))
            with self.assertRaises(ValueError):verified_zip(p)
    def test_source_launcher_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('start.py','print(1)')
            with self.assertRaises(ValueError):verified_zip(p)
    def test_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../bad','x')
            with self.assertRaises(ValueError):verified_zip(p)
    def test_checkpoint_digest_and_progress_valid(self):
        b=io.BytesIO();blob=b'checkpoint fixture'
        with zipfile.ZipFile(b,'w') as z:z.writestr('heads/warm/checkpoint_000025.pt',blob)
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            self.assertEqual(checkpoint_pointer(z,'heads/warm/',{'checkpoint':'checkpoint_000025.pt','checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'steps':25},2000),25)
    def test_checkpoint_corruption_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('heads/warm/checkpoint.pt',b'x')
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):checkpoint_pointer(z,'heads/warm/',{'checkpoint':'checkpoint.pt','checkpoint_sha256':'0'*64,'steps':25},2000)
    def test_checkpoint_pointer_traversal_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('placeholder',b'x')
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):checkpoint_pointer(z,'heads/warm/',{'checkpoint':'../bad.pt','steps':0,'checkpoint_sha256':'0'*64},2000)
    def test_checkpoint_steps_outside_stage_rejected(self):
        b=io.BytesIO();blob=b'x'
        with zipfile.ZipFile(b,'w') as z:z.writestr('checkpoint.pt',blob)
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):checkpoint_pointer(z,'',{'checkpoint':'checkpoint.pt','checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'steps':241},240)

if __name__=='__main__':unittest.main()
