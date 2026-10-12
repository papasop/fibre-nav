import ast,hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from c10_protocol import *
from geometry import null_project,ridge_correction
from checkpoint_numpy_audit import load_dense_checkpoint
ROOT=Path(__file__).parent

class Tests(unittest.TestCase):
    def test_all_training_classes(self):self.assertEqual({s.split('__')[0] for s in STATES},set(CASES));self.assertEqual(len(STATES),12)
    def test_new_ids_disjoint(self):
        ids={s['id'] for s in test_specs()};self.assertEqual(len(ids),8);self.assertFalse(ids&set(STATES))
    def test_new_seeds_disjoint_from_C9(self):
        old={109031+100*c+i for c in range(4) for i in range(2)}
        new={s['seed'] for s in test_specs()};self.assertEqual(len(new),8);self.assertFalse(old&new)
    def test_each_functional_update_covers_four_classes(self):
        for step in range(3):self.assertEqual({STATES[i].split('__')[0] for i in functional_indices(step)},set(CASES))
    def test_three_updates_cover_all_training_variants(self):
        self.assertEqual({STATES[i] for step in range(3) for i in functional_indices(step)},set(STATES))
    def test_fixed_gates(self):self.assertEqual([SETTINGS[k] for k in ['response_budget','kl_budget','margin_gate']],[.002,.005,.05])
    def test_response_training_scale_equals_budget(self):self.assertEqual(SETTINGS['response_loss_scale'],SETTINGS['response_budget'])
    def test_eval_repair_disabled(self):
        self.assertFalse(SETTINGS['writer_evaluation_online_Jacobian']);self.assertFalse(SETTINGS['writer_evaluation_online_SVD']);self.assertFalse(SETTINGS['writer_evaluation_response_correction'])
    def test_fixed_total_steps(self):self.assertEqual(SETTINGS['distill_steps']+SETTINGS['functional_steps'],1360)
    def test_all_pass_decision(self):
        ok={'state_usable':True,'distilled':{'episode_pass':True}}
        self.assertEqual(decision([ok]*12,[ok]*8),'C10_RAW_LOCAL_STATE_GENERALISATION_PASS')
    def test_reference_fail_not_dropped(self):
        ok={'state_usable':True,'distilled':{'episode_pass':True}}
        self.assertEqual(decision([ok]*12,[ok]*7+[{'state_usable':False}]),'C10_NEW_STATE_REFERENCES_INCONCLUSIVE')
    def test_training_failure_separate(self):
        ok={'state_usable':True,'distilled':{'episode_pass':True}};bad={'state_usable':True,'distilled':{'episode_pass':False}}
        self.assertEqual(decision([ok]*11+[bad],[ok]*8),'C10_RAW_TRAINING_FUNCTION_FAILURE')
    def test_raw_new_failure_reported(self):
        ok={'state_usable':True,'distilled':{'episode_pass':True}};bad={'state_usable':True,'distilled':{'episode_pass':False}}
        self.assertEqual(decision([ok]*12,[ok]*7+[bad]),'C10_RAW_LOCAL_STATE_GENERALISATION_FAIL')
    def test_denominator_mismatch(self):
        with self.assertRaises(ValueError):decision([],[])
    def test_writer_evaluation_no_repair_or_backward(self):
        text=(ROOT/'worker_c10.py').read_text();tree=ast.parse(text)
        f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate_raw')
        calls=[ast.unparse(n.func) for n in ast.walk(f) if isinstance(n,ast.Call)]
        self.assertIn('one_forward_proposal',calls)
        self.assertFalse(any(c in ('correct','jacobian','null_project','ridge_correction','torch.autograd.grad') or c.endswith('.backward') or c.endswith('.step') for c in calls))
        self.assertIn("phase['name']='writer_evaluation'",ast.get_source_segment(text,f))
    def test_geometry_phase_guards(self):
        text=(ROOT/'worker_c10.py').read_text();tree=ast.parse(text)
        for name in ['correct','jacobian']:
            f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertIn("phase['name']!='dataset_preparation'",ast.get_source_segment(text,f))
    def test_test_generation_after_freeze(self):
        text=(ROOT/'worker_c10.py').read_text();self.assertLess(text.index("dump(out/'WRITER_FROZEN.json'"),text.index('for si,t in enumerate(test_specs())'))
    def test_supervision_only_parent_training(self):
        text=(ROOT/'worker_c10.py').read_text();self.assertIn("load_obj(c9,'evaluation/training/'+sid+'/endpoint.pt')",text)
        self.assertNotIn("load_obj(c9,'evaluation/new/",text)
        self.assertNotIn('fit_normalisation(X,Y)',text)
    def test_no_teacher_fallback(self):
        text=(ROOT/'worker_c10.py').read_text();tree=ast.parse(text)
        f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='evaluate_raw')
        self.assertNotIn('teacher0',ast.get_source_segment(text,f));self.assertIn("'teacher_fallback_used':False",text)
    def test_freeze_checks_both_arms(self):
        text=(ROOT/'worker_c10.py').read_text();self.assertIn('head_hash(baseline)!=HEAD_SHA',text);self.assertIn('head_hash(head)!=frozen_head',text)
    def test_one_fixed_state_not_resampled(self):
        text=(ROOT/'worker_c10.py').read_text();self.assertNotIn('while ',text);self.assertIn("'generation_retries':0",text)
    def test_null_geometry_nonvacuous(self):
        p,_=null_project([[1,2,0]],[2,3,4]);self.assertLess(np.linalg.norm(np.array([[1,2,0]])@p),1e-12);self.assertGreater(np.linalg.norm(p),0)
    def test_reference_geometry_not_improvement_certificate(self):
        _,info=ridge_correction([[1,0]],[1]);self.assertFalse(info['actual_nonlinear_improvement_guaranteed'])
    def test_wrong_parent_digest_fails_first(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';p.write_bytes(b'bad')
            with self.assertRaises(ValueError):validate_c9(p)
    def test_source_zip_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('worker.py','print(1)')
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):verify_open_zip(z)
    def test_checkpoint_digest(self):
        b=io.BytesIO();blob=b'test'
        with zipfile.ZipFile(b,'w') as z:z.writestr('heads/distill/checkpoint.pt',blob)
        b.seek(0)
        with zipfile.ZipFile(b) as z:self.assertEqual(checkpoint_pointer(z,'heads/distill/',{'checkpoint':'checkpoint.pt','checkpoint_sha256':hashlib.sha256(blob).hexdigest(),'steps':25},1000),25)
    def test_checkpoint_digest_corruption(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('checkpoint.pt',b'test')
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):checkpoint_pointer(z,'',{'checkpoint':'checkpoint.pt','checkpoint_sha256':'0'*64,'steps':0},1000)
    def test_checkpoint_path_traversal(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('x',b'x')
        b.seek(0)
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):checkpoint_pointer(z,'',{'checkpoint':'../x.pt','checkpoint_sha256':'0'*64,'steps':0},1000)
    def test_numpy_checkpoint_reader_rejects_arbitrary_global(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('archive/data.pkl',b'cos\nsystem\n.')
        with self.assertRaises(ValueError):load_dense_checkpoint(b.getvalue())
    def test_full_outcome_requires_response_gate(self):
        value={'bits':[0,0,1,0],'evaluation':{'exact_margins':[1]*4,'heldout_margins':[1]*12},'drift':.1,'kl':.001}
        self.assertFalse(endpoint_gate(value,[0,0,1,0]))
        value['drift']=.001;self.assertTrue(endpoint_gate(value,[0,0,1,0]))

if __name__=='__main__':unittest.main()
