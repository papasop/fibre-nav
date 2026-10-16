"""Synthetic identities, truncation and acceptance accounting; not GPT-2 data."""
import io,json,tempfile,unittest,zipfile,subprocess,sys
import numpy as np
from c34_protocol import *
from c34_math import construct,proposal,aggregate

class Tests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(340017)
        self.js=[self.rng.normal(size=(6,30)) for _ in range(3)]
    def test_stack_control_full_row_rank(self):
        b,d=construct(self.js);A=np.concatenate(self.js)
        np.testing.assert_allclose(A@b['response6'],np.tile(np.eye(6),(3,1)),atol=2e-14)
        self.assertEqual(d['numerical_rank'],18)
    def test_common_is_sum_of_response_basis(self):
        b,_=construct(self.js);np.testing.assert_allclose(b['common1'][:,0],b['response6']@np.ones(6))
    def test_common_unit_control(self):
        b,_=construct(self.js);np.testing.assert_allclose(np.concatenate(self.js)@b['common1'],1.,atol=3e-14)
    def test_repeated_jacobian_rank6(self):
        J=np.pad(np.eye(6),((0,0),(0,4)));b,d=construct([J,J,J])
        self.assertEqual(d['numerical_rank'],6);np.testing.assert_allclose(J@b['response6'],np.eye(6),atol=1e-14)
    def test_zero_geometry_not_marked_feasible(self):
        b,d=construct([np.zeros((6,10))]*3);self.assertEqual(d['numerical_rank'],0)
        self.assertEqual(d['response6_stack_target_error_Linf'],1.);self.assertTrue(d['numerical_rank_is_not_feasibility_certificate'])
    def test_small_singular_values_truncated(self):
        J=np.pad(np.diag([1.,1.,1.,1.,1.,1e-14]),((0,0),(0,2)))
        _,d=construct([J]*3);self.assertEqual(d['numerical_rank'],5)
    def test_incompatible_shapes_rejected(self):
        with self.assertRaises(ValueError):construct([np.ones((6,10)),np.ones((6,11)),np.ones((6,10))])
    def test_nonfinite_geometry_rejected(self):
        self.js[0][0,0]=np.nan
        with self.assertRaises(ValueError):construct(self.js)
    def test_three_states_required(self):
        with self.assertRaises(ValueError):construct(self.js[:2])
    def test_response_oracle_negative_residual(self):
        b={'response6':np.eye(6),'common1':np.ones((6,1))};e=np.arange(6)*.001
        d,r=proposal(b,'response6_oracle',e);np.testing.assert_allclose(d,-e);self.assertTrue(r['measured_raw_response_used'])
    def test_common_oracle_leaves_contrast(self):
        b={'response6':np.eye(6),'common1':np.ones((6,1))};e=np.arange(6)*.001
        d,_=proposal(b,'common1_oracle',e);np.testing.assert_allclose(e+d,e-e.mean())
    def test_radius_clip_actual_norm(self):
        b={'response6':100*np.eye(6),'common1':np.ones((6,1))};d,r=proposal(b,'response6_oracle',np.ones(6))
        self.assertTrue(r['clipped']);self.assertAlmostEqual(np.linalg.norm(d),.05)
    def test_zero_residual_zero_increment(self):
        b={'response6':np.eye(6),'common1':np.ones((6,1))};d,r=proposal(b,'response6_oracle',np.zeros(6))
        np.testing.assert_array_equal(d,0);self.assertFalse(r['clipped'])
    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):proposal({},'native',np.zeros(6))
    def test_nonfinite_residual_rejected(self):
        with self.assertRaises(ValueError):proposal({},'response6_oracle',np.full(6,np.nan))
    def test_target_does_not_modify_basis(self):
        b,_=construct(self.js);old={k:v.copy() for k,v in b.items()};proposal(b,'response6_oracle',np.ones(6))
        for k in b:np.testing.assert_array_equal(b[k],old[k])
    def test_basis_ids_fixed_and_balanced(self):
        ids=SETTINGS['basis_training_ids'];self.assertEqual(len(set(ids)),12)
        self.assertTrue(all(sum(x.startswith(c+'__') for x in ids)==3 for c in CASES))
    def test_group_never_promotes_development(self):
        self.assertEqual(group({'cohort':'prospective','id':'xor_00__c31_new00'}),'inspected_development')
    def test_gate_counts_keep_invalids_and_lost_passes(self):
        records=[]
        for valid in (True,False):
            arms={}
            for arm in ARMS:
                gates={g:True for g in GATES};gates['response']=arm!='common1_oracle'
                arms[arm]={'metrics':{'gates':gates,'pass':all(gates.values()),'drift':.003 if not gates['response'] else .001}}
            records.append({'preparation_valid':valid,'arms':arms})
        a=aggregate(records);self.assertEqual(a['raw']['n_requested'],2);self.assertEqual(a['raw']['complete_passed'],1)
        self.assertEqual(a['common1_oracle']['lost_complete_passes'],1);self.assertEqual(a['raw']['gate_counts']['bits_match'],2)
    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):aggregate([])
    def test_archive_tamper_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('a','wrong');z.writestr('MANIFEST.json',json.dumps({'a':hashlib.sha256(b'right').hexdigest()}))
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):verify(z,True)
    def test_unsafe_path_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('../bad','x');z.writestr('MANIFEST.json','{}')
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):verify(z,True)
    def test_backup_retains_three_and_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);out=r/'s/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL});dump(out/'parent.json',{'v':1});h=sha(out/'parent.json')
            for _ in range(4):snapshot(out,r/'drive')
            self.assertEqual(len(list((r/'drive').glob('*.zip'))),3);self.assertEqual(sha(out/'parent.json'),h)
    def test_no_native_or_fresh_claim(self):
        self.assertFalse(SETTINGS['LC1_closed']);self.assertFalse(SETTINGS['fresh_evaluation']);self.assertFalse(SETTINGS['correction_free_writer_tested'])
    def test_clean_process_imports_launcher_with_root_only(self):
        root=Path(__file__).resolve().parent
        script='import sys;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));import launch_c34,c30_core,process_io;assert Path(c30_core.__file__)==root/"inherited/c30_core.py";assert launch_c34.ROOT==root;print("C34_CLEAN_ROOT_ONLY_IMPORT_OK")'
        result=subprocess.run([sys.executable,'-I','-c',script,str(root)],text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('C34_CLEAN_ROOT_ONLY_IMPORT_OK',result.stdout)

if __name__=='__main__':unittest.main()
