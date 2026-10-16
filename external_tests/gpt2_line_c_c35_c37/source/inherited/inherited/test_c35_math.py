"""Synthetic calibration and leakage guards; not real model results."""
import io,json,sys,subprocess,tempfile,unittest,zipfile
from pathlib import Path
from c35_paths import configure
configure(Path(__file__).resolve().parent)
import numpy as np
from c35_protocol import *
from c35_math import fit,predict,select,choices,geometric
class Tests(unittest.TestCase):
    def setUp(self):
        r=np.random.default_rng(350011);self.g=r.normal(0,.002,(9,6));self.p=r.normal(0,.001,(9,6));self.y=self.g+.1*self.p+.0003
        self.ids=['xor_00__c31_train%02d'%i for i in range(3,12)]
    def test_geometry_control_is_unchanged(self):
        m=fit(self.g,self.p,self.y,choices()[0]);np.testing.assert_array_equal(predict(m,self.g,self.p),self.g)
    def test_ridge_fits_training_offset(self):
        y=self.g+.0002;m=fit(self.g,self.p,y,choices()[1]);np.testing.assert_allclose(predict(m,self.g,self.p),y,atol=1e-15)
    def test_primary_is_learned_ridge(self):
        m,a=select(self.g,self.p,self.y,self.ids);self.assertEqual(m['spec']['kind'],'ridge');self.assertEqual(len(a['candidates']),6)
    def test_prior_not_allowed_in_calibration_CV(self):
        ids=self.ids[:];ids[0]='xor_00__c31_train00'
        with self.assertRaises(ValueError):select(self.g,self.p,self.y,ids)
    def test_exact9_unique_ids_required(self):
        with self.assertRaises(ValueError):select(self.g,self.p,self.y,self.ids[:-1])
    def test_duplicate_ids_rejected(self):
        ids=self.ids[:];ids[0]=ids[1]
        with self.assertRaises(ValueError):select(self.g,self.p,self.y,ids)
    def test_validation_targets_do_not_change_fixed_candidate_predictions(self):
        _,a=select(self.g,self.p,self.y,self.ids);ff=np.array(folds(self.ids,3));y=self.y.copy();y[ff==0]+=.1
        _,b=select(self.g,self.p,y,self.ids)
        for x,z in zip(a['candidates'],b['candidates']):
            np.testing.assert_array_equal(np.array(x['calibration_CV_predictions'])[ff==0],np.array(z['calibration_CV_predictions'])[ff==0])
    def test_prediction_does_not_modify_fitted_normalisation(self):
        m=fit(self.g,self.p,self.y,choices()[1]);mu=m['mean'].copy();predict(m,self.g*100,self.p*100);np.testing.assert_array_equal(mu,m['mean'])
    def test_constant_features_are_finite(self):
        z=np.zeros((9,6));m=fit(z,z,np.ones((9,6))*.003,choices()[1]);np.testing.assert_allclose(predict(m,z,z),.003)
    def test_nonfinite_target_rejected(self):
        self.y[0,0]=np.nan
        with self.assertRaises(ValueError):fit(self.g,self.p,self.y,choices()[1])
    def test_nonfinite_prewrite_feature_rejected(self):
        self.p[0,0]=np.inf
        with self.assertRaises(ValueError):fit(self.g,self.p,self.y,choices()[1])
    def test_geometric_point_is_arithmetic(self):
        m={'J_mean':np.pad(np.eye(6),((0,0),(0,36858))),'point_center':np.zeros(36864),'response_center':np.ones(6)*.001}
        point=np.zeros(36864);point[:6]=.002;np.testing.assert_allclose(geometric(m,point),.003)
    def test_new_seed_specs_fixed_balanced(self):
        ss=specs();self.assertEqual(len(ss),16);self.assertEqual(len(set(x['seed'] for x in ss)),16)
        self.assertTrue(all(sum(x['case']==c for x in ss)==4 for c in CASES));self.assertEqual(ss[0]['seed'],51535001)
    def test_new_seeds_not_old_C31(self):
        from c31_protocol import specs as old
        self.assertFalse({s['seed'] for s in specs()}&{s['seed'] for s in old('new_training')+old('prospective')})
    def test_bootstrap_with_clean_root_only(self):
        root=Path(__file__).resolve().parent;code='import sys;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));import launch_c35,c30_core;assert launch_c35.ROOT==root;assert Path(c30_core.__file__)==root/"inherited/inherited/c30_core.py"'
        r=subprocess.run([sys.executable,'-I','-c',code,str(root)],capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    def test_archive_tamper(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('a','wrong');z.writestr('MANIFEST.json',json.dumps({'a':hashlib.sha256(b'right').hexdigest()}))
        with zipfile.ZipFile(b) as z:
            with self.assertRaises(ValueError):verify(z,True)
    def test_backup_retention_is_scoped(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);out=r/'s/run';out.mkdir(parents=True);dump(out/'identity.json',{'protocol':PROTOCOL});other=r/'drive/other.zip';other.parent.mkdir();other.write_bytes(b'other')
            for _ in range(4):snapshot(out,r/'drive')
            self.assertEqual(len(list(other.parent.glob(RESULT+'*.zip'))),3);self.assertEqual(other.read_bytes(),b'other')
    def test_invalid_finite_predictions_are_not_certificates(self):
        m=fit(self.g,self.p,self.y,choices()[1]);ans=predict(m,self.g,self.p);self.assertEqual(ans.shape,(9,6));self.assertFalse(SETTINGS['internal_parameter_update'])
if __name__=='__main__':unittest.main()
