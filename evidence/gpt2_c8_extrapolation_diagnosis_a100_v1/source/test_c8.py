import ast,hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from diagnostic_math import *
from c8_protocol import candidate_jobs,CASES,STATES,SETTINGS,verified_zip,PROTOCOL,snapshot,resume_score,source_hashes
ROOT=Path(__file__).parent

class Tests(unittest.TestCase):
    def test_feature_affine_span_distinguishes_linear_span(self):
        r=feature_coverage([[2,0],[2,1],[2,2]],[3,1]);self.assertAlmostEqual(r['affine_span']['residual_norm'],1.)
    def test_training_features_have_zero_affine_residual(self):
        X=np.array([[1.,0],[1,1],[1,2]])
        for q in X:self.assertLess(feature_coverage(X,q)['affine_span']['residual_norm'],1e-12)
    def test_identical_features_rank_zero(self):
        r=feature_coverage(np.ones((3,2)),[2,1]);self.assertEqual(r['svd']['rank_at_declared_cutoff'],0);self.assertEqual(r['affine_span']['residual_norm'],1)
    def test_cutoff_rejects_noise_only_rank(self):
        b,s=span_basis(np.eye(2)*1e-15);self.assertEqual(b.shape,(0,2));self.assertEqual(s['rank_at_declared_cutoff'],0)
    def test_near_singular_cutoff_applied(self):
        _,s=span_basis(np.diag([1,1e-12]));self.assertEqual(s['rank_at_declared_cutoff'],1)
    def test_span_projection_orthogonal(self):
        b,_=span_basis([[1,0,0],[2,0,0]]);self.assertEqual(span_residual(b,[2,3,0])['residual_norm'],3.)
    def test_zero_delta_is_not_nan(self):
        b,_=span_basis([[1,0]]);r=delta_comparison([0,0],[1,0],b);self.assertIsNone(r['cosine_with_teacher']);self.assertEqual(r['relative_teacher_error'],1)
    def test_exact_teacher_alignment(self):
        b,_=span_basis([[1,2]]);r=delta_comparison([1,2],[1,2],b);self.assertAlmostEqual(r['cosine_with_teacher'],1);self.assertEqual(r['error_norm'],0)
    def test_opposite_delta_alignment(self):
        b,_=span_basis([[1,0]]);self.assertEqual(delta_comparison([-1,0],[1,0],b)['cosine_with_teacher'],-1)
    def test_committed_delta_accounts_for_rounding(self):
        a=np.array([1e8],dtype=np.float32);b=a+np.float32(1);self.assertEqual(committed_delta(a,b)[0],0)
    def test_commit_requires_float32(self):
        with self.assertRaises(ValueError):committed_delta([1.],[2.])
    def test_nonlinear_remainder_nonvacuous(self):
        r=linear_vs_finite([[2]],[.3],[1],[1.3**2]);self.assertAlmostEqual(r['measured_remainder'][0],.09)
    def test_null_first_order_does_not_mean_zero_finite_drift(self):
        r=linear_vs_finite([[0]],[.5],[0],[.25]);self.assertEqual(r['linear_change'],[0]);self.assertEqual(r['actual_finite_change'],[.25])
    def test_nonfinite_diagnostics_rejected(self):
        with self.assertRaises(ValueError):feature_coverage([[1,2]],[np.nan,0])
    def test_shapes_rejected(self):
        with self.assertRaises(ValueError):feature_coverage([[1,2]],[0])
    def test_candidate_denominator_full_three_eligible(self):
        jobs=candidate_jobs([CASES[0],CASES[1],CASES[3]]);self.assertEqual(len(jobs),120);self.assertEqual(len({j['key'] for j in jobs}),120)
    def test_failed_fold_remains_in_frozen_baseline(self):
        jobs=candidate_jobs([CASES[0],CASES[1],CASES[3]])
        self.assertEqual(sum(j['fold']==CASES[2] and j['family']=='frozen_baseline' for j in jobs),12)
        self.assertEqual(sum(j['fold']==CASES[2] and j['family']=='paired' for j in jobs),0)
    def test_excluded_and_training_denominators(self):
        jobs=candidate_jobs([CASES[0],CASES[1],CASES[3]])
        self.assertEqual(sum(j['family']=='paired' and j['cohort']=='excluded_development' for j in jobs),18)
        self.assertEqual(sum(j['family']=='paired' and j['cohort']=='training' for j in jobs),54)
    def test_fixed_gates_and_no_new_training(self):
        self.assertEqual([SETTINGS[x] for x in ['response_budget','kl_budget','margin_gate']],[.002,.005,.05])
        self.assertEqual(SETTINGS['new_training_steps'],0);self.assertEqual(SETTINGS['new_correction_steps'],0)
    def test_worker_has_no_optimizer_or_correction(self):
        text=(ROOT/'worker_c8.py').read_text();tree=ast.parse(text)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any('optim' in c or c.endswith('.backward') or c.endswith('.step') or 'ridge_correction' in c or 'one_forward_proposal' in c for c in calls))
        self.assertIn('torch.autograd.grad',calls);self.assertIn('weights_only=True',text)
    def test_manifest_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('identity.json',json.dumps({'protocol':PROTOCOL}));z.writestr('MANIFEST.json',json.dumps({'identity.json':'0'*64}))
            with self.assertRaises(ValueError):verified_zip(p)
    def test_source_zip_rejected_as_result(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('launcher.py','print(1)')
            with self.assertRaises(ValueError):verified_zip(p)
    def test_unsafe_archive_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../x','x')
            with self.assertRaises(ValueError):verified_zip(p)
    def test_snapshot_resume_records_not_filename(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';out.mkdir();(out/'parent_c7.zip').write_bytes(b'parent fixture')
            (out/'identity.json').write_text(json.dumps({'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(ROOT),
                                                      'parent_c7_sha256':hashlib.sha256(b'parent fixture').hexdigest()}))
            (out/'candidates').mkdir();(out/'candidates'/'one.json').write_text('{}')
            p=snapshot(out);self.assertEqual(resume_score(p),(0,1,0))
    def test_mutated_parent_resume_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';out.mkdir();(out/'parent_c7.zip').write_bytes(b'changed')
            (out/'identity.json').write_text(json.dumps({'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':source_hashes(ROOT),'parent_c7_sha256':'0'*64}))
            p=snapshot(out)
            with self.assertRaises(ValueError):resume_score(p)

if __name__=='__main__':unittest.main()
