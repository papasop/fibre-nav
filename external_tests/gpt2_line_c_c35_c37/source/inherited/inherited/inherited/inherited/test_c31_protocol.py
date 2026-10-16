import hashlib, io, json, tempfile, unittest, zipfile
from pathlib import Path
from c31_protocol import *


class Tests(unittest.TestCase):
    def test_fixed_denominators(self):
        self.assertEqual(len(specs('new_training')),48);self.assertEqual(len(specs('prospective')),16)

    def test_disjoint_seeds_and_ids(self):
        a,b=specs('new_training'),specs('prospective')
        self.assertFalse({x['id'] for x in a}&{x['id'] for x in b});self.assertFalse({x['seed'] for x in a}&{x['seed'] for x in b})

    def test_training_route_balance(self):
        for c in CASES:self.assertEqual(sum(t['case']==c for t in specs('new_training')),12)

    def test_prospective_route_balance(self):
        for c in CASES:self.assertEqual(sum(t['case']==c for t in specs('prospective')),4)

    def test_radius_balance_and_known_bases(self):
        for c in CASES:
            ts=[x for x in specs('new_training') if x['case']==c]
            self.assertEqual([sum(x['norm']==v for x in ts) for v in (.375,.75)],[6,6])
            self.assertEqual({x['base_id'] for x in ts},{c+'__v%d'%i for i in range(3)})

    def test_specs_deterministic(self):self.assertEqual(specs('new_training'),specs('new_training'))

    def test_no_old_inspected_state_promoted(self):
        self.assertFalse(any('__c25_' in t['id'] for t in specs('new_training')+specs('prospective')))

    def test_thresholds_not_relaxed(self):
        self.assertEqual((SETTINGS['response_budget'],SETTINGS['kl_budget'],SETTINGS['margin_gate']),(.002,.005,.05))

    def test_c30_predictor_grid_unchanged(self):
        self.assertEqual(SETTINGS['predictor_methods'],predictor.SETTINGS)

    def test_preparation_not_postwrite_correction(self):
        self.assertEqual(SETTINGS['post_write_correction_steps'],0);self.assertTrue(SETTINGS['preparation_correction_only'])

    def test_no_native_or_LC1_claim(self):
        self.assertFalse(SETTINGS['LC1_closed']);self.assertFalse(SETTINGS['internal_parameter_update']);self.assertFalse(SETTINGS['fresh_evaluation'])

    def test_target_always_requires_output_flip(self):
        for s in STARTS.values():self.assertNotEqual(s[3],s[0]^s[1])

    def test_exact_inherited_source_and_recipe(self):check_sources(Path(__file__).resolve().parent)

    def test_source_zip_is_not_results(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('SOURCE_MANIFEST.json','{}')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z,self.assertRaisesRegex(ValueError,'结果ZIP'):verify(z)

    def test_manifest_coverage(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('MANIFEST.json','{}');z.writestr('extra','bad')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z,self.assertRaises(ValueError):verify(z)

    def test_crc_read_and_sha(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('MANIFEST.json',json.dumps({'x':'0'*64}));z.writestr('x','bad')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z,self.assertRaises(ValueError):verify(z)

    def test_unsafe_paths(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('../bad','x');z.writestr('MANIFEST.json','{}')
        with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z,self.assertRaises(ValueError):verify(z)

    def test_json_finite_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):read(p)


if __name__=='__main__':unittest.main(verbosity=2)
