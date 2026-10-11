import unittest, json, zipfile, tempfile, hashlib, ast
from pathlib import Path
from continuation_plan import initialise, validate_parent_summary, resume_score, V4_PROTOCOL, PREPARATION_TOTAL, WRITE_TOTAL
from b5_protocol import PROTOCOL, verified_zip
from protocol import endpoint_pass

def archive(path, data):
    data=dict(data)
    data['MANIFEST.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in data.items()}).encode()
    with zipfile.ZipFile(path,'w') as z:
        for n,b in data.items():z.writestr(n,b)

def summary():
    return {'protocol':V4_PROTOCOL,'complete':True,'n_completed':4,'n_passed':2,
            'cases':{k:{'pass':v} for k,v in {'xor_00':True,'xor_01':True,'xor_10':False,'xor_11':False}.items()}}

class ContinuationTests(unittest.TestCase):
    def test_incomplete_and_wrong_outcomes_rejected(self):
        s=summary();validate_parent_summary(s)
        s['complete']=False
        with self.assertRaises(ValueError):validate_parent_summary(s)
        s=summary();s['cases']['xor_10']['pass']=True
        with self.assertRaises(ValueError):validate_parent_summary(s)

    def test_thresholds_remain_and_failed_candidate_rejected(self):
        cfg={'response_budget':.002,'endpoint_kl_gate':.005}
        ev={'exact_accuracy':1.,'heldout_accuracy':1.,'minimum_exact_margin':.05,'minimum_heldout_margin':.05}
        self.assertTrue(endpoint_pass(ev,.00002,.0035,cfg))
        ev['minimum_exact_margin']=.04635
        self.assertFalse(endpoint_pass(ev,.00002,.0035,cfg))
        ev['minimum_exact_margin']=.05;ev['heldout_accuracy']=11/12
        self.assertFalse(endpoint_pass(ev,.00002,.0035,cfg))

    def test_initialise_preserves_parent_and_only_valid_states(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,protocol in [('b4','GPT2_B4_MINIMAL_CAUSAL_A100_V1'),('v3','GPT2_B5_SINGLE_BACKBONE_A100_V3')]:
                archive(root/(name+'.zip'),{'identity.json':json.dumps({'protocol':protocol}).encode()})
            data={'identity.json':json.dumps({'protocol':V4_PROTOCOL}).encode(),
                  'SUMMARY.json':json.dumps(summary()).encode(),
                  'parent_b4.zip':(root/'b4.zip').read_bytes(),'parent_v3.zip':(root/'v3.zip').read_bytes(),
                  'xor_10/preparation_optimizer.pt':b'actual checkpoint placeholder',
                  'xor_10/COMPLETE.json':b'{"pass":false}',
                  'xor_11/PREPARATION.json':json.dumps({'pass':True,'target_bits':[1,1,1,1]}).encode(),
                  'xor_11/prepared.pt':b'prepared tensor',
                  'xor_11/state.pt':b'rolled back state must not be copied',
                  'xor_11/COMPLETE.json':b'{"pass":false}'}
            for n in ['controller_training.pt','controller_identity.json','controller_progress.json',
                      'controller_training_data.json','controller_split.json','controller_readiness.json']:
                data[n]=b'unchanged'
            for c in ['xor_00','xor_01']:data[c+'/COMPLETE.json']=b'{"pass":true}'
            archive(root/'parent.zip',data);original=(root/'parent.zip').read_bytes()
            out=root/'out';initialise(root/'parent.zip',out)
            self.assertEqual((root/'parent.zip').read_bytes(),original)
            self.assertEqual((out/'parent_v4.zip').read_bytes(),original)
            self.assertEqual((out/'controller_training.pt').read_bytes(),b'unchanged')
            self.assertTrue((out/'xor_00/COMPLETE.json').exists())
            self.assertFalse((out/'xor_10/COMPLETE.json').exists())
            self.assertFalse((out/'xor_11/state.pt').exists())
            self.assertFalse((out/'xor_11/COMPLETE.json').exists())
            plan=json.loads((out/'CONTINUATION_PLAN.json').read_text())
            self.assertFalse(plan['thresholds_changed']);self.assertFalse(plan['controller_retrained'])

    def test_resume_progress_beats_filename(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,step in [('older.zip',605),('newer.zip',600)]:
                archive(root/n,{'identity.json':json.dumps({'protocol':PROTOCOL}).encode(),
                    'PHASE_PROGRESS.json':json.dumps({'xor_10/prepare':step}).encode()})
            self.assertGreater(resume_score(root/'older.zip'),resume_score(root/'newer.zip'))

    def test_corrupted_checkpoint_digest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.zip';archive(p,{'identity.json':json.dumps({'protocol':PROTOCOL}).encode(),'tensor.pt':b'good'})
            with zipfile.ZipFile(p) as z:values={n:z.read(n) for n in z.namelist()}
            values['tensor.pt']=b'bad'
            with zipfile.ZipFile(p,'w') as z:
                for n,b in values.items():z.writestr(n,b)
            with self.assertRaises(ValueError):verified_zip(p,PROTOCOL)

    def test_worker_retains_rejected_candidate_and_optimizer(self):
        s=(Path(__file__).parent/'worker_b5.py').read_text();ast.parse(s)
        self.assertIn("'_candidate.pt'",s)
        self.assertIn("'_optimizer.pt'",s)
        self.assertNotIn("pending_write.pt').unlink",s)
        self.assertIn('controller_hash()!=frozen_controller_hash',s)
        self.assertEqual(PREPARATION_TOTAL,1200);self.assertEqual(WRITE_TOTAL,1080)

if __name__=='__main__':unittest.main(verbosity=2)
