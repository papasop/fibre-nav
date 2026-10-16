"""Synthetic full pipeline and failure injection; does not run actual GPT-2."""
import sys,tempfile,unittest,contextlib,io,zipfile,shutil,time
from unittest.mock import patch
from pathlib import Path
from c36_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
import numpy as np
import torch
from c36_protocol import *
from c36_parent import prepare,validate,probe,resume_score
from c36_audit import audit,finite_tensors,compare_replay,writer_identity
from worker_c36 import main,freeze,summary_value
from c35_protocol import PROTOCOL as C35_PROTOCOL,SETTINGS as C35_SETTINGS,sources as c35_sources
from worker_c35 import main as parent_main,proposals
from test_c35_torch import setup,MockRuntime
from c35_fit import records
from c35_writer import ComposedWriter
from c31_runtime import load,save_tensor,tensor_hash

def fake_identity(data):return {'synthetic_core_identity':True}

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.parent_root=Path(cls.temp.name)
        cls.started=time.monotonic()
        print('C36 SELFTEST SETUP 1/3: build synthetic C35 states; no GPT-2 loading',flush=True)
        # Archive corruption and backup retention have dedicated real ZIP tests.
        # Suppress repeated whole-result compression ONLY in this synthetic pipeline.
        # All freeze, tensor, gate, resume and proposal assertions still execute.
        with contextlib.redirect_stdout(io.StringIO()),patch('worker_c35.snapshot'),patch('c35_fit.snapshot'):
            _,cls.c35=setup(cls.parent_root)
            parent_main(cls.c35,runtime_factory=MockRuntime)
        print('C36 SELFTEST SETUP 2/3: create and verify one synthetic archive',flush=True)
        cls.parent_zip=cls.parent_root/'parent_c35.zip'
        m={p.relative_to(cls.c35).as_posix():sha(p) for p in cls.c35.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts}
        with zipfile.ZipFile(cls.parent_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for n in m:z.write(cls.c35/n,n)
            for n,h in read(ROOT/'C35_SOURCE_PINS.json').items():z.write(ROOT/'inherited'/n,'source/'+n);m['source/'+n]=h
            z.writestr('MANIFEST.json',json.dumps(m))
        cls.freeze_sha=sha(cls.c35/'COEFFICIENTS_FROZEN.json')
        cls.template=cls.parent_root/'template';cls.template.mkdir()
        with patch.dict(SETTINGS,{'expected_parent_freeze_sha256':cls.freeze_sha}),contextlib.redirect_stdout(io.StringIO()):
            dump(cls.template/'identity.json',identity(ROOT));prepare(cls.parent_zip,cls.template,ROOT)
            dump(cls.template/'NEW_STATE_SPECS.json',specs())
        print('C36 SELFTEST SETUP 3/3: isolated per-test copies ready; elapsed %.1fs'%(time.monotonic()-cls.started),flush=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):
        print('C36 SELFTEST CASE',self._testMethodName,flush=True)
        self.temp=tempfile.TemporaryDirectory();self.r=Path(self.temp.name);self.out=self.r/'session/run'
        self.setting=patch.dict(SETTINGS,{'expected_parent_freeze_sha256':self.freeze_sha});self.setting.start()
        # Real independent copies, never hard links: injected corruption cannot leak.
        shutil.copytree(self.template,self.out)
    def tearDown(self):self.setting.stop();self.temp.cleanup()
    def run_main(self,rt=MockRuntime):
        with contextlib.redirect_stdout(io.StringIO()),patch('worker_c36.snapshot'):
            return main(self.out,runtime_factory=rt,identity_checker=fake_identity)
    def test_full_pipeline_32_and_completed_resume(self):
        parent_sha=sha(self.parent_zip);s=self.run_main()
        self.assertEqual(s['decision'],'C36_BOTH_FROZEN_WRITERS_LOCAL_REPLICATION_PASS');self.assertEqual(s['new_local']['selected6']['n_requested'],32)
        self.assertEqual(s['new_local']['selected6']['complete_passed'],32);self.assertEqual(sha(self.parent_zip),parent_sha)
        self.assertEqual(s['parent_proposals_exactly_reproduced'],80);self.assertEqual(s['parent_physical_replay_states'],8)
        self.assertFalse(s['LC1_independently_confirmed']);self.assertFalse(s['full_Line_C_complete'])
        def forbidden(*a):raise AssertionError('Completed resume reloaded model')
        self.assertEqual(self.run_main(forbidden),s)
    def test_interrupted_endpoint_resumes_same_frozen_proposals(self):
        class Interrupt(MockRuntime):
            def evaluate(self,*a,**kw):
                if self.evaluation_calls==26:raise RuntimeError('mid endpoint interruption')
                return super().evaluate(*a,**kw)
        with self.assertRaisesRegex(RuntimeError,'mid endpoint'):self.run_main(Interrupt)
        frozen=sha(self.out/'C36_FROZEN.json');point=self.out/'c35/proposals'/specs()[0]['id']/'POINTS.pt';h=sha(point)
        s=self.run_main();self.assertTrue(s['complete']);self.assertEqual(sha(point),h);self.assertEqual(sha(self.out/'C36_FROZEN.json'),frozen)
    def test_parent_tensor_corruption_rejected(self):
        p=self.out/'c35/fit/xor_00/HEAD.pt';p.write_bytes(p.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):validate(self.out,ROOT)
    def test_parent_numeric_false_gate_rejected(self):
        p=self.out/'c35/results/xor_00__c35_new00/COMPLETE.json';r=read(p);r['arms']['selected6']['metrics']['gates']['response']=False;dump(p,r)
        with self.assertRaises(ValueError):audit(self.out/'c35',ROOT,fake_identity)
    def test_actual_core_identity_checker_rejects_synthetic(self):
        with self.assertRaises((KeyError,ValueError)):writer_identity(self.out/'c35')
    def test_proposal_point_tamper_caught_even_with_updated_local_record(self):
        p=self.out/'c35/proposals/xor_00__c35_new00/POINTS.pt';r=load(p);r['points']['selected6'][0]+=.01;save_tensor(p,r)
        with self.assertRaises(ValueError):audit(self.out/'c35',ROOT,fake_identity)
    def test_no_freeze_after_new_generation(self):
        dump(self.out/'PARENT_AUDIT.json',{'test':True});(self.out/'c35/new_states'/specs()[0]['id']).mkdir()
        with self.assertRaisesRegex(ValueError,'Cannot freeze'):freeze(self.out)
    def test_nonfinite_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):finite_tensors({'nested':[torch.tensor(float('nan'))]})
    def test_replay_gate_failure_never_hidden_by_tolerance(self):
        d=read(self.out/'c35/results/xor_00__c35_new00/COMPLETE.json')['arms']['selected6']['metrics'];bad={**d,'bits':[1,0,1,0]}
        with self.assertRaises(ValueError):compare_replay(bad,d,'xor_00')
    def test_partial_branch_results_not_hidden(self):
        s=self.run_main();rows=[read(self.out/'c35/results'/sp['id']/'COMPLETE.json') for sp in specs()]
        rows[0]['arms']['selected6']['complete_accepted']=False
        changed=summary_value(self.out,rows);self.assertEqual(changed['decision'],'C36_ONE_BRANCH_LOCAL_REPLICATION_PASS')
        self.assertTrue(changed['branch_complete_acceptance']['geometry6']);self.assertFalse(changed['branch_complete_acceptance']['selected6'])
    def test_invalid_state_retained_and_branch_fails(self):
        self.run_main();rows=[read(self.out/'c35/results'/sp['id']/'COMPLETE.json') for sp in specs()]
        rows[0]['preparation_valid']=False
        for a in ARMS:rows[0]['arms'][a]['complete_accepted']=False
        changed=summary_value(self.out,rows);self.assertEqual(changed['invalid_preparations'],1);self.assertEqual(changed['new_local']['selected6']['n_requested'],32);self.assertEqual(changed['new_local']['selected6']['complete_passed'],31)
        self.assertEqual(changed['decision'],'C36_LOCAL_REPLICATION_FAILURE')
if __name__=='__main__':unittest.main(verbosity=2)
