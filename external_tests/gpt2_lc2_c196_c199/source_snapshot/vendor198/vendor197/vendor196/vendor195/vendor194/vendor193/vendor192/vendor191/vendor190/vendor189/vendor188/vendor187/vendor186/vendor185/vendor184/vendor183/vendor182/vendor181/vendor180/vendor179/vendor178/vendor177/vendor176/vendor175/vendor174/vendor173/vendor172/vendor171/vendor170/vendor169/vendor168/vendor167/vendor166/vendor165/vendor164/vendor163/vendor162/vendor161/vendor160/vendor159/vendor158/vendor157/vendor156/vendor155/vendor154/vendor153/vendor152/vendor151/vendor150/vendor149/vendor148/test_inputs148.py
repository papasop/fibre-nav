"""Cross-original bindings and tampering controls before any model import."""
import contextlib,copy,io,shutil,tempfile,unittest
from core148 import *
from fixture147 import make_fixture,rebound
from fixture133 import pack
from audit147 import audit
from inputs148 import load

class InputChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):paths,pc=make_fixture(cls.root/'parents');s=audit(paths,cls.root/'audit147',pc)
        s['errors']=[];dump(cls.root/'audit147'/'SUMMARY.json',s);shutil.copytree(ROOT/'vendor147',cls.root/'audit147'/'source',ignore=shutil.ignore_patterns('__pycache__'));p=pack(cls.root/'audit147',cls.root/'C147.zip');cls.paths=dict(paths,C147=p);cls.c=contract();cls.c.update(production=False,binding147=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()})
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_all_four_original_bindings_rerun_and_exact_initial_head(self):
        with contextlib.redirect_stdout(io.StringIO()):d=load(self.paths,self.root/'valid',self.c)
        self.assertEqual(len(d['points']),84);self.assertEqual(set(d['weights']),{sp['state_id'] for sp in registry()});self.assertEqual(d['router']['centers'].shape,(3,len(d['initial_a'])))
        with contextlib.closing(Archive(self.paths['C146'])) as z:exact(d['initial_a'],z.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],'Exact C146 initialization')
    def test_wrong_original_hash_blocks_before_audit(self):
        c=copy.deepcopy(self.c);c['pins']['C147']['sha256']='0'*64
        with self.assertRaises(ValueError):load(self.paths,self.root/'wronghash',c)
    def test_rebound_C147_summary_tamper_blocks(self):
        with contextlib.closing(Archive(self.paths['C147'])) as z:s=z.json('SUMMARY.json')
        s['training_states_audited']=83;p=rebound(self.paths['C147'],self.root/'C147_rebound.zip',{'SUMMARY.json':json.dumps(s).encode()});paths=dict(self.paths,C147=p);c=copy.deepcopy(self.c);c['pins']['C147']['sha256']=sha(p)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(ValueError):load(paths,self.root/'badsummary',c)

if __name__=='__main__':unittest.main(verbosity=2)
