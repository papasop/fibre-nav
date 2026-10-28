"""Slow full-size synthetic C155/C156 archive binding integration; no real measurements or model."""
import contextlib,copy,io,shutil,tempfile,unittest,zipfile
from core157 import *
from inputs157 import load
from fixture156 import fixture,pack
from worker156 import run as run156
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name);a,pc=fixture();out=root/'c156'
        with contextlib.redirect_stdout(io.StringIO()):s=run156(a,out,pc)
        need(s['complete'],'Full synthetic C156 must complete');shutil.copytree(ROOT/'vendor156',out/'source',ignore=shutil.ignore_patterns('__pycache__'));b=root/'C156.zip';pack(out,b);cls.paths=dict(C155=a,C156=b);cls.c=contract();cls.c.update(production=False,parent=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in cls.paths.items()});cls.root=root
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_full_C155_C156_binding_and_summary_recomputation(self):
        data=load(self.paths,self.root/'rebound',self.c);self.assertEqual(len(data['data']['records']),1530);self.assertEqual(len(data['data']['points']),588);self.assertEqual(set(data['replay_absolute']),{(sid,n) for sid in FOCUS for n in ('zero','plus_one')})
    def test_wrong_C156_whole_hash_blocks_before_payload(self):
        c=copy.deepcopy(self.c);c['pins']['C156']['sha256']='0'*64
        with self.assertRaises(ValueError):load(self.paths,self.root/'wrong',c)
if __name__=='__main__':unittest.main(verbosity=2)
