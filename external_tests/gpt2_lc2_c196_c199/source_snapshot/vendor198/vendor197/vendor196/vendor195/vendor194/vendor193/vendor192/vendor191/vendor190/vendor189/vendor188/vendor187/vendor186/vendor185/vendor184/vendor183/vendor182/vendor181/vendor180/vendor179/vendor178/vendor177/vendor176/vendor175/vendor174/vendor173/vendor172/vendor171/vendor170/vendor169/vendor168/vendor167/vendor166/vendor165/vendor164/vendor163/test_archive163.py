"""Eight-original synthetic binding; optional cached seven-parent fixtures."""
import copy,io,os,shutil,subprocess,tempfile,unittest
from core163 import *
from inputs163 import load,audit162
from fixture162 import pack,changed,json_change
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cached=os.environ.get('C163_TEST_PARENTS');cls.legacy=None
  if cached:
   cached=Path(cached);paths=clean_json((cached/'paths.json').read_bytes());pc=clean_json((cached/'contract.json').read_bytes());audit_result=cached/'audit_result'
  else:
   import test_archive161 as old
   from inputs161 import load as load161
   from fixture162 import produce
   cls.legacy=old.Checks;cls.legacy.setUpClass();pp=cls.legacy;paths=dict(pp.paths)
   with contextlib.redirect_stdout(io.StringIO()):ctx=load161(paths,cls.root/'parent_context',pp.c)
   paths['C161']=produce(ctx,paths,pp.c,cls.root/'synthetic161');pc=previous.contract();pc.update(production=False,parent=pp.c,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});stage=cls.root/'seven';stage.mkdir();dump(stage/'paths.json',{k:str(p) for k,p in paths.items()});dump(stage/'contract.json',pc)
   code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import worker162;root=Path(sys.argv[2]);s=worker162.run(json.loads((root/'paths.json').read_text()),root/'audit_result',json.loads((root/'contract.json').read_text()));assert s['complete'],s['errors'];assert 'torch' not in sys.modules"
   r=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'vendor162'),str(stage)],capture_output=True,text=True,timeout=600,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'));need(r.returncode==0,r.stdout+r.stderr);audit_result=stage/'audit_result'
  paths={k:Path(p) for k,p in paths.items()};out=cls.root/'synthetic162';shutil.copytree(audit_result,out);shutil.copytree(ROOT/'vendor162',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p=cls.root/'C162.zip';pack(out,p);paths['C162']=p;cls.paths=paths;cls.pc=pc;cls.c=contract();cls.c.update(production=False,parent=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
  with contextlib.redirect_stdout(io.StringIO()):cls.ctx=load(paths,cls.root/'eight_bound',cls.c)
 @classmethod
 def tearDownClass(cls):
  cls.tmp.cleanup()
  if cls.legacy is not None:cls.legacy.tearDownClass()
 def test_all_eight_originals_bound_and_exact_step8_head(self):
  self.assertTrue(self.ctx['C162_receipt']['all672_transitions_and336_precision_rows_recomputed']);np.testing.assert_array_equal(self.ctx['initial_head'],self.ctx['C161_audit']['heads'][8]);self.assertEqual(len(self.ctx['parent_final_FP64_records']),84);self.assertEqual(len(self.ctx['parent_final_native_records']['0']),84)
 def test_rebound_C162_summary_forgery_blocks(self):
  p=changed(self.paths['C162'],self.root/'forged162.zip','SUMMARY.json',json_change(lambda a:a['diagnostics'].__setitem__('maximizer_set_switches',999)),True);d=clean_json((self.root/'eight_bound/C162_recomputed_diagnostics/DIAGNOSTIC_SUMMARY.json').read_bytes())
  with self.assertRaises(ValueError):audit162(p,self.ctx['C161_audit'],d,self.root/'eight_bound/C162_recomputed_diagnostics',self.c)
 def test_rebound_C162_source_forgery_blocks(self):
  p=changed(self.paths['C162'],self.root/'forgedsource162.zip','source/core162.py',lambda b:b+b'\n# forged\n',True);d=clean_json((self.root/'eight_bound/C162_recomputed_diagnostics/DIAGNOSTIC_SUMMARY.json').read_bytes())
  with self.assertRaises(ValueError):audit162(p,self.ctx['C161_audit'],d,self.root/'eight_bound/C162_recomputed_diagnostics',self.c)
if __name__=='__main__':unittest.main(verbosity=2)
