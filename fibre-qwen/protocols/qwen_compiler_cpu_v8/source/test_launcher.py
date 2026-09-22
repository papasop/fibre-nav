"""Companion launcher checks; no model weights or scientific evaluations."""
import subprocess,hashlib,importlib.util,json,os,sys,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
path=Path(os.environ.get('QWEN_COMPILER_V8_LAUNCHER',str(Path(__file__).resolve().parents[1]/'COLAB_LAUNCHER_QWEN_COMPILER_CPU_V8.py')))
spec=importlib.util.spec_from_file_location('qwen_zero_v3_launcher',path);L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
class LauncherTests(unittest.TestCase):
    def source(self,root):
        data={'protocol.json':json.dumps({'protocol_id':L.PROTOCOL_ID}).encode(),'run_cpu.py':b'pass\n'}
        manifest={n:L.sha(d) for n,d in data.items()};p=root/'renamed_bundle.zip'
        with zipfile.ZipFile(p,'w') as z:
            for n,d in data.items():z.writestr('nested/source/'+n,d)
            z.writestr('nested/source/MANIFEST.json',json.dumps(manifest))
        return p,manifest
    def test_renamed_nested_source_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,m=self.source(root)
            with patch.object(L,'SOURCE_MANIFEST_SHA256',L.sha(L.canonical(m))):desc=L.inspect_archive(p)
            self.assertEqual(desc[0],'source');out=root/'out';L.materialize(p,desc,out)
            self.assertEqual((out/'run_cpu.py').read_bytes(),b'pass\n')
            (out/'run_cpu.py').write_text('modified')
            with self.assertRaises(RuntimeError):L.materialize(p,desc,out)
    def test_source_hash_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p,m=self.source(Path(d))
            with patch.object(L,'SOURCE_MANIFEST_SHA256','wrong'):
                with self.assertRaises(ValueError):L.inspect_archive(p)
            for n in ['../bad','/absolute','C:/bad','x\\bad']:
                with self.assertRaises(ValueError):L.safe_name(n)
    def test_resume_checksum_and_old_protocol_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.zip'
            def make(protocol,corrupt=False):
                data=json.dumps(dict(protocol_id=protocol,source_manifest_sha256=L.SOURCE_MANIFEST_SHA256)).encode()
                with zipfile.ZipFile(p,'w') as z:
                    z.writestr('run/RUN_LOCK.json',data)
                    z.writestr('run/RESULT_MANIFEST.json',json.dumps({'RUN_LOCK.json':'bad' if corrupt else L.sha(data)}))
            make(L.PROTOCOL_ID);self.assertEqual(L.inspect_archive(p)[0],'resume')
            make('ADDRESS_CALIBRATION_CPU_V13_FROZEN_FULL_READOUT_TEST')
            with self.assertRaises(ValueError):L.inspect_archive(p)
            make(L.PROTOCOL_ID,True)
            with self.assertRaises(ValueError):L.inspect_archive(p)
    def test_bootstrap_exact_argv_in_real_subprocess(self):
        # Original bootstrap left script at argv[1], causing argparse to abort.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/'parser.py'
            script.write_text("import argparse,json,sys; p=argparse.ArgumentParser(); p.add_argument('--worker'); p.add_argument('--out'); a=p.parse_args(); print(json.dumps([sys.argv,a.worker,a.out]))")
            args=['--worker','case_1','--out',str(root/'out with spaces')]
            cmd=[sys.executable,'-I','-S','-c',L.bootstrap_code(),str(root),str(script)]+args
            result=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True).stdout)
            self.assertEqual(result,[[str(script)]+args,'case_1',str(root/'out with spaces')])
if __name__=='__main__':unittest.main(verbosity=2)
