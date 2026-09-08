import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch
from safetensors.torch import save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from checkpoint import restore
from core import Budget

class Chart:
    def __init__(self):self.modules={'cell':SimpleNamespace(A=torch.ones(1,2),B=torch.zeros(1,1))}
    def vector(self):return self.modules['cell'].B.flatten().clone()
    def set_vector(self,v):self.modules['cell'].B.copy_(v.reshape(1,1))
    def read(self):return {'value':int(self.vector()[0]>0)}
    def audit(self,value):return {'response':0.,'kl':0.,'margin':float(self.vector()[0])*(1 if value else -1)}

class Tests(unittest.TestCase):
    def test_valid_load_wrong_content_rollback_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'chart.safetensors'
            save_file({'cell.A':torch.ones(1,2),'cell.B':torch.ones(1,1)*2},str(p),
                      metadata={'model':'test','revision':'fixed','scale':'2.0'})
            sha=hashlib.sha256(p.read_bytes()).hexdigest();c=Chart()
            kwargs=dict(expected_sha256=sha,model='test',revision='fixed',budget=Budget())
            ctl,r=restore(c,p,value=1,**kwargs)
            self.assertTrue(ctl.written);self.assertTrue(r['all_B_exact'])
            before=c.vector()
            with self.assertRaises(ValueError):restore(c,p,value=0,**kwargs)
            self.assertTrue(torch.equal(before,c.vector()))
            kwargs['expected_sha256']='wrong'
            with self.assertRaises(ValueError):restore(c,p,value=1,**kwargs)
            self.assertTrue(torch.equal(before,c.vector()))

    def test_fixed_chart_and_nan_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'chart.safetensors';c=Chart()
            for a,b in [(torch.zeros(1,2),torch.ones(1,1)),(torch.ones(1,2),torch.full((1,1),float('nan')))]:
                save_file({'cell.A':a,'cell.B':b},str(p),metadata={'model':'test','revision':'fixed','scale':'2.0'})
                with self.assertRaises(ValueError):
                    restore(c,p,expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model='test',revision='fixed',value=1,budget=Budget())
                self.assertEqual(float(c.vector()[0]),0.)

if __name__=='__main__':unittest.main()
