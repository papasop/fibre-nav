from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_seeds import controls
from core import Budget

class Chart:
    def __init__(self):self.v=torch.tensor([0.,2.])
    def vector(self):return self.v.clone()
    def set_vector(self,v):self.v=v.clone()
    def jacobian(self):return torch.tensor([[1.,0.]])
    def audit(self,value):return {'response':abs(float(self.v[0])),'kl':0.,'margin':float(self.v[1])}

class Tests(unittest.TestCase):
    def test_norm_match_and_state_restoration(self):
        b=Chart();saved=b.vector()
        r=controls(b,torch.zeros(2),torch.tensor([0.,1.]),1,90001,Budget())
        self.assertEqual(len(r['random']),3)
        self.assertTrue(all(x['norm_error']<1e-6 for x in r['random']))
        self.assertTrue(torch.equal(b.vector(),saved))
        self.assertEqual(r['no_move']['audit']['margin'],0.)
    def test_restore_on_control_error(self):
        b=Chart();saved=b.vector()
        def broken():raise RuntimeError('test')
        b.jacobian=broken
        with self.assertRaises(RuntimeError):controls(b,torch.zeros(2),torch.ones(2),1,90001,Budget())
        self.assertTrue(torch.equal(b.vector(),saved))

if __name__=='__main__':unittest.main()
