import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Budget,MFIController,near_kernel

class Chart:
    def __init__(self, fail=False): self.v=torch.zeros(2);self.fail=fail
    def vector(self):return self.v.clone()
    def set_vector(self,v):self.v=v.clone()
    def jacobian(self):return torch.tensor([[1.,0.]])
    def gradient(self,value):return torch.tensor([0.,-1. if value else 1.])
    def read(self):return {'value':int(self.v[1]>0)}
    def audit(self,value):
        return {'response':abs(float(self.v[0])), 'kl':float(self.v[1]**2) if self.fail else 0.,
                'margin':float(self.v[1])*(1 if value else -1)}

class Tests(unittest.TestCase):
    def test_project_and_write_overwrite(self):
        self.assertLess(abs(float(near_kernel(torch.ones(2),torch.tensor([[1.,0.]]),1e-8)[0])),1e-6)
        b=Chart();c=MFIController(b,Budget(margin=.5,step_norm=.5,steps=4))
        self.assertTrue(c.execute('WRITE',1)['accepted'])
        self.assertTrue(c.execute('OVERWRITE',0)['accepted'])
        self.assertEqual(c.read()['value'],0)

    def test_budget_failure_rolls_back_whole_operation(self):
        b=Chart(fail=True);c=MFIController(b,Budget(margin=1.,kl=.01,steps=4))
        r=c.execute('WRITE',1)
        self.assertFalse(r['accepted']);self.assertTrue(torch.equal(b.vector(),torch.zeros(2)))
        self.assertFalse(c.written)
        with self.assertRaises(ValueError):c.execute('OVERWRITE',0)

    def test_exception_rolls_back(self):
        b=Chart();c=MFIController(b,Budget(margin=2.,step_norm=.5))
        old=b.gradient;calls=[0]
        def broken(value):
            calls[0]+=1
            if calls[0]>1:raise RuntimeError('test')
            return old(value)
        b.gradient=broken
        with self.assertRaises(RuntimeError):c.execute('WRITE',1)
        self.assertTrue(torch.equal(b.vector(),torch.zeros(2)))

    def test_invalid_op_and_nonfinite(self):
        c=MFIController(Chart())
        with self.assertRaises(ValueError):c.execute('SWAP',1)
        with self.assertRaises(ValueError):c.execute('WRITE',True)
        self.assertFalse(c.passes({'response':0.,'kl':float('nan'),'margin':9.}))

if __name__=='__main__':unittest.main()
