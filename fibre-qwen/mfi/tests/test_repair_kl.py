import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import Budget,near_kernel
from repair_kl import KLAwareController
from test_core import Chart

class KLChart(Chart):
    def kl_gradient(self):return torch.tensor([0.,1.])

class Tests(unittest.TestCase):
    def test_inward_direction_when_kl_active(self):
        b=KLChart();c=KLAwareController(b)
        d=c.direction(1,b.jacobian(),{'kl':.009})
        self.assertLess(float(d@b.kl_gradient()),0)
        self.assertLess(abs(float(b.jacobian()@d)),1e-6)
    def test_original_direction_below_activation(self):
        b=KLChart();c=KLAwareController(b)
        self.assertTrue(torch.equal(c.direction(1,b.jacobian(),{'kl':0.}),near_kernel(-b.gradient(1),b.jacobian(),c.budget.ridge)))
    def test_active_gradient_exception_rolls_back(self):
        b=KLChart();b.v[1]=.2
        b.audit=lambda v:{'response':0.,'kl':.009,'margin':0.}
        def fail():raise RuntimeError('gradient failed')
        b.kl_gradient=fail;c=KLAwareController(b);before=b.vector()
        with self.assertRaises(RuntimeError):c.execute('WRITE',1)
        self.assertTrue(torch.equal(before,b.vector()));self.assertFalse(c.written)
    def test_original_command_checks_retained(self):
        c=KLAwareController(KLChart())
        with self.assertRaises(ValueError):c.execute('OVERWRITE',1)
        with self.assertRaises(ValueError):c.execute('WRITE',True)

if __name__=='__main__':unittest.main()
