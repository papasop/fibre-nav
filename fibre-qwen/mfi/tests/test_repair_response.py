import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from repair_response import ResponseKLController
from test_core import Chart

class ResponseChart(Chart):
    source_response=torch.zeros(1,dtype=torch.float64)
    def response(self):return self.v[:1]
    def kl_gradient(self):return torch.tensor([0.,1.])

class Tests(unittest.TestCase):
    def test_response_moves_inward_and_target_keeps_component(self):
        b=ResponseChart();b.v[0]=.018;c=ResponseKLController(b)
        d=c.direction(1,b.jacobian(),{'response':.018,'kl':0})
        self.assertLess(float(d[0]),0);self.assertGreater(float(d[1]),0)
    def test_both_active_normals_point_inward(self):
        b=ResponseChart();b.v[0]=.018;c=ResponseKLController(b)
        d=c.direction(1,b.jacobian(),{'response':.018,'kl':.009})
        self.assertLess(float(d[0]),0);self.assertLess(float(d[1]),0)
    def test_response_gradient_failure_restores_whole_operation(self):
        b=ResponseChart();b.v[0]=.018;before=b.vector()
        def fail():raise RuntimeError('response failed')
        b.response=fail;c=ResponseKLController(b)
        with self.assertRaises(RuntimeError):c.execute('WRITE',1)
        self.assertTrue(torch.equal(before,b.vector()));self.assertFalse(c.written)

if __name__=='__main__':unittest.main()
