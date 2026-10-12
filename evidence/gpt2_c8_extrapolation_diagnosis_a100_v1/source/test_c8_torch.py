"""Required in Colab; does not download GPT-2. Curved toy chart audits autograd."""
import unittest
import numpy as np
import torch
from diagnostic_math import committed_delta,linear_vs_finite

class Tests(unittest.TestCase):
    def test_curved_chart_jacobian_and_finite_change(self):
        theta=torch.tensor([1.,2.],requires_grad=True,dtype=torch.float32)
        R=theta[0]**2+theta[1];J=torch.autograd.grad(R,theta)[0]
        self.assertTrue(torch.equal(J,torch.tensor([2.,1.])))
        d=np.array([.1,-.2]);r=linear_vs_finite(J.numpy().reshape(1,-1),d,[3],[1.1**2+1.8])
        self.assertLess(r['linear_norm2'],1e-12);self.assertAlmostEqual(r['measured_remainder'][0],.01)
    def test_frozen_base_and_selected_chart_derivatives(self):
        base=torch.nn.Parameter(torch.tensor([3.]),requires_grad=False)
        chart=torch.nn.Parameter(torch.tensor([2.]),requires_grad=True);version=base._version
        g=torch.autograd.grad(base*chart,chart)[0];self.assertEqual(g.item(),3)
        self.assertIsNone(base.grad);self.assertEqual(base._version,version)
    def test_graph_rows_no_accumulated_grad(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));outputs=torch.stack([p[0]*p[1],p.square().sum()])
        rows=[torch.autograd.grad(outputs[i],p,retain_graph=i==0)[0] for i in range(2)]
        self.assertTrue(torch.equal(torch.stack(rows),torch.tensor([[2.,1.],[2.,4.]])));self.assertIsNone(p.grad)
    def test_float32_physical_commit_delta(self):
        p=torch.nn.Parameter(torch.tensor([1e8]),requires_grad=False);prepared=p.detach().clone()
        with torch.no_grad():p.copy_(prepared+1.)
        self.assertEqual(committed_delta(prepared.numpy(),p.numpy())[0],0)
    def test_torch_weights_only_checkpoint_roundtrip(self):
        import io
        b=io.BytesIO();torch.save({'J':torch.ones(2,3),'DH':torch.zeros(4,3)},b);b.seek(0)
        r=torch.load(b,weights_only=True);self.assertEqual(r['J'].shape,(2,3))

if __name__=='__main__':unittest.main()
