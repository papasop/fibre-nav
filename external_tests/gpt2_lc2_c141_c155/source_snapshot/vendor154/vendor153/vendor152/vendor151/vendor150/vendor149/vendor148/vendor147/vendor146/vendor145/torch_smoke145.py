"""Mandatory actual CPU Torch tests before the GPT2 diagnostic."""
import unittest
import torch
# Bootstrap the pinned vendored import chain before importing inherited smoke tests.
from core145 import *
import torch_smoke143 as previous
from functional143 import Endpoint,torch_proposal
from functional145 import scalar_gradients

class ExistingGraphChecks(previous.Checks):
    pass

class NewChecks(unittest.TestCase):
    def constants(self):
        theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[:3]=[.001,.002,.004];V=np.zeros((3,DIM));V[:3,:3]=np.eye(3);return theta,raw,V
    def test_two_actual_scalar_backwards_at_one_fixed_control(self):
        rt=previous.runtime();ep=Endpoint(rt);theta,raw,V=self.constants();a=torch.tensor([.01,-.02,.03],dtype=torch.float64,requires_grad=True);point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),a);events=[]
        r=scalar_gradients(ep,point,torch.from_numpy(theta),a,'xor_00',lambda:events.append('attempt'),lambda:events.append('complete'))
        self.assertEqual(events,['attempt','complete','attempt','complete']);expected=np.zeros(3);expected[2]=float(raw[2])*(1+.2*float(point[2].detach()))/.002;np.testing.assert_allclose(r['response_gradient'],expected,atol=1e-9,rtol=2e-6);self.assertTrue(np.isfinite(r['original_gradient']).all());ep.verify(rehash=True)
    def test_response_softmax_gradient_matches_full_torch_graph(self):
        a=torch.tensor([.02,-.01],dtype=torch.float64,requires_grad=True);M=torch.arange(168,dtype=torch.float64).reshape(84,2)/167;q=.3+(M@a+torch.linspace(.01,.3,84,dtype=torch.float64)).square();gs=torch.stack([torch.autograd.grad(x,a,retain_graph=True)[0] for x in q]);full=torch.logsumexp(32*q,0)/32;expected=torch.autograd.grad(full,a)[0].numpy();w=torch.softmax(32*q.detach(),0).numpy();np.testing.assert_allclose(w@gs.numpy(),expected,atol=1e-13,rtol=1e-12)
    def test_abs_amax_ties_use_valid_averaged_subgradient(self):
        x=torch.tensor([.003,-.003,.001],dtype=torch.float64,requires_grad=True);g=torch.autograd.grad(x.abs().amax()/.002,x)[0];torch.testing.assert_close(g,torch.tensor([250.,-250.,0.],dtype=torch.float64))
    def test_original_scalar_units_match_independent_NumPy(self):
        rt=previous.runtime();ep=Endpoint(rt);theta,raw,V=self.constants();aa=np.array([.01,-.02,.03]);a=torch.tensor(aa,dtype=torch.float64,requires_grad=True);point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),a);r=scalar_gradients(ep,point,torch.from_numpy(theta),a,'xor_00');terms=auditcore.loss(r['observed'],r['KL'],point.detach().numpy(),theta,aa,'xor_00');compare(r['original_objective_parts'],terms,atol=1e-10,rtol=1e-9);self.assertAlmostEqual(r['response_ratio'],qvalue(r['observed']))

if __name__=='__main__':unittest.main(verbosity=2)
