"""Mandatory actual CPU Torch tests, before GPT2 loading; initialize vendored paths first."""
import unittest
import torch
from core146 import *
import torch_smoke145 as previous
from functional146 import Endpoint,torch_proposal,loss_parts,scalar_gradients

class OriginalGraphChecks(previous.ExistingGraphChecks):pass
class PriorDirectionalChecks(previous.NewChecks):pass

class WorstResponseChecks(unittest.TestCase):
    def constants(self):return previous.NewChecks().constants()
    def test_objective_parts_match_independent_numpy(self):
        theta,raw,V=self.constants();a=np.array([.01,-.02,.03]);point=proposal(theta,raw,V,a)[0];o=np.r_[[-.003,.004,.001,0,.002,-.001],np.linspace(-.15,.15,16)];k=.006
        actual=loss_parts(torch.from_numpy(o),torch.tensor(k,dtype=torch.float64),torch.from_numpy(point),torch.from_numpy(theta),torch.from_numpy(a),'xor_00');compare({k:float(v) for k,v in actual.items()},objective(o,k,point,theta,a,'xor_00'),atol=1e-10,rtol=1e-9)
    def test_actual_two_scalars_and_regularizer_gradient(self):
        rt=previous.previous.runtime();ep=Endpoint(rt);theta,raw,V=self.constants();a=torch.tensor([.01,-.02,.03],dtype=torch.float64,requires_grad=True);point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),a);events=[]
        r=scalar_gradients(ep,point,torch.from_numpy(theta),a,'xor_00',lambda:events.append('attempt'),lambda:events.append('done'));self.assertEqual(events,['attempt','done','attempt','done']);q=max(abs(np.asarray(r['observed'])[:6]))/RESPONSE_SAFETY;expected=np.zeros(3);expected[2]=2*(q-1)/RESPONSE_SAFETY*float(raw[2])*(1+.2*float(point[2].detach()));np.testing.assert_allclose(r['response_gradient'],expected,rtol=3e-6,atol=1e-9);np.testing.assert_allclose(r['guard_gradient'],2*L2*a.detach().numpy()/3,atol=1e-12,rtol=1e-9);ep.verify(rehash=True)
    def test_global_exact_max_gradient_matches_torch_graph(self):
        a=torch.tensor([.02,-.01],dtype=torch.float64,requires_grad=True);M=torch.arange(168,dtype=torch.float64).reshape(84,2)/167;P=M.flip(0);R=(M@a+torch.linspace(.01,.3,84,dtype=torch.float64)).square();G=(P@a+.02).square();gr=torch.stack([torch.autograd.grad(x,a,retain_graph=True)[0] for x in R]);gg=torch.stack([torch.autograd.grad(x,a,retain_graph=True)[0] for x in G]);full=.9*R.amax()+.1*R.mean()+G.mean();expected=torch.autograd.grad(full,a)[0].numpy()
        rows=[dict(state_id=sp['state_id'],objective_parts=dict(response_loss=float(R[i].detach()),read_loss=float(G[i].detach())),response_gradient=gr[i].numpy().tolist(),guard_gradient=gg[i].numpy().tolist()) for i,sp in enumerate(registry())];g,agg=aggregate(rows,2);np.testing.assert_allclose(g,expected,rtol=1e-12,atol=1e-13);self.assertAlmostEqual(agg['objective'],float(full.detach()),14)
    def test_exact_state_ties_average_subgradient(self):
        a=torch.tensor([0.,0.],dtype=torch.float64,requires_grad=True);R=torch.cat([1+a,torch.zeros(82,dtype=torch.float64)]);full=.9*R.amax()+.1*R.mean();expected=torch.autograd.grad(full,a)[0].numpy();rows=[]
        for i,sp in enumerate(registry()):rows.append(dict(state_id=sp['state_id'],objective_parts=dict(response_loss=float(R[i].detach()),read_loss=0.),response_gradient=([1.,0.] if i==0 else [0.,1.] if i==1 else [0.,0.]),guard_gradient=[0.,0.]))
        g,agg=aggregate(rows,2);np.testing.assert_allclose(g,expected,rtol=1e-12,atol=1e-13);self.assertEqual(agg['exact_max_tie_count'],2)
    def test_coordinate_tie_sign_and_safety_units(self):
        o=torch.tensor([.003,-.003,.001,0.,0.,0.]+[.1]*16,dtype=torch.float64,requires_grad=True);a=torch.zeros(3,dtype=torch.float64);p=torch.zeros(DIM,dtype=torch.float32);loss=loss_parts(o,torch.tensor(0.),p,p,a,'xor_00')['response_loss'];g=torch.autograd.grad(loss,o)[0];v=(.003/RESPONSE_SAFETY-1)/RESPONSE_SAFETY;torch.testing.assert_close(g[:3],torch.tensor([v,-v,0.],dtype=torch.float64))
    def test_zero_and_boundary_response_hinge_gradient_zero(self):
        for value in (0.,RESPONSE_SAFETY):
            o=torch.tensor([value]*6+[.1]*16,dtype=torch.float64,requires_grad=True);p=torch.zeros(DIM,dtype=torch.float32);a=torch.zeros(3,dtype=torch.float64);R=loss_parts(o,torch.tensor(0.),p,p,a,'xor_00')['response_loss'];g=torch.autograd.grad(R,o)[0];self.assertEqual(float(R),0.);self.assertEqual(float(g.abs().max()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
