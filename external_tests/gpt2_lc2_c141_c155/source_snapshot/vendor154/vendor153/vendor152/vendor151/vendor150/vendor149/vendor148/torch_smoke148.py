"""Actual CPU Torch tests, run on Colab before any GPT2 load."""
from core148 import *
import torch,unittest
import torch_smoke146 as prior
from functional148 import Backend
from functional146 import loss_parts,scalar_gradients,Endpoint,torch_proposal

class OriginalGraphChecks(prior.OriginalGraphChecks):pass
class PriorDirectionalChecks(prior.PriorDirectionalChecks):pass
class PriorWorstResponseChecks(prior.WorstResponseChecks):pass

class RoutingChecks(unittest.TestCase):
    def test_actual_backend_effective_leaf_and_float32_commit(self):
        theta,raw,V=prior.WorstResponseChecks().constants();w=np.array([.2,.3,.5]);A=np.array([[.01,-.02,.03],[.02,-.01,.01],[-.01,.03,.02]]);d=dict(V=V,weights=dict(toy=w),points=dict(toy=dict(prepared=theta,raw=raw)));be=Backend(prior.previous.previous.runtime(),d,None,{});point,a=be.point('toy',torch.tensor(A,dtype=torch.float64,requires_grad=True),TRAIN_ARMS[1]);expected,_=proposal(theta,raw,V,effective(A,w,TRAIN_ARMS[1]));np.testing.assert_allclose(point.detach().numpy(),expected,atol=2e-6,rtol=0);exact(a.detach().numpy(),effective(A,w,TRAIN_ARMS[1]),'Exact committed effective control');self.assertTrue(a.is_leaf);g=torch.autograd.grad(point.double().square().sum(),a)[0];self.assertTrue(bool(torch.isfinite(g).all()));self.assertGreater(float(g.abs().max()),0.);be.verify(rehash=True)
    def test_linear_pullback_matches_direct_autograd(self):
        w=torch.tensor([.2,.3,.5],dtype=torch.float64);A=torch.arange(9,dtype=torch.float64).reshape(3,3).clone().requires_grad_();a=w@A;loss=(a.square().sum()+a.prod());effective_g=torch.autograd.grad(loss,a,retain_graph=True)[0];direct=torch.autograd.grad(loss,A)[0];np.testing.assert_allclose(direct.numpy(),lift_gradient(effective_g.numpy(),w.numpy(),TRAIN_ARMS[1]),atol=1e-12,rtol=1e-12)
    def test_router_effective_values_match_torch(self):
        U=np.arange(84*3,dtype=float).reshape(84,3)/100;r,W,_=fit_router(U);A=np.arange(9,dtype=float).reshape(3,3)/100
        np.testing.assert_allclose(effective(A,W[13],TRAIN_ARMS[1]),(torch.from_numpy(W[13])@torch.from_numpy(A)).numpy(),atol=1e-15)
    def test_actual_endpoint_local_scalar_gradients_match_direct_full_head(self):
        theta,raw,V=prior.WorstResponseChecks().constants();w=torch.tensor([.2,.3,.5],dtype=torch.float64);A=torch.tensor([[.01,-.02,.03],[.02,-.01,.01],[-.01,.03,.02]],dtype=torch.float64,requires_grad=True);a=w@A;ep=Endpoint(prior.previous.previous.runtime());p=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),a);r=scalar_gradients(ep,p,torch.from_numpy(theta),a,'xor_00');A2=A.detach().clone().requires_grad_();a2=w@A2;p2=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),a2);o,k=ep(p2);parts=loss_parts(o,k,p2,torch.from_numpy(theta),a2,'xor_00');gr=torch.autograd.grad(parts['response_loss'],A2,retain_graph=True)[0];gg=torch.autograd.grad(sum(v for key,v in parts.items() if key!='response_loss'),A2)[0]
        for key,actual in [('response_gradient',gr),('guard_gradient',gg)]:np.testing.assert_allclose(lift_gradient(np.array(r[key]),w.numpy(),TRAIN_ARMS[1]),actual.detach().numpy(),rtol=1e-10,atol=1e-12)
        ep.verify(rehash=True)
    def test_state_regularizer_uses_effective_control(self):
        A=torch.tensor([[.01,.02],[.02,.03],[-.02,.01]],dtype=torch.float64,requires_grad=True);w=torch.tensor([.1,.2,.7],dtype=torch.float64);a=w@A;l=.001*a.square().mean();g=torch.autograd.grad(l,A)[0];np.testing.assert_allclose(g.numpy(),lift_gradient(.002*a.detach().numpy()/2,w.numpy(),TRAIN_ARMS[1]),atol=1e-15)
    def test_global_regularizer_unchanged(self):
        a=torch.tensor([.01,-.02,.03],dtype=torch.float64,requires_grad=True);l=.001*a.square().mean();g=torch.autograd.grad(l,a)[0];np.testing.assert_allclose(g.numpy(),.002*a.detach().numpy()/3,atol=1e-15)
    def test_float32_routed_commit_has_connected_graph(self):
        theta,raw,V=prior.WorstResponseChecks().constants();A=torch.full((3,3),.01,dtype=torch.float64,requires_grad=True);w=torch.tensor([.2,.3,.5],dtype=torch.float64);point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),w@A);g=torch.autograd.grad(point.double().square().sum(),A)[0];self.assertEqual(point.dtype,torch.float32);self.assertTrue(bool(torch.isfinite(g).all()));self.assertGreater(float(g.abs().max()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
