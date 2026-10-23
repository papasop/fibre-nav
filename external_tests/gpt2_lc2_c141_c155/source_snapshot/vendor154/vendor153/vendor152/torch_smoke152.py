"""Actual CPU Torch tests, run on Colab before any GPT2 load."""
from core152 import *
import torch,unittest
import torch_smoke146 as prior
from functional152 import Backend,loss_parts,scalar_gradients,Endpoint,torch_proposal
from worker152 import resolve_default_factories
from types import SimpleNamespace

class OriginalGraphChecks(prior.OriginalGraphChecks):pass
class PriorDirectionalChecks(prior.PriorDirectionalChecks):pass
class PriorWorstResponseChecks(prior.WorstResponseChecks):pass

class CouplingChecks(unittest.TestCase):
    def constants(self):return prior.WorstResponseChecks().constants()
    def test_worker_default_factory_creates_current_backend(self):
        rf,bf=resolve_default_factories();self.assertEqual(rf.__module__,'runtime152');self.assertEqual(bf.__module__,'functional152')
        theta,raw,V=self.constants();d=dict(V=V,weights=dict(toy=np.array([.2,.3,.5])),points=dict(toy=dict(prepared=theta,raw=raw)));be=bf(prior.previous.previous.runtime(),d,None,{})
        self.assertEqual(type(be).__module__,'functional152');self.assertEqual(set(be.stats()['per_arm']),set(TRAIN_ARMS));be.verify(rehash=True)
    def test_worker_default_backend_zero_parity_accepts_both_cyclic_rate_arms(self):
        _,bf=resolve_default_factories();theta,raw,V=self.constants();d=dict(V=V,weights=dict(toy=np.array([.2,.3,.5])),points=dict(toy=dict(prepared=theta,raw=raw)))
        for arm in TRAIN_ARMS:
            seen=[]
            phase=SimpleNamespace(phase='functional_zero',functional=lambda sid,p,a,point:(seen.append((a,p.shape)) or 0.))
            be=bf(prior.previous.previous.runtime(),d,phase,{});p=np.zeros((3,2,3));r=be.parity(dict(state_id='toy',case='xor_00'),p,arm)
            self.assertEqual(r['arm'],arm);self.assertEqual(seen,[(arm,(3,2,3))]);self.assertEqual(be.stats()['functional_zero_calls'],1);be.verify(rehash=True)
    def test_coupled_backend_effective_leaf_and_exact_numpy_commit(self):
        theta,raw,V=self.constants();w=np.array([.2,.3,.5]);A=np.arange(18,dtype=float).reshape(3,2,3)/200;d=dict(V=V,weights=dict(toy=w),points=dict(toy=dict(prepared=theta,raw=raw)));be=Backend(prior.previous.previous.runtime(),d,None,{});point,e=be.point('toy',torch.tensor(A,dtype=torch.float64,requires_grad=True),TRAIN_ARMS[1]);expected,_=proposal(theta,raw,V,effective(A,w,TRAIN_ARMS[1]));np.testing.assert_allclose(point.detach().numpy(),expected,atol=2e-6,rtol=0);exact(e.detach().numpy(),effective(A,w,TRAIN_ARMS[1]),'Exact effective leaf');self.assertTrue(e.is_leaf);g=torch.autograd.grad(point.double().square().sum(),e)[0];self.assertGreater(float(g[1].abs().max()),0.);be.verify(rehash=True)
    def test_coupled_direct_autograd_matches_manual_router_pullback(self):
        w=torch.tensor([.2,.3,.5],dtype=torch.float64);A=torch.arange(18,dtype=torch.float64).reshape(3,2,3).clone().requires_grad_();e=torch.stack([w@A[:,j,:] for j in range(2)]);l=e.square().sum()+e.prod();g=torch.autograd.grad(l,e,retain_graph=True)[0];direct=torch.autograd.grad(l,A)[0];np.testing.assert_allclose(direct.numpy(),pullback(g.numpy(),w.numpy(),TRAIN_ARMS[1]),atol=1e-10,rtol=1e-12)
    def test_coupled_endpoint_scalar_gradients_match_direct_full_head(self):
        theta,raw,V=self.constants();w=torch.tensor([.2,.3,.5],dtype=torch.float64);A=torch.arange(18,dtype=torch.float64).reshape(3,2,3)/200;A.requires_grad_();e=torch.stack([w@A[:,j,:] for j in range(2)]);ep=Endpoint(prior.previous.previous.runtime());p=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),e);r=scalar_gradients(ep,p,torch.from_numpy(theta),e,'xor_00');B=A.detach().clone().requires_grad_();f=torch.stack([w@B[:,j,:] for j in range(2)]);q=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(V),f);o,k=ep(q);parts=loss_parts(o,k,q,torch.from_numpy(theta),f,'xor_00');gr=torch.autograd.grad(parts['response_loss'],B,retain_graph=True)[0];gg=torch.autograd.grad(sum(v for key,v in parts.items() if key!='response_loss'),B)[0]
        for key,direct in [('response_gradient',gr),('guard_gradient',gg)]:np.testing.assert_allclose(pullback(np.array(r[key]),w.numpy(),TRAIN_ARMS[1]),direct.numpy(),rtol=1e-10,atol=1e-12)
        ep.verify(rehash=True)
    def test_zero_cross_block_preserves_original_regularizer_and_commit(self):
        theta,raw,V=self.constants();e=torch.tensor([.01,-.02,.03],dtype=torch.float64,requires_grad=True);f=torch.stack([e,torch.zeros_like(e)]);T,R,M=map(torch.from_numpy,(theta,raw,V));p=torch_proposal(T,R,M,e);q=torch_proposal(T,R,M,f);self.assertTrue(torch.equal(p,q));o,k=Endpoint(prior.previous.previous.runtime())(p);a=loss_parts(o,k,p,T,e,'xor_00');b=loss_parts(o,k,q,T,f,'xor_00');self.assertEqual(float(a['regularizer']),float(b['regularizer']));self.assertEqual(float(a['regularizer']),float(.001*e.square().mean()))
    def test_cross_block_uses_positive_fixed_cyclic_shift(self):
        T=torch.zeros(3,dtype=torch.float32);R=torch.tensor([1.,2.,3.],dtype=torch.float32);V=torch.eye(3,dtype=torch.float64);e=torch.tensor([[0.,0.,0.],[.01,.02,.03]],dtype=torch.float64,requires_grad=True);p=torch_proposal(T,R,V,e);np.testing.assert_allclose(p.detach().numpy(),[1.03,2.02,3.06],atol=2e-7);g=torch.autograd.grad(p.sum(),e)[0];np.testing.assert_allclose(g[1].numpy(),[3.,1.,2.])
if __name__=='__main__':unittest.main(verbosity=2)
