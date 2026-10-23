"""Mandatory real CPU PyTorch graph/parity smoke, before loading GPT2."""
import types,unittest
import torch
from functional143 import Endpoint,torch_proposal,loss_parts
from core143 import *

class Batch(dict):
    def to(self,device):return Batch({k:v.to(device) for k,v in self.items()})

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.b1=torch.nn.Parameter(torch.zeros(4),requires_grad=False);self.b2=torch.nn.Parameter(torch.zeros(DIM-4),requires_grad=False);self.fixed=torch.nn.Parameter(torch.tensor(.1),requires_grad=False);self.register_buffer('buffer',torch.zeros(8))
    def forward(self,input_ids,attention_mask,use_cache=False):
        p=torch.cat([self.b1,self.b2]);v=torch.cat([p[:6]+.1*p[:6].square(),self.fixed[None],(-self.fixed)[None]])+self.buffer
        return types.SimpleNamespace(logits=v[None,None,:].expand(*input_ids.shape,-1))

def runtime():
    rt=types.SimpleNamespace(model=Tiny(),device=torch.device('cpu'),anchors=['a']*12,read_prompts=['r']*16,responseids=list(range(6)),readids=[6,7]);rt.params=[rt.model.b1,rt.model.b2]
    rt.tok=lambda prompts,**kw:Batch(input_ids=torch.zeros((len(prompts),2),dtype=torch.long),attention_mask=torch.ones((len(prompts),2),dtype=torch.long))
    with torch.no_grad():rt.logits0=rt.model(**rt.tok(rt.anchors)).logits[:,1];rt.response0=rt.logits0[:,:6].mean(0)
    def invariant():need(all(not p.requires_grad and p.grad is None for p in rt.model.parameters()),'Unexpected active model gradient')
    rt.invariant=invariant;return rt

class Checks(unittest.TestCase):
    def constants(self):
        theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[:6]=np.arange(1,7)*.001;v=np.zeros((3,DIM));v[:3,:3]=np.eye(3);return theta,raw,v
    def test_actual_functional_gradient_and_parameter_restoration(self):
        rt=runtime();ep=Endpoint(rt);theta,raw,v=self.constants();a=torch.tensor([.01,-.02,.03],dtype=torch.float64,requires_grad=True);point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(v),a);o,k=ep(point);g=torch.autograd.grad(o[:6].sum(),a)[0]
        expected=torch.from_numpy(raw[:3].astype(float))*(1+.2*point[:3].detach().double());torch.testing.assert_close(g,expected,rtol=2e-6,atol=1e-10);self.assertGreater(float(g.norm()),0);ep.verify(rehash=True)
    def test_zero_control_is_exact_raw_and_nonzero_numpy_torch_map_matches(self):
        theta,raw,v=self.constants()
        for a in (np.zeros(3),np.array([.05,-.04,.03])):
            point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(v),torch.from_numpy(a));expected,_=proposal(theta,raw,v,a);np.testing.assert_allclose(point.numpy(),expected,atol=2e-6,rtol=0)
            if not a.any():self.assertEqual(safe.tensor_hash(point.numpy()),safe.tensor_hash(raw))
    def test_training_loss_units_match_numpy(self):
        theta,raw,v=self.constants();a=np.array([.01,-.02,.03]);point,_=proposal(theta,raw,v,a);observed=np.r_[np.array([.003,-.004,.001,.002,.0,-.003]),np.linspace(-.15,.15,16)];kl=.006
        t=loss_parts(torch.from_numpy(observed),torch.tensor(kl,dtype=torch.float64),torch.from_numpy(point),torch.from_numpy(theta),torch.from_numpy(a),'xor_00');n=objective(observed,kl,point,theta,a,'xor_00')
        compare({k:float(x) for k,x in t.items()},n,'Torch loss parity',atol=1e-10,rtol=1e-9)
    def test_manual_Adam_matches_torch_projected_first_update(self):
        a=np.array([.001,-.002,.003]);g=np.array([2.,-3.,4.]);new,_,_,_=adam_step(a,g,np.zeros(3),np.zeros(3),1);p=torch.nn.Parameter(torch.from_numpy(a.copy()));clipped=g*min(1.,1./np.linalg.norm(g));p.grad=torch.from_numpy(clipped);opt=torch.optim.Adam([p],lr=RATE,betas=(ADAM_B1,ADAM_B2),eps=ADAM_EPS);opt.step()
        with torch.no_grad():p.clamp_(-GAIN_BOUND,GAIN_BOUND)
        np.testing.assert_allclose(p.detach().numpy(),new,atol=2e-15,rtol=1e-10)
    def test_frozen_backbone_and_buffer_mutation_detected(self):
        rt=runtime();ep=Endpoint(rt)
        with torch.no_grad():rt.model.fixed.add_(.1)
        with self.assertRaises(ValueError):ep.verify(rehash=True)
    def test_physical_chart_mutation_detected(self):
        rt=runtime();ep=Endpoint(rt)
        with torch.no_grad():rt.model.b1[0]=.1
        with self.assertRaises(ValueError):ep.verify()

    def test_weighted_state_gradients_match_full_torch_logsumexp_graph(self):
        a=torch.tensor([.04,-.03,.02],dtype=torch.float64,requires_grad=True)
        matrix=torch.arange(N*3,dtype=torch.float64).reshape(N,3)/(N*3-1)
        losses=.01+((matrix@a)+torch.linspace(.01,.3,N,dtype=torch.float64)).square()
        unweighted=torch.stack([torch.autograd.grad(x,a,retain_graph=True)[0] for x in losses])
        full=UNIFORM_FRACTION*losses.mean()+(1-UNIFORM_FRACTION)*(torch.logsumexp(TEMPERATURE*losses,dim=0)-np.log(N))/TEMPERATURE
        expected=torch.autograd.grad(full,a)[0].numpy();actual,rec=aggregate(losses.detach().numpy(),unweighted.numpy())
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-12);self.assertAlmostEqual(rec['objective'],float(full.detach()),14)

    def test_nonzero_initial_control_preserves_functional_gradient_connection(self):
        rt=runtime();ep=Endpoint(rt);theta,raw,v=self.constants();a=torch.tensor([.046,-.025,.015],dtype=torch.float64,requires_grad=True)
        point=torch_proposal(torch.from_numpy(theta),torch.from_numpy(raw),torch.from_numpy(v),a);o,k=ep(point);terms=loss_parts(o,k,point,torch.from_numpy(theta),a,'xor_00')
        g=torch.autograd.grad(sum(terms.values()),a)[0];self.assertTrue(bool(torch.isfinite(g).all()));self.assertGreater(float(g.norm()),0);ep.verify(rehash=True)

if __name__=='__main__':unittest.main(verbosity=2)
