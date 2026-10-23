"""Mandatory real CPU Torch forward-only smoke, before GPT2; no backward."""
import torch,unittest
from core154 import *
from worker154 import resolve_default_runtime_factory

class Checks(unittest.TestCase):
    def test_actual_default_factory_chain_is_C154_then_pinned_native(self):
        f=resolve_default_runtime_factory();self.assertEqual(f.__module__,'runtime154');import runtime154;self.assertEqual(runtime154.resolve_pinned_factory().__module__,'runtime152')
    def test_all_signed_proposals_match_Torch_float32_commit(self):
        theta=np.zeros(DIM,dtype=np.float32);raw=theta.copy();raw[:3]=[1.,2.,3.];V=np.zeros((3,DIM));V[:,:3]=np.eye(3);w=np.array([.2,.3,.5]);b=np.arange(18,dtype=float).reshape(3,2,3)/1000;a=b+np.linspace(-1e-5,1e-5,18).reshape(b.shape)
        with torch.no_grad():
            T,R,M=map(torch.from_numpy,(theta,raw,V));u=M@(R.double()-T.double())
            for alpha in LABELS.values():
                p=interpolate(b,a,alpha);e=prior.effective(p,w,ARM);E=torch.from_numpy(e);point=(R.double()+M.T@(E[0]*u+E[1]*torch.roll(u,1))).float();want=prior.proposal(theta,raw,V,e);np.testing.assert_allclose(point.numpy(),want,atol=2e-6,rtol=0);self.assertFalse(point.requires_grad)
    def test_scalar_loss_units_match_original_C152(self):
        o=np.r_[np.array([.003,-.002,.001,0.,0.,0.]),np.linspace(-.1,.1,16)];p=np.zeros(DIM,dtype=np.float32);e=np.full((2,3),.01);expected=prior.loss(o,.006,p,p,e,'xor_01')
        with torch.no_grad():
            O=torch.from_numpy(o);E=torch.from_numpy(e);sign=torch.tensor(2*np.tile([0,1,1,1],4)-1,dtype=torch.float64);R=torch.relu(O[:6].abs().max()/.0018-1).square();G=torch.relu(1-sign*O[6:]/.075).square().mean();self.assertAlmostEqual(float(R),expected['response_loss']);self.assertAlmostEqual(float(G),expected['read_loss']);self.assertAlmostEqual(float(.001*E.square().sum()/3),expected['regularizer'])
    def test_forward_only_frozen_parameters_have_no_gradients(self):
        m=torch.nn.Linear(3,2).cpu().float();m.requires_grad_(False)
        with torch.no_grad():o=m(torch.ones(4,3))
        self.assertFalse(o.requires_grad);self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.parameters()))
if __name__=='__main__':unittest.main(verbosity=2)
