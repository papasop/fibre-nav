"""Required Colab tests; toy models, no downloads."""
import io,unittest
import torch
from proposer import WriterHead,fit_normalisation,normalise,one_forward_proposal

class Tests(unittest.TestCase):
    def test_writer_has_finite_gradients(self):
        h=WriterHead(3,4,8);x=torch.randn(12,3);y=torch.randn(12,4);loss=(h(x)-y).square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))
    def test_training_normalisation_independent_of_test(self):
        X=torch.tensor([[1.,2],[2,3],[3,4]]);Y=torch.ones(3,2);n=fit_normalisation(X,Y)
        before={k:v.clone() for k,v in n.items()};normalise(torch.tensor([1e6,-1e6]),n)
        self.assertTrue(all(torch.equal(n[k],v) for k,v in before.items()))
    def test_one_forward_frozen_proposal_no_update(self):
        h=WriterHead(3,2,8);h.eval()
        for p in h.parameters():p.requires_grad_(False)
        n=fit_normalisation(torch.randn(12,3),torch.randn(12,2));before=[p.clone() for p in h.parameters()]
        one_forward_proposal(h,torch.randn(3),n)
        self.assertTrue(all(torch.equal(p,b) for p,b in zip(h.parameters(),before)))
    def test_optimizer_roundtrip_reproduces_next_step(self):
        torch.manual_seed(1);h=WriterHead(3,2,4);opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=0.)
        X=torch.randn(12,3);Y=torch.randn(12,2)
        def step(a,o):o.zero_grad();(a(X)-Y).square().mean().backward();o.step()
        step(h,opt);b=io.BytesIO();torch.save({'h':h.state_dict(),'o':opt.state_dict()},b);b.seek(0);cp=torch.load(b,weights_only=True)
        h2=WriterHead(3,2,4);h2.load_state_dict(cp['h']);o2=torch.optim.AdamW(h2.parameters(),lr=.001,weight_decay=0.);o2.load_state_dict(cp['o'])
        step(h,opt);step(h2,o2);self.assertTrue(all(torch.equal(p,q) for p,q in zip(h.parameters(),h2.parameters())))
    def test_functional_candidate_gradient_frozen_memory(self):
        from torch.func import functional_call
        m=torch.nn.Linear(2,1,bias=False);m.weight.requires_grad_(False);original=m.weight.clone();version=m.weight._version
        h=torch.nn.Linear(2,2);candidate=h(torch.ones(1,2));out=functional_call(m,{'weight':candidate},(torch.ones(1,2),))
        out.sum().backward();self.assertIsNotNone(h.weight.grad);self.assertIsNone(m.weight.grad)
        self.assertTrue(torch.equal(m.weight,original));self.assertEqual(m.weight._version,version)
    def test_no_grad_physical_state_commit(self):
        p=torch.nn.Parameter(torch.zeros(3),requires_grad=False)
        with torch.no_grad():p.copy_(torch.ones(3))
        self.assertFalse(p.requires_grad);self.assertTrue(torch.equal(p,torch.ones(3)))

if __name__=='__main__':unittest.main()
