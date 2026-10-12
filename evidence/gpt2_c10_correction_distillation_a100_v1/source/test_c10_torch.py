import io,unittest
import torch
from proposer import WriterHead,one_forward_proposal
from distillation_loss import budget_terms,weighted_terms

class Tests(unittest.TestCase):
    def test_budget_scaled_response_has_gradient(self):
        logits=torch.zeros(5,4,requires_grad=True);reference=torch.zeros(1,4)
        with torch.no_grad():logits[0,0]=.01
        t=budget_terms(logits,1,[0],reference,[1,2],torch.tensor([0,1,1,0]))
        self.assertAlmostEqual(t['response'].item(),4.5,places=5);t['response'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(logits.grad[0,0].item(),0)
    def test_zero_response_reference_zero_loss(self):
        logits=torch.zeros(5,4);t=budget_terms(logits,1,[0],torch.zeros(1,4),[1,2],torch.tensor([0,1,1,0]))
        self.assertEqual(t['response'].item(),0);self.assertEqual(t['KL'].item(),0)
    def test_four_graph_accumulation_equals_mean_gradient(self):
        p=torch.nn.Parameter(torch.tensor(2.));targets=torch.tensor([1.,2.,3.,4.])
        for t in targets:((p-t).square()/4).backward()
        grad=p.grad.clone();p.grad=None;(p-targets).square().mean().backward();self.assertTrue(torch.equal(grad,p.grad))
    def test_writer_target_fit_has_finite_gradients(self):
        h=WriterHead(3,2,4);loss=(h(torch.randn(12,3))-torch.randn(12,2)).square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters()))
    def test_frozen_writer_one_forward_no_change(self):
        h=WriterHead(3,2,4);h.eval()
        for p in h.parameters():p.requires_grad_(False)
        before=[p.clone() for p in h.parameters()];n={'mean':torch.zeros(3),'scale':torch.ones(3),'output_scale':torch.tensor(1.)}
        one_forward_proposal(h,torch.ones(3),n);self.assertTrue(all(torch.equal(p,q) for p,q in zip(h.parameters(),before)))
    def test_functional_override_preserves_memory_parameter(self):
        from torch.func import functional_call
        m=torch.nn.Linear(2,1,bias=False);m.weight.requires_grad_(False);version=m.weight._version;original=m.weight.clone()
        h=torch.nn.Linear(2,2);c=h(torch.ones(1,2));functional_call(m,{'weight':c},(torch.ones(1,2),)).sum().backward()
        self.assertIsNotNone(h.weight.grad);self.assertIsNone(m.weight.grad);self.assertEqual(m.weight._version,version);self.assertTrue(torch.equal(m.weight,original))
    def test_checkpoint_optimizer_next_step_replay(self):
        torch.manual_seed(1);h=WriterHead(3,2,4);o=torch.optim.AdamW(h.parameters(),lr=1e-5,weight_decay=0.);x=torch.randn(12,3);y=torch.randn(12,2)
        def step(a,opt):opt.zero_grad();(a(x)-y).square().mean().backward();opt.step()
        step(h,o);b=io.BytesIO();torch.save({'h':h.state_dict(),'o':o.state_dict()},b);b.seek(0);cp=torch.load(b,weights_only=True)
        h2=WriterHead(3,2,4);h2.load_state_dict(cp['h']);o2=torch.optim.AdamW(h2.parameters(),lr=1e-5,weight_decay=0.);o2.load_state_dict(cp['o'])
        step(h,o);step(h2,o2);self.assertTrue(all(torch.equal(p,q) for p,q in zip(h.parameters(),h2.parameters())))

if __name__=='__main__':unittest.main()
