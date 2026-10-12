import types,unittest,torch
from torch.func import functional_call
from functional_writer import chart_spec,flat_overrides,last_logits,objective_terms,combined_loss
from proposer import WriterHead,fit_normalisation,normalise
from c7_protocol import SETTINGS

class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.base=torch.nn.Linear(2,4,bias=False)
  self.lora_B=torch.nn.ModuleDict({'default':torch.nn.Linear(2,4,bias=False)})
  for p in self.parameters():p.requires_grad_(False)
 def forward(self,input_ids,attention_mask,use_cache=False):
  x=torch.stack([input_ids.float(),input_ids.float()*.3+1],-1)
  return types.SimpleNamespace(logits=self.base(x)+self.lora_B['default'](x))

class TorchTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(3);self.model=Toy();self.params=[self.model.lora_B['default'].weight]
  self.spec=chart_spec(self.model,self.params)
  self.batch={'input_ids':torch.tensor([[1,2],[2,3],[1,4],[2,4],[1,2]]),'attention_mask':torch.ones(5,2,dtype=torch.long)}
 def test_real_functional_path_gradients_reach_head_and_model_unchanged(self):
  before={n:p.clone() for n,p in self.model.named_parameters()}
  head=WriterHead(3,8,4);delta=head(torch.ones(1,3))[0]
  base=self.params[0].detach().flatten();logits=last_logits(self.model,self.batch,base+delta,self.spec)
  loss=logits.square().sum();loss.backward()
  self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.parameters()),0)
  self.assertTrue(all(torch.equal(before[n],p) for n,p in self.model.named_parameters()))
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))
 def test_functional_equals_committed_forward(self):
  theta=self.params[0].detach().flatten()+.02
  a=last_logits(self.model,self.batch,theta,self.spec)
  with torch.no_grad():self.params[0].copy_(theta.reshape(4,2))
  b=self.model(**self.batch).logits[:,-1]
  torch.testing.assert_close(a,b)
 def test_negative_control_copy_severs_writer_gradient(self):
  head=WriterHead(3,8,4);delta=head(torch.ones(1,3))[0]
  with torch.no_grad():self.params[0].copy_(delta.reshape(4,2))
  loss=self.model(**self.batch).logits.square().sum()
  self.assertFalse(loss.requires_grad)
 def test_wrong_chart_dim_rejected(self):
  with self.assertRaises(ValueError):flat_overrides(torch.zeros(7),self.spec)
 def test_identical_head_initialisation_for_two_arms(self):
  torch.manual_seed(6);a=WriterHead(3,8,4)
  torch.manual_seed(6);b=WriterHead(3,8,4)
  self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
 def test_normalisation_only_training_data(self):
  X=torch.tensor([[1.,2.],[3.,4.]])
  norm=fit_normalisation(X,torch.ones(2,8));self.assertTrue(torch.equal(norm['mean'],torch.tensor([2.,3.])))
 def test_objective_terms_actual_margin_response_KL(self):
  ref=torch.zeros(1,4);logits=torch.zeros(5,4,requires_grad=True)
  terms=objective_terms(logits,1,[0,1],ref,[2,3],torch.tensor([0,1,1,0]))
  self.assertEqual(float(terms['maximum_response_drift'].detach()),0)
  self.assertEqual(float(terms['measured_mean_KL'].detach()),0)
  loss=combined_loss(logits.square().mean(),terms,SETTINGS['loss_weights']);loss.backward()
  self.assertGreater(float(logits.grad.abs().sum()),0)
 def test_response_surrogate_gradient_bounded_and_not_budget_scaled(self):
  logits=torch.zeros(5,4,requires_grad=True)
  with torch.no_grad():logits[0,:2]=1.
  terms=objective_terms(logits,1,[0,1],torch.zeros(1,4),[2,3],torch.tensor([0,1,1,0]))
  weighted=SETTINGS['loss_weights']['response']*terms['response']
  grad=torch.autograd.grad(weighted,logits)[0]
  self.assertLessEqual(float(grad.abs().max()),.050001)
  self.assertAlmostEqual(float(weighted.detach()),.095,places=6)
 def test_stage_clone_same_initial_values_distinct_storage(self):
  a=WriterHead(3,8,4);b=WriterHead(3,8,4);b.load_state_dict(a.state_dict())
  for x,y in zip(a.parameters(),b.parameters()):
   self.assertTrue(torch.equal(x,y));self.assertNotEqual(x.data_ptr(),y.data_ptr())
 def test_full_batch_mse_decreases_on_three_distinct_teachers(self):
  torch.manual_seed(11);h=WriterHead(3,8,8);opt=torch.optim.AdamW(h.parameters(),lr=.02,weight_decay=0)
  X=torch.eye(3);Y=torch.arange(24).reshape(3,8).float()/10
  before=float((h(X)-Y).square().mean().detach())
  for _ in range(80):
   opt.zero_grad();loss=(h(X)-Y).square().mean();loss.backward();opt.step()
  after=float((h(X)-Y).square().mean().detach());self.assertLess(after,before/20)
 def test_optimizer_resume_matches_uninterrupted(self):
  import io
  torch.manual_seed(7);head=WriterHead(3,8,4);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0)
  x=torch.ones(1,3);y=torch.arange(8).float().reshape(1,-1)
  def step(h,o):o.zero_grad();(h(x)-y).square().mean().backward();o.step()
  step(head,opt);b=io.BytesIO();torch.save({'head':head.state_dict(),'opt':opt.state_dict()},b)
  step(head,opt);expected=[p.clone() for p in head.parameters()]
  cp=torch.load(io.BytesIO(b.getvalue()),weights_only=True)
  other=WriterHead(3,8,4);other.load_state_dict(cp['head']);oo=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=0);oo.load_state_dict(cp['opt'])
  step(other,oo);self.assertTrue(all(torch.equal(a,c) for a,c in zip(expected,other.parameters())))
 def test_override_restored_on_exception(self):
  class Bad(Toy):
   def forward(self,**kw):raise RuntimeError('deliberate')
  m=Bad();p=m.lora_B['default'].weight;before=p.clone();spec=chart_spec(m,[p])
  with self.assertRaises(RuntimeError):last_logits(m,self.batch,torch.zeros(8),spec)
  self.assertTrue(torch.equal(before,m.lora_B['default'].weight))

if __name__=='__main__':unittest.main()
