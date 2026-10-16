import copy,hashlib,io,unittest
import torch
from raw_writer import StateWriter,one_raw_proposal
from guarded_update import attempt_update,cpu_state
from functional_writer import last_logits

def same_tree(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
 return a==b
class Tests(unittest.TestCase):
 def fixture(self):
  h=torch.nn.Linear(1,1,bias=False)
  with torch.no_grad():h.weight.fill_(1)
  opt=torch.optim.AdamW(h.parameters(),lr=.1,weight_decay=0.)
  return h,opt
 def test_rejection_restores_head_and_optimizer(self):
  h,o=self.fixture();old=cpu_state(h);opt=copy.deepcopy(o.state_dict())
  r,rec=attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),lambda:{},lambda r:{'accepted':False},3)
  self.assertIsNone(r);self.assertFalse(rec['accepted']);self.assertTrue(same_tree(old,cpu_state(h)));self.assertTrue(same_tree(opt,o.state_dict()))
 def test_exception_restores_head_and_moments(self):
  h,o=self.fixture();old=cpu_state(h);opt=copy.deepcopy(o.state_dict())
  def bad():raise RuntimeError('physical evaluator failure')
  with self.assertRaises(RuntimeError):attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),bad,lambda r:{'accepted':True})
  self.assertTrue(same_tree(old,cpu_state(h)));self.assertTrue(same_tree(opt,o.state_dict()))
 def test_acceptance_keeps_real_adam_moments(self):
  h,o=self.fixture();r,rec=attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),lambda:{'ok':True},lambda r:{'accepted':True})
  self.assertTrue(rec['accepted']);self.assertEqual(int(next(iter(o.state.values()))['step']),1);self.assertLess(float(h.weight.detach()),1.)
 def test_fractional_update_uses_original_point(self):
  h,o=self.fixture();seen=[]
  def evaluate():seen.append(float(h.weight.detach()));return {'w':seen[-1]}
  r,rec=attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),evaluate,lambda r:{'accepted':len(seen)==3},4)
  self.assertEqual(rec['alpha'],.25);self.assertAlmostEqual(1-seen[2],(1-seen[0])*.25,places=6)
 def test_preaccumulated_gradient_path(self):
  h,o=self.fixture()
  def loss():
   h(torch.ones(1,1)).square().mean().backward();return torch.tensor(1.),{'gradients_accumulated':True}
  r,rec=attempt_update(h,o,loss,lambda:{},lambda r:{'accepted':True});self.assertTrue(rec['accepted']);self.assertLess(float(h.weight.detach()),1.)
 def test_backtracking_tries_all_declared_scales(self):
  h,o=self.fixture();seen=[];attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),lambda:{},lambda r:{'accepted':False},8,progress=lambda b,a:seen.append(a))
  self.assertEqual(seen,[2.**(-i) for i in range(8)])
 def test_nonfinite_loss_rejected(self):
  h,o=self.fixture();old=cpu_state(h)
  with self.assertRaises(ValueError):attempt_update(h,o,lambda:(h.weight.sum()*float('nan'),{}),lambda:{},lambda r:{'accepted':True})
  self.assertTrue(same_tree(old,cpu_state(h)))
 def test_resume_actual_adam_matches_uninterrupted(self):
  h,o=self.fixture();fn=lambda:(h(torch.ones(1,1)).square().mean(),{});attempt_update(h,o,fn,lambda:{},lambda r:{'accepted':True})
  b=io.BytesIO();torch.save({'head':cpu_state(h),'optimiser':o.state_dict()},b);b.seek(0);saved=torch.load(b,weights_only=True)
  h2,o2=self.fixture();h2.load_state_dict(saved['head']);o2.load_state_dict(saved['optimiser'])
  attempt_update(h,o,fn,lambda:{},lambda r:{'accepted':True});attempt_update(h2,o2,lambda:(h2(torch.ones(1,1)).square().mean(),{}),lambda:{},lambda r:{'accepted':True})
  self.assertTrue(same_tree(cpu_state(h),cpu_state(h2)));self.assertTrue(same_tree(o.state_dict(),o2.state_dict()))
 def test_frozen_one_call_no_gradient(self):
  h=StateWriter(3,2,4)
  for p in h.parameters():p.requires_grad_(False)
  calls=[];hook=h.register_forward_hook(lambda *a:calls.append(1));one_raw_proposal(h,torch.ones(3),torch.tensor(1),torch.tensor(.1));hook.remove()
  self.assertEqual(len(calls),1);self.assertTrue(all(p.grad is None for p in h.parameters()))
 def test_inactive_expert_momentum_does_not_move_it(self):
  h=StateWriter(3,2,4);o=torch.optim.AdamW(h.parameters(),lr=.01,weight_decay=0.)
  for p in h.experts[0].parameters():p.grad=torch.ones_like(p)
  o.step();o.zero_grad(set_to_none=True);before={k:v.clone() for k,v in h.experts[0].state_dict().items()}
  for i,e in enumerate(h.experts):
   for p in e.parameters():p.requires_grad_(i==1);p.grad=None
  attempt_update(h,o,lambda:(h(torch.ones(1,3),torch.tensor([1])).square().mean(),{}),lambda:{},lambda r:{'accepted':True})
  self.assertTrue(same_tree(before,h.experts[0].state_dict()))
 def test_functional_override_grads_only_writer(self):
  class Tiny(torch.nn.Module):
   def __init__(self):super().__init__();self.B=torch.nn.Parameter(torch.zeros(2),requires_grad=False);self.base=torch.nn.Parameter(torch.ones(2),requires_grad=False)
   def forward(self,input_ids,attention_mask,use_cache=False):
    class Out:pass
    r=Out();r.logits=(self.B+self.base).view(1,1,2).expand(input_ids.shape[0],input_ids.shape[1],2);return r
  model=Tiny();head=torch.nn.Linear(1,2);candidate=head(torch.ones(1))[0:2];batch={'input_ids':torch.ones(1,2,dtype=torch.long),'attention_mask':torch.ones(1,2,dtype=torch.long)}
  logits=last_logits(model,batch,candidate,[('B',(2,),2)]);logits.square().sum().backward()
  self.assertIsNotNone(head.weight.grad);self.assertIsNone(model.B.grad);self.assertIsNone(model.base.grad);self.assertTrue(torch.equal(model.B,torch.zeros(2)))
 def test_full_36_state_guard_on_curved_physical_response(self):
  import c17_protocol as protocol
  h,o=self.fixture()
  def physical_rows():
   # Analytical nonlinear synthetic response, NOT GPT-2 evidence.
   weights=[.0005]*16+[.0064]*20;value=float(h.weight.detach())
   ans={}
   for t,sensitivity in zip(protocol.training_specs(),weights):
    drift=sensitivity*value*value;g={g:True for g in protocol.GATES};g['response']=drift<=.002
    ans[t['id']]={'reference_valid':True,'metrics':{'drift':drift,'gates':g,'episode_pass':all(g.values())}}
   return ans
  current=physical_rows();protected=set(protocol.quality(current)['passed_ids']);self.assertEqual(len(protected),16)
  for _ in range(12):
   new,r=attempt_update(h,o,lambda:(h(torch.ones(1,1)).square().mean(),{}),physical_rows,lambda rows:protocol.guard(current,rows))
   if new is not None:current=new
   self.assertTrue(protected.issubset(protocol.quality(current)['passed_ids']))
   if len(protocol.quality(current)['passed_ids'])==36:break
  self.assertEqual(len(protocol.quality(current)['passed_ids']),36)
 def test_rejected_update_restores_nonempty_optimizer_history(self):
  h,o=self.fixture();loss=lambda:(h(torch.ones(1,1)).square().mean(),{})
  attempt_update(h,o,loss,lambda:{},lambda r:{'accepted':True});old=cpu_state(h);mom=copy.deepcopy(o.state_dict())
  attempt_update(h,o,loss,lambda:{},lambda r:{'accepted':False},8)
  self.assertTrue(same_tree(old,cpu_state(h)));self.assertTrue(same_tree(mom,o.state_dict()))
 def test_no_teacher_in_one_raw_proposal(self):
  h=StateWriter(3,2,4);x=torch.ones(3);a=one_raw_proposal(h,x,torch.tensor(0),torch.tensor(.1));self.assertTrue(torch.equal(a,one_raw_proposal(h,x,torch.tensor(0),torch.tensor(.1))))

if __name__=='__main__':unittest.main()
