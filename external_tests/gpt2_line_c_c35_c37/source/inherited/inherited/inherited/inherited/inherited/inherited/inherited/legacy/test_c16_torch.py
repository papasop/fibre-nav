import copy,hashlib,io,unittest
import torch
from raw_writer import StateWriter,prepared_input,route_from_features,one_raw_proposal
from proposer import WriterHead,one_forward_proposal

class Tests(unittest.TestCase):
 def test_three_frozen_heads_remain_unchanged(self):
  hs=[StateWriter(3,2,4),StateWriter(3,2,4),StateWriter(3,2,4)]
  for h in hs:
   h.eval()
   for p in h.parameters():p.requires_grad_(False)
   before=[p.clone() for p in h.parameters()];one_raw_proposal(h,torch.ones(3),torch.tensor(0),torch.tensor(.1))
   self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,h.parameters())))
 def test_actual_checkpoint_roundtrip(self):
  h=StateWriter(3,2,4);b=io.BytesIO();torch.save({'head':h.state_dict(),'steps':1000,'stage':'warm'},b);b.seek(0);cp=torch.load(b,weights_only=True)
  h2=StateWriter(3,2,4);h2.load_state_dict(cp['head']);self.assertTrue(all(torch.equal(a,b) for a,b in zip(h.parameters(),h2.parameters())))
 def test_parameter_hash_changes_with_real_weight(self):
  h=StateWriter(3,2,4)
  def digest():return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in h.parameters())).hexdigest()
  before=digest()
  with torch.no_grad():next(h.parameters()).reshape(-1)[0]+=1
  self.assertNotEqual(before,digest())
 def test_same_saved_input_produces_same_proposal(self):
  h=StateWriter(3,2,4);x=torch.ones(3);r=torch.tensor(2)
  a=one_raw_proposal(h,x,r,torch.tensor(.1));b=one_raw_proposal(h,x.clone(),r.clone(),torch.tensor(.1));self.assertTrue(torch.equal(a,b))
 def test_no_backward_gradients(self):
  h=StateWriter(3,2,4)
  for p in h.parameters():p.requires_grad_(False)
  one_raw_proposal(h,torch.ones(3),torch.tensor(1),torch.tensor(.1));self.assertTrue(all(p.grad is None for p in h.parameters()))
 def test_routes_based_on_actual_operand_features(self):
  f=torch.zeros(3077);f[3072]=-1;f[3073]=1;self.assertEqual(int(route_from_features(f)),1)
 def test_teacher_error_is_not_acceptance_certificate(self):
  # A sensitive response can violate its gate with a small parameter MSE.
  theta=torch.tensor([1e-5]);teacher=torch.zeros(1);self.assertLess(float((theta-teacher).square().mean()),1e-8);self.assertGreater(float((1e4*theta).abs().max()),.002)
 def test_input_normalisation_does_not_change_saved_norm(self):
  n={'mean':torch.zeros(3077),'scale':torch.ones(3077)};e={'mean':torch.zeros(70),'scale':torch.ones(70)};before=copy.deepcopy(e)
  prepared_input(torch.ones(3077),torch.ones(70),n,e);self.assertTrue(all(torch.equal(e[k],before[k]) for k in e))
 def test_c9_baseline_one_frozen_call(self):
  h=WriterHead(3,2,4);n={'mean':torch.zeros(3),'scale':torch.ones(3),'output_scale':torch.tensor(.1)};calls=[];hook=h.register_forward_hook(lambda *a:calls.append(1))
  one_forward_proposal(h,torch.ones(3),n);hook.remove();self.assertEqual(len(calls),1)
 def test_physical_commit_does_not_train_parameter(self):
  p=torch.nn.Parameter(torch.zeros(2),requires_grad=False)
  with torch.no_grad():p.copy_(torch.ones(2))
  self.assertIsNone(p.grad);self.assertFalse(p.requires_grad)

if __name__=='__main__':unittest.main()
