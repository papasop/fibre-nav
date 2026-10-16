import copy,io,unittest
import numpy as np
import torch
from raw_writer import StateWriter,one_raw_proposal
from output_layer import layer_matrix,set_layer_matrix,hidden_augmented
from functional_writer import last_logits
from joint_calibration import prepare_system,solve_direction,response_prediction

class Tests(unittest.TestCase):
 def test_only_selected_output_layer_changes(self):
  h=StateWriter(3,2,4);old={k:v.clone() for k,v in h.state_dict().items()};layer=layer_matrix(h,0);set_layer_matrix(h,0,layer+.01)
  changed=[k for k,v in h.state_dict().items() if not torch.equal(v,old[k])]
  self.assertEqual(changed,['experts.0.2.weight','experts.0.2.bias'])
 def test_hidden_version_check_excludes_expert_zero_output(self):
  h=StateWriter(3,2,4);versions={n:p._version for n,p in h.named_parameters() if n.split('.')[2]=='0'};set_layer_matrix(h,0,layer_matrix(h,0)+.01)
  self.assertTrue(all(p._version==versions[n] for n,p in h.named_parameters() if n in versions))
 def test_bias_is_final_augmented_coordinate(self):
  h=StateWriter(3,2,4);x=torch.ones(3);H=hidden_augmented(h,x,2);self.assertEqual(float(H[-1]),1.);torch.testing.assert_close(layer_matrix(h,2)@H,h.experts[2](x))
 def test_output_layer_derivative_chain_rule(self):
  h=StateWriter(3,2,4);x=torch.tensor([.1,.2,.3]);H=hidden_augmented(h,x,1);U=layer_matrix(h,1).double().requires_grad_();scale=.07
  J=torch.tensor([[.3,-.7],[1.,.2]],dtype=torch.float64);response=J@(scale*(U@H.double()));g=torch.autograd.grad(response[0],U)[0]
  torch.testing.assert_close(g,scale*J[0].unsqueeze(1)*H.double().unsqueeze(0))
 def test_ridge_calibration_of_existing_linear_head(self):
  h=StateWriter(3,2,4);r=np.random.default_rng(18);X=torch.tensor(r.normal(size=(3,3)),dtype=torch.float32)
  H=np.stack([hidden_augmented(h,x,0).double().numpy() for x in X]);J=np.tile(np.eye(2),(3,1,1));scale=.07
  old=layer_matrix(h,0);E=np.stack([one_raw_proposal(h,x,torch.tensor(0),torch.tensor(scale)).double().numpy() for x in X]);D,i=solve_direction(prepare_system(J,H,E,scale),1e-8,1e9,1e9)
  set_layer_matrix(h,0,(old.double()+torch.from_numpy(D)).float());after=np.stack([one_raw_proposal(h,x,torch.tensor(0),torch.tensor(scale)).numpy() for x in X])
  self.assertLess(np.max(np.abs(after)),1e-5)
 def test_restore_layer_rolls_back_rejection(self):
  h=StateWriter(3,2,4);old=copy.deepcopy(h.state_dict());U=layer_matrix(h,1);set_layer_matrix(h,1,U+.1);set_layer_matrix(h,1,U)
  self.assertTrue(all(torch.equal(old[k],v) for k,v in h.state_dict().items()))
 def test_checkpoint_restores_only_output_layers(self):
  h=StateWriter(3,2,4);set_layer_matrix(h,2,layer_matrix(h,2)+.01);cp={'head':h.state_dict(),'rounds':3,'branches':{'xor_10':{'trust':.25}},'optimiser':None}
  b=io.BytesIO();torch.save(cp,b);b.seek(0);v=torch.load(b,weights_only=True);h2=StateWriter(3,2,4);h2.load_state_dict(v['head']);self.assertEqual(v['rounds'],3);self.assertEqual(v['branches']['xor_10']['trust'],.25)
  self.assertTrue(all(torch.equal(a,b) for a,b in zip(h.parameters(),h2.parameters())))
 def test_one_raw_call_without_gradient(self):
  h=StateWriter(3,2,4)
  for p in h.parameters():p.requires_grad_(False)
  calls=[];hook=h.register_forward_hook(lambda *a:calls.append(1));one_raw_proposal(h,torch.ones(3),torch.tensor(0),torch.tensor(.07));hook.remove();self.assertEqual(len(calls),1);self.assertTrue(all(p.grad is None for p in h.parameters()))
 def test_bad_matrix_rejected_without_update(self):
  h=StateWriter(3,2,4);U=layer_matrix(h,0)
  with self.assertRaises(ValueError):set_layer_matrix(h,0,torch.ones(9,9))
  self.assertTrue(torch.equal(U,layer_matrix(h,0)))
 def test_candidate_J_does_not_train_frozen_model(self):
  class Tiny(torch.nn.Module):
   def __init__(self):super().__init__();self.B=torch.nn.Parameter(torch.zeros(2),requires_grad=False)
   def forward(self,input_ids,attention_mask,use_cache=False):
    class Out:pass
    r=Out();r.logits=self.B.view(1,1,2).expand(input_ids.shape[0],input_ids.shape[1],2);return r
  model=Tiny();candidate=torch.ones(2,requires_grad=True);batch={'input_ids':torch.ones(1,2,dtype=torch.long),'attention_mask':torch.ones(1,2,dtype=torch.long)}
  R=last_logits(model,batch,candidate,[('B',(2,),2)]).mean(0);g=torch.autograd.grad(R[0],candidate)[0];torch.testing.assert_close(g,torch.tensor([1.,0.]));self.assertIsNone(model.B.grad);self.assertFalse(model.B.requires_grad);self.assertTrue(torch.equal(model.B,torch.zeros(2)))
 def test_complete_joint_pipeline_on_curved_36_state_fixture(self):
  import c18_protocol as protocol
  # Synthetic nonlinear response and content coordinate; this is not GPT-2 evidence.
  h=StateWriter(1,2,2)
  with torch.no_grad():
   for e in h.experts:
    e[0].weight.copy_(torch.tensor([[1.],[0.]]));e[0].bias.zero_()
  for route in range(4):set_layer_matrix(h,route,torch.tensor([[.15,0.,0.],[0.,0.,1.]]))
  for p in h.parameters():p.requires_grad_(False)
  values=[.1,.2,.25,.3,.5,.6,.7,.8,1.];data=[]
  for case in protocol.CASES:
   for i,t in enumerate(x for x in protocol.training_specs() if x['case']==case):data.append((t['id'],case,torch.tensor([values[i]])))
  def measure():
   result={}
   for sid,case,x in data:
    theta=one_raw_proposal(h,x,torch.tensor(protocol.CASES.index(case)),torch.tensor(1.));drift=float(theta[0].square());content=abs(float(theta[1])-1.)<1e-6
    gates={g:content for g in protocol.GATES};gates['response']=drift<=.002
    result[sid]={'reference_valid':True,'metrics':{'drift':drift,'gates':gates,'episode_pass':all(gates.values())}}
   return result
  current=measure();protected=set(protocol.quality(current)['passed_ids']);self.assertEqual(len(protected),16)
  for _ in range(2):
   for route,case in enumerate(protocol.CASES):
    group=[r for r in data if r[1]==case];H=[];J=[];E=[]
    for sid,c,x in group:
     theta=one_raw_proposal(h,x,torch.tensor(route),torch.tensor(1.));H.append(hidden_augmented(h,x,route).double().numpy());J.append([[2*float(theta[0]),0.]]);E.append([float(theta[0].square())])
    D,info=solve_direction(prepare_system(np.array(J),np.array(H),np.array(E),1.),1e-8,1e9,1e9)
    old=layer_matrix(h,route);set_layer_matrix(h,route,(old.double()+torch.from_numpy(D)).float());candidate=measure()
    self.assertTrue(protocol.guard(current,candidate)['accepted']);current=candidate
    self.assertTrue(protected.issubset(protocol.quality(current)['passed_ids']))
  self.assertEqual(len(protocol.quality(current)['passed_ids']),36)
 def test_actual_float32_layer_step_records_rounding(self):
  U=torch.ones((2,3),dtype=torch.float32);D=torch.full((2,3),1e-12,dtype=torch.float64);trial=(U.double()+D).float();self.assertTrue(torch.equal(trial,U));self.assertEqual(float((trial.double()-U.double()).norm()),0.)
 def test_branch_update_does_not_change_other_routes(self):
  h=StateWriter(3,2,4);x=torch.ones(3);before=one_raw_proposal(h,x,torch.tensor(3),torch.tensor(.07));set_layer_matrix(h,1,layer_matrix(h,1)+.01)
  self.assertTrue(torch.equal(before,one_raw_proposal(h,x,torch.tensor(3),torch.tensor(.07))))

if __name__=='__main__':unittest.main()
