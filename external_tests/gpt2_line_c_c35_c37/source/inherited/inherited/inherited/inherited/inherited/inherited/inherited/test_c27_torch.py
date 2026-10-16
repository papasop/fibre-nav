import copy,io,unittest
import numpy as np
import torch
from c27_features import *
from c27_math import residual_candidates,trial_matrix
from c27_protocol import specs


class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.experts=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,2)) for _ in range(4)])
 def forward(self,x,r):
  v=torch.stack([e(x) for e in self.experts],1);return v[torch.arange(len(x)),r]


class Tests(unittest.TestCase):
 def setUp(self):torch.manual_seed(123);self.base=Tiny().eval();self.x=torch.randn(5,3);self.ids=['s%d'%i for i in range(5)]
 def test_zero_branch_bitwise_original(self):
  fm=fit_map(self.x.numpy(),self.ids,'state_bypass',2);r=ResidualWriter(self.base,fm,'state_bypass',2)
  routes=torch.tensor([0,1,2,3,2]);self.assertTrue(torch.equal(r(self.x,routes),self.base(self.x,routes)))
 def test_branch_only_selected_route_changes(self):
  fm=fit_map(self.x.numpy(),self.ids,'state_bypass',2);r=ResidualWriter(self.base,fm,'state_bypass',2);r.set_matrix(torch.ones(2,3)*.1)
  routes=torch.tensor([0,1,2,3,2]);before=self.base(self.x,routes);after=r(self.x,routes)
  self.assertTrue(torch.equal(before[[0,1,3]],after[[0,1,3]]));self.assertGreater(float((before[[2,4]]-after[[2,4]]).abs().max()),0)
 def test_original_parameters_unchanged(self):
  old={n:p.detach().clone() for n,p in self.base.named_parameters()};fm=fit_map(self.x.numpy(),self.ids,'state_bypass',2);r=ResidualWriter(self.base,fm,'state_bypass',2)
  r.set_matrix(torch.randn(2,3));r(self.x,torch.tensor([2]*5))
  self.assertTrue(all(torch.equal(p,old[n]) for n,p in self.base.named_parameters()));self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))
 def test_feature_map_uses_only_declared_training_ids(self):
  fm=fit_map(self.x.numpy(),self.ids,'state_bypass',2);self.assertEqual(fm['fit_ids'],self.ids);self.assertFalse(fm['fit_uses_development'])
 def test_maps_dont_change_on_development(self):
  fm=fit_map(self.x.numpy(),self.ids,'state_bypass',2);before=copy.deepcopy(fm)
  map_features(torch.ones(10,3)*100,fm)
  self.assertTrue(torch.equal(fm['basis'],before['basis']));self.assertTrue(torch.equal(fm['scale'],before['scale']))
 def test_identical_hidden_not_noise_whitened(self):
  x=np.ones((19,64));x[-1,0]+=5.96e-8;fm=fit_map(x,['x%d'%i for i in range(19)],'hidden_control',8)
  y=map_features(torch.tensor(x,dtype=torch.float32),fm)
  self.assertLess(float((y[:,:-1]-y[:1,:-1]).abs().max()),.001);self.assertTrue(torch.all(fm['scale']>=.001))
 def test_unsaturated_features_retain_state_variation(self):
  x=np.arange(19,dtype=float)[:,None]*np.array([[1.,2.,3.]])
  fm=fit_map(x,['x%d'%i for i in range(19)],'state_bypass',8);y=map_features(torch.tensor(x,dtype=torch.float32),fm)
  self.assertGreater(float((y[:,:-1]-y[:1,:-1]).abs().max()),1.)
 def test_hidden_and_input_arms_same_output_dimension(self):
  with torch.no_grad():h=self.base.experts[2][1](self.base.experts[2][0](self.x))
  a=ResidualWriter(self.base,fit_map(h.numpy(),self.ids,'hidden_control',2),'hidden_control',2)
  b=ResidualWriter(self.base,fit_map(self.x.numpy(),self.ids,'state_bypass',2),'state_bypass',2)
  self.assertEqual(a.matrix.numel(),b.matrix.numel());self.assertEqual(a.matrix.shape,b.matrix.shape)
 def test_combined_module_no_online_backward_or_solver(self):
  import inspect
  source=inspect.getsource(ResidualWriter.forward)+inspect.getsource(ResidualWriter.features)
  for x in ('autograd','backward','svd','solve','teacher'):self.assertNotIn(x,source)
 def test_bad_matrix_nonfinite_rejected(self):
  r=ResidualWriter(self.base,fit_map(self.x.numpy(),self.ids,'state_bypass',2),'state_bypass',2)
  with self.assertRaises(ValueError):r.set_matrix(torch.full((2,3),float('nan')))
 def test_serialisation_preserves_actual_matrix(self):
  m=torch.randn(36864,9);b=io.BytesIO();torch.save({'matrix':m},b);b.seek(0)
  recovered=torch.load(b,map_location='cpu',weights_only=True)['matrix'];self.assertEqual(tensor_hash(m),tensor_hash(recovered))
 def test_interval_residual_direction_nonzero_and_caps(self):
  # 19 states,6 responses,d=6; independent one-hot read geometry and varying features.
  J=np.tile(np.eye(6)[None,:,:],(19,1,1));Phi=np.ones((19,9));Phi[:,1]=np.linspace(-1,1,19)
  E=np.tile(np.array([.004,0,0,0,0,0]),(19,1))
  system,bases,infos=residual_candidates(J,Phi,E)
  matrices=[]
  for spec in specs():
   m,info=trial_matrix(system,bases,spec);self.assertTrue(torch.isfinite(m).all());self.assertLessEqual(float(m.double().norm()),.500001);matrices.append(m)
  self.assertGreater(max(float(m.norm()) for m in matrices),0.)
  best=min(np.max(np.abs(E+SETTINGS['output_scale']*np.einsum('npd,nd->np',J,(m.double().numpy()@Phi.T).T))) for m in matrices)
  self.assertLess(best,.004)
 def test_saturation_audit_known_large_preactivation(self):
  # The actual audit's3147-feature group layout, tiny output dimension is sufficient.
  base=Tiny()
  for i in range(4):
   base.experts[i][0]=torch.nn.Linear(3147,4)
   with torch.no_grad():base.experts[i][0].weight.zero_();base.experts[i][0].bias.fill_(20.)
  cases=('xor_00','xor_01','xor_10','xor_11');x={c:torch.randn(2,3147) for c in cases};ids={c:[c+'a',c+'b'] for c in cases}
  audit=activation_audit(base,x,ids)
  self.assertEqual(audit['xor_10']['hidden_abs_above_0999_fraction'],1.)
  self.assertEqual(audit['xor_10']['tanh_derivative_maximum'],0.)
  self.assertEqual(audit['xor_10']['input_group_contributions_float64_diagnostic']['parameter_sketch']['maximum_abs_contribution'],0.)
 def test_normalisation_padding_components_finite(self):
  cp=fit_map(np.zeros((3,2)),['a','b','c'],'state_bypass',8);y=map_features(torch.zeros(3,2),cp)
  self.assertEqual(tuple(y.shape),(3,9));self.assertTrue(torch.isfinite(y).all())


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
