import copy,inspect,unittest
import numpy as np,torch
from c28_features import *
from c28_math import directions,candidate
from c28_protocol import specs
from c27_inputs import validate_saved_input
class Base(torch.nn.Module):
 def __init__(self):super().__init__();self.layer=torch.nn.Linear(3147,3)
 def forward(self,x,r):return self.layer(x)
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(13);self.x=torch.randn(19,3147);self.ids=['s%d'%i for i in range(19)];self.base=Base().eval()
 def maps(self,arm):return [feature_map(self.x.numpy(),self.ids,arm) for _ in range(4)]
 def test_equal_arm_parameter_counts(self):
  a=FourRouteWriter(self.base,self.maps('pca_control'),3);b=FourRouteWriter(self.base,self.maps('response_aware'),3);self.assertEqual(a.matrices.numel(),b.matrices.numel());self.assertEqual(tuple(a.matrices.shape),(4,3,9))
 def test_zero_is_exact_base(self):
  wr=FourRouteWriter(self.base,self.maps('response_aware'),3);r=torch.arange(19)%4;self.assertTrue(torch.equal(wr(self.x,r),self.base(self.x,r)))
 def test_one_route_only(self):
  wr=FourRouteWriter(self.base,self.maps('response_aware'),3);wr.set_route(2,torch.ones(3,9)*.01);r=torch.arange(19)%4;b=self.base(self.x,r);a=wr(self.x,r);self.assertTrue(torch.equal(a[r!=2],b[r!=2]));self.assertFalse(torch.equal(a[r==2],b[r==2]))
 def test_four_routes_can_change(self):
  wr=FourRouteWriter(self.base,self.maps('response_aware'),3)
  for route in range(4):wr.set_route(route,torch.ones(3,9)*(.01+route*.001))
  r=torch.arange(19)%4;self.assertTrue(all(bool((wr(self.x,r)[r==route]!=self.base(self.x,r)[r==route]).any()) for route in range(4)))
 def test_base_stays_frozen(self):
  old=state_hash(self.base.state_dict());wr=FourRouteWriter(self.base,self.maps('response_aware'),3);wr.set_route(0,torch.randn(3,9));wr(self.x,torch.arange(19)%4);self.assertEqual(old,state_hash(self.base.state_dict()));self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))
 def test_direct_response_coordinates_retained(self):
  cp=feature_map(self.x.numpy(),self.ids,'response_aware');self.assertTrue(torch.equal(cp['basis'][3077:3083,:6],torch.eye(6)));self.assertEqual(int(torch.count_nonzero(cp['basis'][:3077,:6])),0);self.assertEqual(int(torch.count_nonzero(cp['basis'][3083:,:6])),0)
 def test_remaining_components_exclude_response(self):
  cp=feature_map(self.x.numpy(),self.ids,'response_aware');self.assertEqual(int(torch.count_nonzero(cp['basis'][3077:3083,6:])),0)
 def test_fits_training_ids_only(self):
  for arm in ('pca_control','response_aware'):
   cp=feature_map(self.x.numpy(),self.ids,arm);self.assertEqual(cp['fit_ids'],self.ids);self.assertFalse(cp['fit_uses_development'])
 def test_feature_map_not_refit_on_evaluation(self):
  cp=feature_map(self.x.numpy(),self.ids,'response_aware');old=state_hash({k:v for k,v in cp.items() if isinstance(v,torch.Tensor)});m=Map(cp);m(torch.randn(4,3147)*100);self.assertEqual(old,state_hash({k:v for k,v in cp.items() if isinstance(v,torch.Tensor)}))
 def test_constant_features_finite(self):
  cp=feature_map(np.ones((19,3147)),self.ids,'response_aware');self.assertTrue(torch.isfinite(Map(cp)(torch.ones(3,3147))).all());self.assertTrue((cp['scale']>=.001).all())
 def test_nonfinite_matrix_rejected(self):
  wr=FourRouteWriter(self.base,self.maps('response_aware'),3)
  with self.assertRaises(ValueError):wr.set_route(0,torch.full((3,9),float('nan')))
 def test_online_no_geometry_or_autograd(self):
  code=inspect.getsource(FourRouteWriter.forward)+inspect.getsource(Map.forward)
  for forbidden in ('autograd','backward','svd','solve','teacher','correction'):self.assertNotIn(forbidden,code)
 def test_historical_input_without_extras(self):
  f=torch.zeros(3077);f[3072]=1;f[3073]=-1;r={'features':f,'input':torch.cat([f,torch.ones(70)]),'route':torch.tensor(2)};n={'mean':torch.zeros(3077),'scale':torch.ones(3077)};e={'mean':torch.zeros(70),'scale':torch.ones(70)};x,_,a=validate_saved_input(r,n,e,2,'cpu');self.assertTrue(torch.equal(x,r['input']));self.assertFalse(a['raw_extra_replay_checked'])
 def test_fixed_grid_solve_nonvacuous(self):
  J=np.zeros((19,6,12));J[:,:,:6]=np.eye(6);H=np.concatenate([np.linspace(-1,1,19)[:,None]*np.ones((1,8)),np.ones((19,1))],1);E=np.ones((19,6))*.0019;s,bases,info=directions(J,H,E);m,_=candidate(s,bases,specs()[0]);self.assertGreater(float(m.norm()),0.);self.assertTrue(torch.isfinite(m).all())
if __name__=='__main__':unittest.main()
