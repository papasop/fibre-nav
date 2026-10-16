import copy,unittest
import torch
from c27_inputs import validate_saved_input

class Tests(unittest.TestCase):
 def setUp(self):
  self.n={'mean':torch.ones(3077)*.2,'scale':torch.ones(3077)*2}
  self.e={'mean':torch.ones(70)*.3,'scale':torch.ones(70)*3}
  f=torch.randn(3077);f[3072]=1;f[3073]=-1;e=torch.randn(70)
  self.r={'features':f,'extras':e,'input':torch.cat([(f-self.n['mean'])/self.n['scale'],(e-self.e['mean'])/self.e['scale']]),'route':torch.tensor(2)}
 def check(self,r):return validate_saved_input(r,self.n,self.e,2,torch.device('cpu'))
 def test_full_schema_replays_all_blocks(self):
  x,route,a=self.check(self.r);self.assertTrue(torch.equal(x,self.r['input']));self.assertTrue(a['raw_extra_replay_checked'])
 def test_historical_missing_extras_uses_identical_input(self):
  r=copy.deepcopy(self.r);del r['extras'];before=r['input'].clone();x,_,a=self.check(r)
  self.assertTrue(torch.equal(x,before));self.assertFalse(a['raw_extra_replay_checked']);self.assertFalse(a['missing_extras_reconstructed']);self.assertNotIn('extras',r)
 def test_nonfinite_input_rejected_even_without_extras(self):
  r=copy.deepcopy(self.r);del r['extras'];r['input'][-1]=float('nan')
  with self.assertRaises(ValueError):self.check(r)
 def test_present_malformed_extras_not_silently_ignored(self):
  r=copy.deepcopy(self.r);r['extras']=None
  with self.assertRaises(ValueError):self.check(r)
 def test_present_extra_mismatch_rejected(self):
  r=copy.deepcopy(self.r);r['extras'][0]+=1
  with self.assertRaises(ValueError):self.check(r)
 def test_features_mismatch_rejected_without_extras(self):
  r=copy.deepcopy(self.r);del r['extras'];r['input'][0]+=1
  with self.assertRaises(ValueError):self.check(r)
 def test_route_mismatch_rejected(self):
  r=copy.deepcopy(self.r);r['route']=torch.tensor(1)
  with self.assertRaises(ValueError):self.check(r)
 def test_missing_extras_does_not_decode_or_replace_extra_block(self):
  r=copy.deepcopy(self.r);del r['extras'];r['input'][-70:]=12345;before=r['input'].clone();x,_,_=self.check(r)
  self.assertTrue(torch.equal(x,before))
 def test_nonpositive_scale_rejected(self):
  self.e['scale'][0]=0
  with self.assertRaises(ValueError):self.check(self.r)
 def test_float_route_rejected(self):
  self.r['route']=torch.tensor(2.)
  with self.assertRaises(ValueError):self.check(self.r)
if __name__=='__main__':unittest.main()
