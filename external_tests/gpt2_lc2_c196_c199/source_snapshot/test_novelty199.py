"""Actual ZIP/JSONL novelty inventory with source exclusion and collision blocking."""
import tempfile,unittest,zipfile,copy
from core199 import *
from fixture199 import fixture,history
from novelty199 import inventory
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);_,_,cls.ctx=fixture(cls.root/'fixture');cls.paths={}
  for k in INPUT_KEYS:
   p=cls.root/(k+'.zip')
   with zipfile.ZipFile(p,'w') as z:
    z.writestr('measurements.json',json.dumps(dict(state_id='archived_'+k,seed=123,point_sha256='1'*64)));z.writestr('records.jsonl',json.dumps(dict(state_id='line_'+k,seed=456,theta_sha256='2'*64))+'\n');z.writestr('source/code.json',json.dumps(dict(state_id=registry()[0]['state_id'],seed=registry()[0]['seed'])));z.writestr('unused.npz',b'invalid tensor bytes should never load')
   cls.paths[k]=p
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_all44_JSON_and_JSONL_metadata_scanned_without_arrays_or_source(self):
  h=inventory(self.paths,self.ctx,self.root/'inventory');self.assertTrue({'1'*64,'2'*64}<=h['hashes']);self.assertIn('line_C198',h['state_ids']);self.assertTrue({123,456}<=h['seeds']);self.assertNotIn(registry()[0]['state_id'],h['state_ids']);self.assertNotIn(registry()[0]['seed'],h['seeds']);p,b=generate(self.ctx['data']['d'],h);self.assertEqual(len(p),48)
 def test_bound_record_seed_collision_blocks_before_model(self):
  h=inventory(self.paths,self.ctx,self.root/'collision');h['seeds'].add(registry()[0]['seed'])
  with self.assertRaises(ValueError):generate(self.ctx['data']['d'],h)
 def test_inherited_coordinate_hashes_are_retained(self):
  ctx=copy.deepcopy(self.ctx);ctx['data']['d']['historical_hashes']=['3'*64];h=inventory(self.paths,ctx,self.root/'inherited');self.assertIn('3'*64,h['hashes'])
if __name__=='__main__':unittest.main(verbosity=2)
