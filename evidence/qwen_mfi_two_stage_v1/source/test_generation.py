import unittest
from types import SimpleNamespace
from generation_audit import locked_generate,content_parse
class Model:
 def _get_logits_processor(self,generation_config):return []
 def generate(self,**kw):
  cfg=SimpleNamespace(**kw);cfg.to_dict=lambda:{k:v for k,v in kw.items() if k not in ('input_ids','attention_mask')}
  self._get_logits_processor(cfg)
  return [1]
class BadModel(Model):
 def generate(self,**kw):
  kw['do_sample']=True
  return super().generate(**kw)
class Tests(unittest.TestCase):
 def test_capture_and_restore(self):
  m=Model();tok=SimpleNamespace(eos_token_id=2,bos_token_id=1)
  result,cfg=locked_generate(m,tok,None,None)
  self.assertFalse(cfg['do_sample']);self.assertEqual(result,[1]);self.assertNotIn('_get_logits_processor',m.__dict__)
 def test_reject_override(self):
  m=BadModel();tok=SimpleNamespace(eos_token_id=2,bos_token_id=1)
  with self.assertRaises(RuntimeError):locked_generate(m,tok,None,None)
  self.assertNotIn('_get_logits_processor',m.__dict__)
 def test_content(self):
  self.assertEqual(content_parse('Reasoning text.\nFINAL: 1'),1)
  for s in ('A=1, B=0; Answer: 1','Output: 1\nExplanation: 0','0 or 1'):
   self.assertIsNone(content_parse(s))
  self.assertIsNone(content_parse('1',True))

class TensorLike:
 def __init__(self,value):self.value=value
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return self.value
class TensorConfigModel(Model):
 def _get_logits_processor(self,generation_config):return []
 def generate(self,**kw):
  cfg=SimpleNamespace(**kw)
  cfg.to_dict=lambda:dict(do_sample=kw['do_sample'],_eos_token_tensor=TensorLike([151645,151643]),nested={'scalar':TensorLike(3),'tuple':(TensorLike([1]),)})
  self._get_logits_processor(cfg)
  return [1]
class SerializationTests(unittest.TestCase):
 def test_runtime_config_roundtrip(self):
  import json
  model=TensorConfigModel();tok=SimpleNamespace(eos_token_id=2,bos_token_id=1)
  _,cfg=locked_generate(model,tok,None,None)
  row={'raw':'1','effective_configs':[cfg,cfg]}
  result=json.loads(json.dumps([row],allow_nan=False))
  self.assertEqual(result[0]['effective_configs'][0]['_eos_token_tensor'],[151645,151643])
  self.assertEqual(cfg['nested']['scalar'],3)
  self.assertEqual(cfg['nested']['tuple'],[[1]])
 def test_unknown_not_silently_stringified(self):
  from generation_audit import json_safe
  with self.assertRaises(TypeError):json_safe(object())
 def test_actual_torch_when_available(self):
  try:import torch
  except ImportError:self.skipTest('torch not installed locally; this test runs in Colab before audit')
  import json
  from generation_audit import json_safe
  self.assertEqual(json.loads(json.dumps(json_safe({'t':torch.tensor([1,2]),'s':torch.tensor(3)}))),{'t':[1,2],'s':3})

class FinalAnswerTests(unittest.TestCase):
 def test_ambiguous_and_truncated(self):
  for s in ('FINAL: 0\nFINAL: 1','FINAL: 1\nextra','Answer: 1','FINAL: 0 or FINAL: 1'):
   self.assertIsNone(content_parse(s))
  self.assertIsNone(content_parse('FINAL: 1',True))
  self.assertEqual(content_parse('FINAL: 0'),0)
