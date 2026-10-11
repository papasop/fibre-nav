import unittest,types
from runtime_a100 import a100_device
class Fake:
 def __init__(self,available=True,name='NVIDIA A100-SXM4-40GB'):
  self.cuda=types.SimpleNamespace(is_available=lambda:available,get_device_name=lambda i:name)
  self.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True))
  self.precision=None;self.deterministic=None
 def set_float32_matmul_precision(self,v):self.precision=v
 def use_deterministic_algorithms(self,v):self.deterministic=v
 def device(self,v):return v
class Tests(unittest.TestCase):
 def test_cpu_refused(self):
  with self.assertRaises(RuntimeError):a100_device(Fake(False))
 def test_non_a100_refused(self):
  with self.assertRaises(RuntimeError):a100_device(Fake(name='Tesla T4'))
 def test_precision_and_device(self):
  t=Fake();d,n=a100_device(t);self.assertEqual(d,'cuda:0')
  self.assertFalse(t.backends.cuda.matmul.allow_tf32);self.assertFalse(t.backends.cudnn.allow_tf32)
  self.assertEqual(t.precision,'highest');self.assertTrue(t.deterministic)
if __name__=='__main__':unittest.main()
