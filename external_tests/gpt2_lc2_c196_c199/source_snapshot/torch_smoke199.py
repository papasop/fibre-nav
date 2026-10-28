"""Actual tiny CPU Torch tests for reused native wrapper; not GPT-2/HF."""
import unittest
from unittest import mock
from core199 import *
import torch_smoke198 as old
from runtime199 import factory
class Checks(old.Checks):
 def setup_runtime(self):
  with mock.patch.object(old,'factory',factory):return super().setup_runtime()
 def test_current_protocol_policy_and_same_reference(self):
  rt,c,t=self.setup_runtime();r=clean_json((rt.out/'REFERENCE_POLICY.json').read_bytes());self.assertEqual(r['protocol'],PROTOCOL);self.assertFalse(r['reference_recentered']);self.assertEqual(c['model_forwards'],1);rt.close()
if __name__=='__main__':unittest.main(verbosity=2)
