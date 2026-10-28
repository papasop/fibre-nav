"""Unchanged pinned FP64/native runtime; C168 forbids every backward request."""
from core168 import *
from runtime165 import DirectionRuntime
class FixedProbeRuntime(DirectionRuntime):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);r=clean_json(self.reference_file.read_bytes());r['protocol']=PROTOCOL;dump(self.reference_file,r);self.reference_file_hash=sha(self.reference_file);self.verify(True)
 def coefficient_record(self,sid,p,expected,gradient=False):
  need(gradient is False,'C168 forbids new gradient or backward requests');return super().coefficient_record(sid,p,expected,False)
def factory(*args):return FixedProbeRuntime(*args)
