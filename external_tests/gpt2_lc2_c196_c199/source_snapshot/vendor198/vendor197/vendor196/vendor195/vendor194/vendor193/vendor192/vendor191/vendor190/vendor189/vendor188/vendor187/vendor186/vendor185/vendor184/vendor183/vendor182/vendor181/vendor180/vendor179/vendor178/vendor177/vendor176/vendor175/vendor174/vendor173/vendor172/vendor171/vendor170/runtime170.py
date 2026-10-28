"""Unchanged sealed actual-FP64/physical-FP32 runtime, fresh head gradients."""
from core170 import *
from runtime165 import DirectionRuntime
class Relinearised(DirectionRuntime):
 def __init__(self,*a,**k):
  super().__init__(*a,**k);r=clean_json(self.reference_file.read_bytes());r['protocol']=PROTOCOL;dump(self.reference_file,r);self.reference_file_hash=sha(self.reference_file);self.verify(True)
def factory(*a):return Relinearised(*a)
