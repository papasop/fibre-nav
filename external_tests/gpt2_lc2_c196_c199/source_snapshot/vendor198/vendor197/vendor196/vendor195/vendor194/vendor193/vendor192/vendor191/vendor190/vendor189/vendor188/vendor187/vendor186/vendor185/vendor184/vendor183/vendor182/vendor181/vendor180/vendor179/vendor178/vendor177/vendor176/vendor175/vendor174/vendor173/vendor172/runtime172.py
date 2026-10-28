"""Same pinned native/FP64 functions; only one sealed anchor AD batch."""
from core172 import *
from runtime170 import Relinearised
class FinalHeadProbes(Relinearised):
 def __init__(self,*a,**k):
  super().__init__(*a,**k);r=clean_json(self.reference_file.read_bytes());r['protocol']=PROTOCOL;dump(self.reference_file,r);self.reference_file_hash=sha(self.reference_file);self.verify(True)
 def coefficient_record(self,sid,p,expected,gradient=False):
  need(not gradient or (self.batch is not None and self.batch['folder'].name=='anchor'),'AD only at registered C170 final anchor; probes/repeat forbid backward');return super().coefficient_record(sid,p,expected,gradient)
def factory(*a):return FinalHeadProbes(*a)
