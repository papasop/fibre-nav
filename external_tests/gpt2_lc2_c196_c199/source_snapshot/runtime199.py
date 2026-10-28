"""Same pinned physical FP32 model and original references; forward only."""
from core199 import *
from runtime198 import factory as original_factory
def factory(out,ctx,calls):
 rt=original_factory(out,ctx,calls);p=rt.out/'REFERENCE_POLICY.json';r=clean_json(p.read_bytes());r['protocol']=PROTOCOL;r['implementation']='Immutable C198 native runtime, unchanged numerical rules';dump(p,r);rt.reference_file_hash=sha(p);rt.verify(True);return rt
