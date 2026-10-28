"""Legacy synthetic-only point builders for full archival fixtures without Torch.

Only trusted delivered source definitions are adapted, never archived source.
These are manufactured NumPy points and no model/AD evidence is claimed.
Production workers, audit readers, source files and source pins are unchanged.
"""
import inspect

def numpy_fixture_freeze(module):
 source=inspect.getsource(module.freeze)
 source=source.replace(' from map159 import torch_constants,torch_map\n','').replace(' import torch\n','').replace(" torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)\n",'').replace(';tc=torch_constants(c)','')
 source=source.replace("   with torch.no_grad():point,pp,ee=torch_map(tc,sid,torch.tensor(float(alpha),dtype=torch.float64))\n   point=point.numpy().copy();", "   point=want.copy();")
 source=source.replace("   with torch.no_grad():pt,pp,e=torch_map(tc,sid,torch.tensor(float(a),dtype=torch.float64))\n   pt=pt.numpy().copy();", "   pt=want.copy();")
 if 'torch.' in source or 'torch_map(' in source:raise ValueError('Legacy synthetic adapter must not load Torch')
 namespace=dict(module.__dict__);exec(compile(source,'<trusted-synthetic-numpy-freezer>','exec'),namespace);return namespace['freeze']
