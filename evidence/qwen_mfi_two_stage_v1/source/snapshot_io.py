"""Validate the complete snapshot before replacing any memory B tensor."""
import hashlib,json
from pathlib import Path

def verify_files(root):
 root=Path(root);manifest=json.loads((root/'source_integrity.json').read_text())
 for name,digest in manifest['files'].items():
  if Path(name).name!=name:raise ValueError('Unsafe file name')
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Hash mismatch: '+name)
 return manifest

def validate_snapshot(modules,data,zero=False):
 import torch
 expected={n+'.'+k for n in modules for k in ('A','B')}
 if set(data)!=expected:raise ValueError('Unexpected snapshot keys')
 for n,m in modules.items():
  for k in ('A','B'):
   v=data[n+'.'+k];dest=getattr(m,k)
   if v.shape!=dest.shape or v.dtype!=dest.dtype or not torch.isfinite(v).all():raise ValueError('Invalid snapshot tensor: '+n+'.'+k)
   if k=='A' and not torch.equal(v.cpu(),dest.detach().cpu()):raise ValueError('Memory A differs from original seeded chart')
   if zero and k=='B' and torch.count_nonzero(v).item()!=0:raise ValueError('Nonzero zero snapshot')

def load_snapshot(backend,data,zero=False):
 import torch
 validate_snapshot(backend.modules,data,zero)
 with torch.no_grad():
  for n,m in backend.modules.items():m.B.copy_(data[n+'.B'].to(m.B.device))
 for n,m in backend.modules.items():
  if not torch.equal(m.B.detach().cpu(),data[n+'.B'].cpu()):raise RuntimeError('Snapshot copy mismatch')
 backend.assert_frozen()
