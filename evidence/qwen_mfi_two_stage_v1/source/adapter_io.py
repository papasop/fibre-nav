import json,hashlib
import torch
from safetensors.torch import load_file
from qwen_l1 import LoRAB

def install_rule(model,root):
 manifest=json.loads((root/'manifest.json').read_text());path=root/'rule_adapter_final.safetensors'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['adapter_sha256']:raise ValueError('Rule adapter hash mismatch')
 data=load_file(str(path));locations=[]
 for i in range(len(model.model.layers)-2,len(model.model.layers)):
  attn=model.model.layers[i].self_attn
  for name in ('q_proj','v_proj'):locations.append((f'layers.{i}.{name}',attn,name))
 if set(data)!={n+'.'+k for n,_,_ in locations for k in ('A','B')}:raise ValueError('Unexpected rule adapter keys')
 for n,attn,name in locations:
  wrapper=LoRAB(getattr(attn,name),rank=8)
  with torch.no_grad():
   for k in ('A','B'):
    v=data[n+'.'+k];dest=getattr(wrapper,k)
    if v.shape!=dest.shape or v.dtype!=dest.dtype or not torch.isfinite(v).all():raise ValueError('Invalid rule tensor')
    dest.copy_(v)
  wrapper.requires_grad_(False);setattr(attn,name,wrapper)
 model.requires_grad_(False);model.eval()
 return manifest

