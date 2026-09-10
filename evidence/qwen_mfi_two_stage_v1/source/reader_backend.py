"""Add a separately serialized reader branch; source rule and memory remain frozen."""
import torch
import torch.nn.functional as F
from qwen_l1 import LoRAB,ANCHORS
from reader_protocol import CONFIG,SEED

class ReaderBranch(torch.nn.Module):
 def __init__(self,base,linear):
  super().__init__();self.base=base
  self.register_buffer('A',torch.randn(CONFIG['rank'],linear.in_features,device=linear.weight.device,dtype=linear.weight.dtype)/linear.in_features**.5)
  self.B=torch.nn.Parameter(torch.zeros(linear.out_features,CONFIG['rank'],device=linear.weight.device,dtype=linear.weight.dtype));self.scale=CONFIG['scale']
 def forward(self,x):return self.base(x)+(x@self.A.T@self.B.T)*self.scale

def attach_reader(b):
 torch.manual_seed(SEED);modules={}
 for name,memory in b.modules.items():
  _,layer,proj=name.split('.');attn=b.model.model.layers[int(layer)].self_attn
  r=ReaderBranch(memory,memory.base.base);setattr(attn,proj,r);modules[name]=r
 ps=[r.B for r in modules.values()]
 b.model.requires_grad_(False)
 return modules,ps

def completion_ce(b,prompt,target):
 enc=b.tok.encode(prompt,add_special_tokens=False);answer=b.tok.encode(target,add_special_tokens=False)+[b.tok.eos_token_id]
 x=torch.tensor([enc+answer],device=b.device);labels=torch.tensor(answer,device=b.device)
 logits=b.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,len(enc)-1:-1].float()
 if logits.shape[0]!=len(answer):raise RuntimeError('Completion alignment mismatch')
 return F.cross_entropy(logits,labels)

def reader_tensors(modules):
 return {name+'.'+key:getattr(m,key).detach().cpu().contiguous() for name,m in modules.items() for key in ('A','B')}
