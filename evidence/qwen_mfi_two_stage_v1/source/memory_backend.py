"""Fresh rank4 memory chart around the pinned frozen v4 rule adapter."""
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from qwen_l1 import MODEL,REVISION,ANCHORS,LoRAB
from adapter_io import install_rule
from rescue_protocol import SEED,read_prompt

class MemoryChart:
 def __init__(self,root,check):
  torch.manual_seed(SEED);self.check=check;self.device='cpu'
  self.tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.float32,trust_remote_code=False).eval()
  install_rule(self.model,root);self.modules={}
  for i in range(len(self.model.model.layers)-2,len(self.model.model.layers)):
   for name in ('q_proj','v_proj'):
    attn=self.model.model.layers[i].self_attn;rule=getattr(attn,name)
    memory=LoRAB(rule.base,rank=4);memory.base=rule;setattr(attn,name,memory);self.modules[f'layers.{i}.{name}']=memory
  self.model.to(self.device);self.ps=[m.B for m in self.modules.values()]
  self.bits=[self.single(str(i)) for i in (0,1)];self.selectors=[self.single(x) for x in ('A','B')]
  memory_ids={id(p) for p in self.ps}
  if {id(p) for p in self.model.parameters() if p.requires_grad}!=memory_ids:raise RuntimeError('Unexpected trainable parameters')
  self.model.requires_grad_(False)
  self.frozen=[(t,t._version) for t in list(self.model.parameters())+list(self.model.buffers()) if id(t) not in memory_ids]
  self.initial=self.vector().clone();self.active=0;self.other=None;self.other_value=None;self.other_ref=None
  with torch.no_grad():
   logits=[self.logits(p) for p in ANCHORS]
   self.ref_logp=[x.double().log_softmax(-1).detach() for x in logits]
   self.ref_response=torch.stack([x[self.selectors[0]]-x[self.selectors[1]] for x in logits]).double().detach()
 def assert_frozen(self):
  if any(t._version!=v for t,v in self.frozen):raise RuntimeError('Frozen model/rule/A tensor changed')
 def single(self,s):
  x=self.tok.encode(s,add_special_tokens=False)
  if len(x)!=1:raise ValueError('Requires single token: '+s)
  return x[0]
 def logits(self,prompt):
  self.check();x=self.tok(prompt,return_tensors='pt',add_special_tokens=False).to(self.device)
  return self.model(**x,use_cache=False).logits[0,-1].float()
 def vector(self):return torch.cat([p.detach().flatten() for p in self.ps])
 def gap(self,cell):
  l=self.logits(read_prompt(cell));return l[self.bits[1]]-l[self.bits[0]]
 def sequence(self,cell,value):
  self.check();enc=self.tok(read_prompt(cell),return_tensors='pt',add_special_tokens=False).to(self.device)
  ids=torch.cat([enc['input_ids'],enc['input_ids'].new_tensor([[self.bits[value]]])],1)
  ls=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-2:].float()
  labels=ids.new_tensor([self.bits[value],self.tok.eos_token_id]);return ls,labels
 def seq_margins(self,cell,value):
  ls,y=self.sequence(cell,value);own=ls.gather(1,y[:,None]).squeeze(1);others=ls.clone();others.scatter_(1,y[:,None],float('-inf'))
  return own-others.max(-1).values
 @torch.no_grad()
 def audit(self,value):
  vals=[];kls=[]
  for p,ref in zip(ANCHORS,self.ref_logp):
   l=self.logits(p);vals.append(l[self.selectors[0]]-l[self.selectors[1]]);lp=l.double().log_softmax(-1)
   kls.append((ref.exp()*(ref-lp)).sum())
  target=self.seq_margins(self.active,value).tolist();target.append(float((2*value-1)*self.gap(self.active)))
  protect=self.seq_margins(self.other,self.other_value).tolist() if self.other is not None else []
  gap=float(self.gap(self.other)) if self.other is not None else None
  return dict(response=float((torch.stack(vals).double()-self.ref_response).abs().max()),kl=float(torch.stack(kls).mean()),
   margin=min(target),target_margins=target,protected_drift=abs(gap-self.other_ref) if gap is not None else 0.,
   protected_min_margin=min(protect+[(2*self.other_value-1)*gap]) if protect else 1.,protected_sequence_margins=protect)
 @torch.no_grad()
 def state_audit(self,state):
  saved=(self.active,self.other,self.other_value,self.other_ref)
  self.active=0;self.other=None
  a=self.audit(int(state[0]));a['margin']=min(a['margin'],*self.seq_margins(1,int(state[1])).tolist(),float((2*int(state[1])-1)*self.gap(1)))
  self.active,self.other,self.other_value,self.other_ref=saved
  return a
