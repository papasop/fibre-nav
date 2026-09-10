import json,torch
from memory_backend import MemoryChart

class ReadOnlyChart(MemoryChart):
 def __init__(self,root,check,protection_cases):
  super().__init__(root,check)
  record=json.loads((root/'source_protection_reference.json').read_text());self.rule_floors=record['floors'];self.examples=[]
  for c in protection_cases:
   p=self.tok.encode(c['prompt'],add_special_tokens=False);y=self.tok.encode(c['target'],add_special_tokens=False)+[self.tok.eos_token_id]
   self.examples.append((torch.tensor([p+y],device=self.device),len(p),torch.tensor(y,device=self.device)))
 @torch.no_grad()
 def rule_audit(self):
  values=[]
  for ids,n,y in self.examples:
   self.check();ls=self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,n-1:-1].float()
   if ls.shape[0]!=len(y):raise ValueError('Token alignment mismatch')
   own=ls.gather(1,y[:,None]).squeeze(1);other=ls.clone();other.scatter_(1,y[:,None],float('-inf'))
   values.append(float((own-other.max(-1).values).min()))
  return dict(rule_margins=values,rule_floors=self.rule_floors,rule_protection_pass=len(values)==8 and all(v>=f for v,f in zip(values,self.rule_floors)))
