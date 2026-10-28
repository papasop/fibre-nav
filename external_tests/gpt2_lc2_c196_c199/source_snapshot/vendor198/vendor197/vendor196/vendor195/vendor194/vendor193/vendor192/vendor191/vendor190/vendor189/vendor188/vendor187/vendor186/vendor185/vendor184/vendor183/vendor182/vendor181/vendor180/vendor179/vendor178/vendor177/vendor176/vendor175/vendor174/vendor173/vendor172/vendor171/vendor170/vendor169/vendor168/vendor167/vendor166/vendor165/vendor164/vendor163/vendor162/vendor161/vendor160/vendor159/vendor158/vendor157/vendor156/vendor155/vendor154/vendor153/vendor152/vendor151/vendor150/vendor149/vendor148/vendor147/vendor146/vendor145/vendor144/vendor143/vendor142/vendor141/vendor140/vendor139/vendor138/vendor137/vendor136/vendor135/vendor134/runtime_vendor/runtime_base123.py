"""Fresh pinned GPT-2/zero-B chart; all fitting geometry is training-only."""
from pathlib import Path
import hashlib,os,random,time
import numpy as np
import torch
from lc2_protocol import *
from lc2_math import inverse,limited

def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pt.tmp');torch.save(value,t);os.replace(t,p)
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def head_hash(head):
 h=hashlib.sha256()
 for k in sorted(head):
  t=head[k];h.update(k.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.contiguous().numpy().tobytes())
 return h.hexdigest()

class Runtime:
 def __init__(self,out):
  global w
  self.out=Path(out)
  self.online=False
  import writer_core as w
  from run_l5_audit import prompts
  self.device=torch.device('cpu');torch.set_num_threads(2)
  torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
  w.set_seed(SETTINGS['model_seed']);cfg={'model_name':SETTINGS['model_name'],'model_revision':SETTINGS['revision'],'lora_rank':8,'lora_alpha':16}
  print('C123_CPU Loading pinned GPT-2 on CPU float32; fresh zero LoRA-B; no GPU requirement.',flush=True)
  self.tok,self.model,self.params=w.make_model(cfg,self.device);self.model.config.use_cache=False
  require(all(p.device.type=='cpu' and p.dtype==torch.float32 for p in self.model.parameters()),'CPU float32 placement differs')
  require(sum(p.numel() for p in self.params)==36864 and w.param_flat(self.params).abs().max().item()==0,'Fresh declared chart differs')
  require(getattr(self.model.config,'_commit_hash',None)==SETTINGS['revision'],'Model revision differs')
  for p in self.model.parameters():p.requires_grad_(False);p.grad=None
  self.zero=torch.zeros(36864,dtype=torch.float32)
  self.versions={n:p._version for n,p in self.model.named_parameters() if 'lora_B.default.' not in n}
  self.readids=[x[1] for x in w.single_token_pair(self.tok)]
  rng=random.Random(82931);self.anchors=rng.sample(w.ANCHORS,12);self.responseids=rng.sample(list(range(100,min(50000,self.tok.vocab_size))),6)
  _,self.exact,_,self.dev=prompts();self.read_prompts=self.exact+self.dev
  with torch.no_grad():self.logits0=w.next_logits(self.model,self.tok,self.anchors,self.device).detach();self.response0=self.logits0[:,self.responseids].mean(0)
  self.sketch=torch.from_numpy((np.random.default_rng(SETTINGS['state_sketch_seed']).normal(size=(36864,64))/np.sqrt(36864)).astype(np.float32))
  ah=hashlib.sha256()
  for n,p in self.model.named_parameters():
   if 'lora_A' in n:ah.update(p.detach().cpu().numpy().tobytes())
  pins={'revision':SETTINGS['revision'],'seed':82931,'lora_A_sha256':ah.hexdigest(),'chart_shapes':[list(p.shape) for p in self.params],
        'symbols':w.single_token_pair(self.tok),'anchors':self.anchors,'response_ids':self.responseids,'sketch_sha256':tensor_hash(self.sketch),
        'torch':str(torch.__version__),'gpu':'CPU','device':'cpu','dtype':'float32','threads':torch.get_num_threads()}
  p=self.out/'MODEL_PINS.json'
  if p.exists():require(read(p)==json.loads(json.dumps(pins)),'Resume model/A/runtime pins differ')
  else:dump(p,pins)
 def invariant(self):
  require(all(p._version==self.versions[n] for n,p in self.model.named_parameters() if n in self.versions),'Frozen backbone/A changed')
  require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),'Unexpected active model gradient')
 def assign(self,theta):
  theta=torch.as_tensor(theta,dtype=torch.float32,device=self.device)
  require(theta.shape==(36864,) and bool(torch.isfinite(theta).all()),'Nonfinite/wrong parameter proposal')
  w.assign_flat(self.params,theta)
 def outputs(self):
  al=w.next_logits(self.model,self.tok,self.anchors,self.device)
  rl=w.next_logits(self.model,self.tok,self.read_prompts,self.device)[:,self.readids]
  return torch.cat([al[:,self.responseids].mean(0)-self.response0,rl[:,1]-rl[:,0]]),al
 @torch.no_grad()
 def metrics(self,theta,case,write=True):
  self.assign(theta);o,al=self.outputs();a=o.detach().cpu().double().numpy();truth=target(case) if write else bits(case)
  observed=[int(v) for v in (a[6:10]>=0)];m=(2*np.array(truth*4)-1)*a[6:]
  kvals=(self.logits0.softmax(-1)*(self.logits0.log_softmax(-1)-al.log_softmax(-1))).sum(-1)
  kl=max(0.,float(kvals.mean().cpu()));R=float(np.max(np.abs(a[:6])))
  gate={'bits_match':observed==truth,'response':R<=.002,'KL':kl<=.005,'exact_read_margin':float(min(m[:4]))>=.05,'development_read_margin':float(min(m[4:]))>=.05}
  if write:gate.update(output_flip=observed[3]!=bits(case)[3],unselected_preserved=observed[:3]==bits(case)[:3])
  self.invariant()
  return {'observed':a.tolist(),'bits':observed,'response_drift':R,'KL':kl,'exact_margins':m[:4].tolist(),'development_margins':m[4:].tolist(),
          'per_anchor_KL':kvals.cpu().tolist(),'gates':{k:bool(v) for k,v in gate.items()},'pass':bool(all(gate.values())),
          'theta_sha256':tensor_hash(torch.as_tensor(theta,dtype=torch.float32))}
 def jacobian(self,theta):
  require(not self.online,'Online Jacobian forbidden')
  self.assign(theta)
  for p in self.params:p.requires_grad_(True)
  try:
   with torch.enable_grad():
    o,_=self.outputs();rows=[]
    for i in range(22):
     gs=torch.autograd.grad(o[i],self.params,retain_graph=i<21)
     rows.append(torch.cat([g.reshape(-1) for g in gs]).detach().cpu().double())
   J=torch.stack(rows).numpy();require(np.isfinite(J).all(),'Nonfinite offline Jacobian')
  finally:
   for p in self.params:p.requires_grad_(False);p.grad=None
  self.invariant();return J
 @torch.no_grad()
 def features(self,theta,desired):
  self.assign(theta);batch=self.tok(self.exact,return_tensors='pt',padding=True,truncation=True,max_length=48).to(self.device)
  r=self.model(**batch,use_cache=False,output_hidden_states=True);last=batch['attention_mask'].sum(1)-1;rows=torch.arange(4,device=self.device)
  hidden=r.hidden_states[-1][rows,last].reshape(-1).cpu();margin=(r.logits[rows,last][:,self.readids][:,1]-r.logits[rows,last][:,self.readids][:,0]).cpu()
  o,_=self.outputs();state=torch.as_tensor(theta,dtype=torch.float32).cpu();extra=torch.cat([o[:6].cpu()/.002,state@self.sketch])
  f=torch.cat([hidden,margin,torch.tensor([desired],dtype=torch.float32),extra]);require(f.shape==(3147,) and bool(torch.isfinite(f).all()),'Feature schema differs')
  self.invariant();return f.double().numpy(),o.cpu().double().numpy()
 def teacher(self,initial,case,write,folder,backup=''):
  """Offline Newton reference/preparation solver. Never called for online proposals."""
  require(not self.online,'Online teacher/repair forbidden')
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=folder/'SOLVER.pt';initial=torch.as_tensor(initial,dtype=torch.float32).cpu()
  ih=tensor_hash(initial)
  if p.exists():
   cp=load(p);require(cp['initial_sha256']==ih and cp['case']==case and cp['write']==write,'Solver resume input differs')
  else:cp={'initial_sha256':ih,'case':case,'write':write,'theta':initial,'steps':0,'history':[],'done':False}
  theta=cp['theta'].numpy().astype(np.float64);truth=target(case) if write else bits(case);desired=np.r_[np.zeros(6),(2*np.array(truth*4)-1)*SETTINGS['teacher_target_margin']]
  scale=np.r_[np.full(6,.002),np.ones(16)]
  def score(m):
   o=np.asarray(m['observed']);sg=(2*np.array(truth*4)-1)*o[6:]
   return float(np.mean((o[:6]/.002)**2)+np.mean((np.maximum(SETTINGS['teacher_safety_margin']-sg,0)/.2)**2)+max(m['KL']/.0045-1,0)**2)
  if not cp['done']:
   for k in range(cp['steps'],SETTINGS['teacher_max_steps']):
    m=self.metrics(theta,case,write)
    safe=m['pass'] and m['response_drift']<=SETTINGS['teacher_safety_response'] and min(m['exact_margins']+m['development_margins'])>=SETTINGS['teacher_safety_margin'] and m['KL']<=.0045
    if safe:cp['done']=True;cp['stop']='SAFE_GATES';break
    J=self.jacobian(theta);B,info=inverse(J/scale[:,None],SETTINGS['teacher_ridge']);d=limited(B@((desired-np.array(m['observed']))/scale),SETTINGS['teacher_step_radius'])
    before=score(m);accepted=False
    for j in range(SETTINGS['teacher_backtracks']):
     cand=(theta+(2.**(-j))*d).astype(np.float32).astype(np.float64);cm=self.metrics(cand,case,write)
     if score(cm)<before-1e-12:theta=cand;accepted=True;break
    cp.update(theta=torch.from_numpy(theta.astype(np.float32)),steps=k+1)
    cp['history'].append({'step':k+1,'accepted':accepted,'response_before':m['response_drift'],'score_before':before,'step_norm':float(np.linalg.norm(d)),'numeric_rank':info['numeric_rank']})
    save(p,cp)
    dump(folder/'PROGRESS.json',{'steps':cp['steps'],'done':False})
    if (k+1)%5==0:print('C38 OFFLINE SOLVER',case,'write' if write else 'prepare',folder.name,k+1,'R',m['response_drift'],'accepted',accepted,flush=True)
    if (k+1)%20==0:snapshot(self.out,backup)
    if not accepted:cp['done']=True;cp['stop']='NO_DECLARED_STEP_IMPROVEMENT';break
   else:cp['done']=True;cp['stop']='MAX_STEPS'
  cp['theta']=torch.from_numpy(theta.astype(np.float32));cp['done']=True;save(p,cp);dump(folder/'PROGRESS.json',{'steps':cp['steps'],'done':True})
  metric=self.metrics(cp['theta'],case,write);dump(folder/'FINAL.json',{'case':case,'write':write,'steps':cp['steps'],'stop':cp.get('stop'),'metrics':metric})
  return cp['theta'],metric
