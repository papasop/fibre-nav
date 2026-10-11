import json,random
from pathlib import Path
import torch
import writer_core as w
from controller_data import *
from b5_protocol import ACTIONS, dispatch

def train_controller(cfg,device,out,save_tensor,dump,pack,drive,select_action):
 local=dict(cfg);local.update(seed=CONTROLLER_SEED,lora_rank=16,lora_alpha=32)
 w.set_seed(CONTROLLER_SEED)
 print('V2 CONTROLLER_LOADING: separate GPT-2; frozen base/A; train controller B only',flush=True)
 tok,model,params=w.make_model(local,device);model.config.use_cache=False
 rows=training_samples();dump(out/'controller_training_data.json',rows)
 dump(out/'controller_split.json',{'train_tuples':TRAIN_TUPLES,'heldout_tuples':HELDOUT,'train_steps':TRAIN_STEPS,'seed':CONTROLLER_SEED,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE})
 encoded=[]
 for r in rows:
  prefix=tok.encode(r['prompt'],add_special_tokens=False);action=tok.encode(' '+r['action']+'\n',add_special_tokens=False)
  encoded.append((prefix+action,[-100]*len(prefix)+action))
 if max(len(x) for x,y in encoded)>model.config.n_positions:raise RuntimeError('Controller training context overflow')
 cp=out/'controller_training.pt';opt=torch.optim.AdamW(params,lr=LEARNING_RATE,weight_decay=0)
 start=0
 if cp.exists():
  state=torch.load(cp,map_location=device,weights_only=True)
  w.assign_flat(params,state['theta']);opt.load_state_dict(state['optimizer']);start=state['steps']
 frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 for step in range(start,TRAIN_STEPS):
  batch=random.Random(CONTROLLER_SEED+step).sample(encoded,BATCH_SIZE);length=max(len(x) for x,y in batch)
  x=torch.tensor([x+[tok.pad_token_id]*(length-len(x)) for x,y in batch],device=device)
  y=torch.tensor([y+[-100]*(length-len(y)) for x,y in batch],device=device)
  mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x,y in batch],device=device)
  opt.zero_grad(set_to_none=True);loss=model(input_ids=x,attention_mask=mask,labels=y,use_cache=False).loss
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite controller training loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
  if (step+1)%20==0 or step+1==TRAIN_STEPS:
   save_tensor(cp,{'theta':w.param_flat(params).detach().cpu(),'optimizer':opt.state_dict(),'steps':step+1})
   dump(out/'controller_progress.json',{'steps':step+1,'loss':float(loss.detach()),'done':step+1==TRAIN_STEPS})
   print('V2 CONTROLLER_TRAIN',step+1,'/',TRAIN_STEPS,'loss',float(loss.detach()),flush=True);pack(out,drive)
 if any(p._version!=frozen[n] for n,p in model.named_parameters() if not p.requires_grad):raise RuntimeError('Controller base/A changed')
 model.eval()
 for p in params:p.requires_grad_(False)
 readiness=out/'controller_readiness.json'
 if readiness.exists():report=json.loads(readiness.read_text())
 else:
  episodes=[]
  for task in HELDOUT:
   for x in (0,1):
    for y in (0,1):
     cells={a:0 for a in ADDRESSES};cells[task[0]]=x;cells[task[1]]=y;before=dict(cells);regs={};trace=[];halt=False
     def write(a,val):
      if a!=task[2]:return False
      cells[a]=val;return True
     for k in range(12):
      action,scores=select_action(model,tok,prompt(task,trace),device)
      feedback=dispatch(action,regs,lambda a:cells[a],write)
      trace.append({'action':action,'feedback':feedback,'scores':scores})
      if action=='HALT':halt=True;break
     required={f'READ {task[0]} R0',f'READ {task[1]} R1','XOR R0 R1 R2',f'WRITE {task[2]} R2'}
     passed=(halt and required.issubset({r['action'] for r in trace}) and cells[task[2]]==x^y
             and all(cells[a]==before[a] for a in ADDRESSES if a!=task[2])
             and not any(r['feedback'].startswith('ERROR') or r['feedback']=='WRITE_REJECTED' for r in trace))
     episodes.append({'task':task,'input_bits':[x,y],'pass':passed,'trace':trace})
     print('V2 CONTROLLER_HELDOUT',task,x,y,passed,flush=True)
  report={'pass':all(e['pass'] for e in episodes),'episodes':episodes,'scope':'symbolic runtime readiness; held-out address tuples; no model memory writes; not B5 acceptance'}
  dump(readiness,report);pack(out,drive)
 return tok,model,report
