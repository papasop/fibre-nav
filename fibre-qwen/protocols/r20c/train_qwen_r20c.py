#!/usr/bin/env python3
"""Conversation-derived bootstrap LoRA for Qwen3-0.6B; deliberately non-confirmatory."""
import argparse,json,math,random
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--outdir",required=True);ap.add_argument("--model",default="Qwen/Qwen3-0.6B");ap.add_argument("--epochs",type=int,default=5);a=ap.parse_args()
 import torch
 from torch.utils.data import DataLoader,Dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,get_peft_model
 torch.manual_seed(20260902);random.seed(20260902);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()];train=[r for r in rows if r["split"]=="train"]
 tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
 model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"));model.train()
 class D(Dataset):
  def __len__(self):return len(train)
  def __getitem__(self,i):
   r=train[i];prompt=tok.apply_chat_template([{"role":"user","content":r["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False);full=prompt+r["preferred"]+tok.eos_token
   x=tok(full,max_length=512,truncation=True);p=len(tok(prompt,max_length=512,truncation=True)["input_ids"]);lab=x["input_ids"].copy();lab[:p]=[-100]*min(p,len(lab));return {"input_ids":x["input_ids"],"attention_mask":x["attention_mask"],"labels":lab}
 def collate(batch):
  m=max(len(x["input_ids"]) for x in batch);res={k:[] for k in ("input_ids","attention_mask","labels")}
  for x in batch:
   n=m-len(x["input_ids"]);res["input_ids"].append(x["input_ids"]+[tok.pad_token_id]*n);res["attention_mask"].append(x["attention_mask"]+[0]*n);res["labels"].append(x["labels"]+[-100]*n)
  return {k:torch.tensor(v,device=model.device) for k,v in res.items()}
 dl=DataLoader(D(),batch_size=2,shuffle=True,collate_fn=collate);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);losses=[]
 for ep in range(a.epochs):
  for b in dl:
   loss=model(**b).loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad();losses.append(float(loss));print(f"[train] epoch={ep+1}/{a.epochs} loss={losses[-1]:.6f}",flush=True)
 adapter=out/"fibre_qwen_r20c_adapter";model.save_pretrained(adapter);tok.save_pretrained(adapter)
 summary={"protocol":"FIBRE_QWEN3_0P6B_CONVERSATION_BOOTSTRAP_R20C","model":a.model,"records":len(rows),"train_records":len(train),"development_records":sum(r["split"]=="development" for r in rows),"frozen_records":sum(r["split"]=="frozen" for r in rows),"epochs":a.epochs,"initial_loss":losses[0],"final_loss":losses[-1],"adapter_path":adapter.name,"scientific_status":"R20C_BOOTSTRAP_ADAPTER_TRAINED_NONCONFIRMATORY","claim_boundary":"Twenty conversation-derived, user-authorized bootstrap preferences. Produces a first personal adapter; does not establish generalization, moving-fibre superiority, safety, continual learning, or a confirmed personal model."}
 (out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
