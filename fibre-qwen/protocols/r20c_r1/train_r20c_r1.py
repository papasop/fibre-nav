#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--preferences",required=True);ap.add_argument("--replay",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args()
 import torch
 from torch.utils.data import DataLoader,Dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import LoraConfig,get_peft_model
 torch.manual_seed(20260903);random.seed(20260903);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 prefs=[json.loads(x) for x in Path(a.preferences).read_text().splitlines() if x.strip()];items=[{"prompt":r["prompt"],"answer":r["preferred"],"kind":"preference"} for r in prefs if r["split"]=="train"]
 items += [{**json.loads(x),"kind":"replay"} for x in Path(a.replay).read_text().splitlines() if x.strip()]
 model_id="Qwen/Qwen3-0.6B";tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=True);tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
 model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=.1,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"));model.train()
 class D(Dataset):
  def __len__(self):return len(items)
  def __getitem__(self,i):
   r=items[i];pre=tok.apply_chat_template([{"role":"user","content":r["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False);full=pre+r["answer"]+tok.eos_token
   x=tok(full,max_length=512,truncation=True);n=len(tok(pre,max_length=512,truncation=True)["input_ids"]);lab=x["input_ids"].copy();lab[:min(n,len(lab))]=[-100]*min(n,len(lab));return {"input_ids":x["input_ids"],"attention_mask":x["attention_mask"],"labels":lab}
 def collate(batch):
  m=max(len(x["input_ids"]) for x in batch);z={k:[] for k in ("input_ids","attention_mask","labels")}
  for x in batch:
   n=m-len(x["input_ids"]);z["input_ids"].append(x["input_ids"]+[tok.pad_token_id]*n);z["attention_mask"].append(x["attention_mask"]+[0]*n);z["labels"].append(x["labels"]+[-100]*n)
  return {k:torch.tensor(v,device=model.device) for k,v in z.items()}
 dl=DataLoader(D(),batch_size=2,shuffle=True,collate_fn=collate);opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.02);loss=[]
 for ep in range(2):
  for b in dl:
   q=model(**b).loss;q.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();opt.zero_grad(set_to_none=True);loss.append(float(q));print(f"[train] epoch={ep+1}/2 loss={loss[-1]:.6f}",flush=True)
 adapter=out/"fibre_qwen_r20c_r1_adapter";model.save_pretrained(adapter);tok.save_pretrained(adapter)
 summary={"protocol":"FIBRE_QWEN3_0P6B_STABLE_BOOTSTRAP_R20C_R1","preference_train_records":14,"replay_records":len(items)-14,"frozen_records_untouched":3,"epochs":2,"learning_rate":5e-5,"lora_rank":4,"lora_alpha":8,"initial_loss":loss[0],"final_loss":loss[-1],"scientific_status":"R20C_R1_STABLE_BOOTSTRAP_ADAPTER_TRAINED_NONCONFIRMATORY","claim_boundary":"Stability-oriented retraining on 14 conversation-derived preferences plus 12 generic replay records. Requires frozen preference and generation-health evaluation; not yet a validated personal model or moving-fibre result."}
 (out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
