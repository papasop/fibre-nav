#!/usr/bin/env python3
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
root=Path(__file__).resolve().parent;profile=(root/"research_constitution.txt").read_text();mid="Qwen/Qwen3-0.6B";tok=AutoTokenizer.from_pretrained(mid,trust_remote_code=True);m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,device_map="auto",trust_remote_code=True).eval();history=[]
print("R21a research-profile Qwen. Type /quit to exit.")
while True:
 q=input("你> ").strip()
 if q=="/quit":break
 history=(history+[{"role":"user","content":q}])[-8:];msg=[{"role":"system","content":profile}]+history;x=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to(m.device)
 with torch.inference_mode():y=m.generate(x,max_new_tokens=384,do_sample=True,temperature=.6,top_p=.9,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
 ans=tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip();print("Qwen>",ans);history.append({"role":"assistant","content":ans})
