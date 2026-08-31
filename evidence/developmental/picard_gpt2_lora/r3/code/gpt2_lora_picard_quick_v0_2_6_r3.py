#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, json, math, random, statistics, time, traceback, urllib.request
from pathlib import Path
import numpy as np

PROTOCOL="GPT2_LORA_PICARD_V0_2_6_R3_METRIC_EXPONENT_DEVELOPMENT"
PILOT_SEED=23087
EVAL_SEEDS=[23209,23227]
WARM_STEPS=50
WARM_LR=5e-4

def parse():
 p=argparse.ArgumentParser()
 p.add_argument("--outdir",default="gpt2_lora_picard_quick_results")
 p.add_argument("--model",default="openai-community/gpt2")
 p.add_argument("--device",default="cuda")
 p.add_argument("--steps",type=int,default=600)
 p.add_argument("--pilot-steps",type=int,default=100)
 p.add_argument("--seq-len",type=int,default=128)
 p.add_argument("--batch-size",type=int,default=8)
 p.add_argument("--max-minutes",type=float,default=55.)
 a,u=p.parse_known_args()
 if u:print("[notice] ignored notebook arguments:",u,flush=True)
 return a

def seed_all(seed,torch):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def sync(torch,device):
 if device.type=="cuda":torch.cuda.synchronize()

def load_text_dataset(load_dataset):
 try:
  ds=load_dataset("wikitext","wikitext-2-raw-v1")
  train="\n".join(x for x in ds["train"]["text"] if x.strip())
  valid="\n".join(x for x in ds["validation"]["text"] if x.strip())
  return train,valid,"HuggingFace wikitext-2-raw-v1"
 except Exception as e:
  print(f"[data] WikiText unavailable ({type(e).__name__}); trying public-domain Tiny Shakespeare",flush=True)
  url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
  text=urllib.request.urlopen(url,timeout=60).read().decode("utf-8")
  cut=int(.9*len(text));return text[:cut],text[cut:],url

def blocks(torch,tokenizer,text,seq_len,limit):
 ids=tokenizer(text,add_special_tokens=False,return_attention_mask=False,verbose=False)["input_ids"]
 n=min(len(ids)//seq_len,limit);ids=ids[:n*seq_len]
 if n<64:raise RuntimeError(f"too few language-model blocks: {n}")
 return torch.tensor(ids,dtype=torch.long).view(n,seq_len)

def make_model(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,model_name,device,seed):
 seed_all(seed,torch)
 dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
 base=AutoModelForCausalLM.from_pretrained(model_name,dtype=dtype)
 cfg=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,bias="none",task_type="CAUSAL_LM",target_modules=["c_attn"])
 model=get_peft_model(base,cfg);model.config.use_cache=False;model.to(device);model.train()
 params=[p for p in model.parameters() if p.requires_grad]
 if not params:raise RuntimeError("PEFT produced no trainable LoRA parameters")
 return model,params,dtype

def schedule(torch,n,steps,batch,seed):
 g=torch.Generator().manual_seed(seed);need=steps*batch;chunks=[]
 while sum(x.numel() for x in chunks)<need:chunks.append(torch.randperm(n,generator=g))
 return torch.cat(chunks)[:need].view(steps,batch)

def loss_on(model,x,device,torch,dtype):
 x=x.to(device,non_blocking=True)
 with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"):
  return model(input_ids=x,labels=x).loss

def evaluate(model,val_blocks,device,torch,dtype,batch=8,max_batches=24):
 model.eval();vals=[]
 with torch.no_grad():
  for i in range(0,min(len(val_blocks),batch*max_batches),batch):vals.append(float(loss_on(model,val_blocks[i:i+batch],device,torch,dtype)))
 model.train();return statistics.mean(vals)

def frozen_metric(model,params,train_blocks,device,torch,dtype,batch_size,seed):
 fisher=[torch.zeros_like(p,dtype=torch.float32) for p in params];cal=schedule(torch,len(train_blocks),8,batch_size,seed)
 for ix in cal:
  model.zero_grad(set_to_none=True);loss=loss_on(model,train_blocks[ix],device,torch,dtype);loss.backward()
  for f,p in zip(fisher,params):f.add_(p.grad.detach().float().square(),alpha=1/len(cal))
 model.zero_grad(set_to_none=True);metric=[]
 for f,p in zip(fisher,params):
  scale=f.mean().clamp_min(1e-20);m=(f/scale+.1).to(dtype=p.dtype);metric.append(m)
 vals=torch.cat([m.flatten() for m in metric]);lo=float(vals.min());hi=float(vals.max());return metric,{"min":lo,"median":float(vals.median()),"max":hi,"max_to_min_ratio":hi/max(lo,1e-30)}

def shared_warm_start(model,params,train_blocks,val_blocks,device,torch,dtype,batch_size,seed):
 order=schedule(torch,len(train_blocks),WARM_STEPS,batch_size,seed+301);opt=torch.optim.AdamW(params,lr=WARM_LR,weight_decay=.01)
 for step in range(WARM_STEPS):
  model.zero_grad(set_to_none=True);loss=loss_on(model,train_blocks[order[step]],device,torch,dtype);loss.backward();opt.step()
 model.zero_grad(set_to_none=True);warm_loss=evaluate(model,val_blocks,device,torch,dtype);del opt;return warm_loss

def run_arm(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,train_blocks,val_blocks,model_name,device,kind,lr,alpha,seed,steps,batch_size,target,deadline,eval_every=50):
 model,params,dtype=make_model(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,model_name,device,seed)
 warm_loss=shared_warm_start(model,params,train_blocks,val_blocks,device,torch,dtype,batch_size,seed)
 metric,metric_stats=frozen_metric(model,params,train_blocks,device,torch,dtype,batch_size,seed+991)
 preconditioner=[g.pow(-alpha) for g in metric] if kind!="adamw" else None
 order=schedule(torch,len(train_blocks),steps,batch_size,seed+17);mom=[torch.zeros_like(p) for p in params]
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01) if kind=="adamw" else None
 for _ in range(10):
  with torch.no_grad():_ = model(input_ids=train_blocks[:batch_size].to(device)).logits[:,0,0].sum()
 sync(torch,device);timed=0.;hit=None;hit_step=None;trace=[];max_update_ratio=0.;stable=True
 for step in range(steps):
  if time.monotonic()>deadline:raise TimeoutError("hard runtime deadline reached")
  model.zero_grad(set_to_none=True);sync(torch,device);t=time.perf_counter();loss=loss_on(model,train_blocks[order[step]],device,torch,dtype);loss.backward()
  if kind=="adamw":opt.step()
  else:
   with torch.no_grad():
    grads=[p.grad.detach() for p in params];direction=torch._foreach_add(grads,params,alpha=.001);direction=torch._foreach_mul(direction,preconditioner);torch._foreach_mul_(mom,.85);torch._foreach_add_(mom,direction,alpha=.15);torch._foreach_add_(params,mom,alpha=-lr)
  sync(torch,device);timed+=time.perf_counter()-t
  if (step+1)%eval_every==0 or step==steps-1:
   vl=evaluate(model,val_blocks,device,torch,dtype);param_sq=sum(float(p.detach().float().square().sum()) for p in params);update_sq=sum(float((lr*m.detach().float()).square().sum()) for m in mom) if kind!="adamw" else 0.;ratio=math.sqrt(update_sq/max(param_sq,1e-30)) if kind!="adamw" else 0.;max_update_ratio=max(max_update_ratio,ratio);finite=math.isfinite(vl) and all(bool(torch.isfinite(p).all()) for p in params);stable=stable and finite and ratio<=1.;trace.append({"step":step+1,"train_kernel_seconds":timed,"val_loss":vl,"update_to_parameter_norm_ratio":ratio,"finite":finite})
   print(f"[{kind}] step={step+1}/{steps} train_s={timed:.2f} val_loss={vl:.5f}",flush=True)
   if hit is None and vl<=target:hit=timed;hit_step=step+1
 final_loss=evaluate(model,val_blocks,device,torch,dtype)
 stable=stable and math.isfinite(final_loss);rec={"kind":kind,"metric_exponent":alpha,"seed":seed,"lr":lr,"stable":stable,"max_update_to_parameter_norm_ratio":max_update_ratio,"shared_warm_steps":WARM_STEPS,"shared_warm_lr":WARM_LR,"shared_warm_val_loss":warm_loss,"steps":steps,"time_to_target":hit,"steps_to_target":hit_step,"timed_train_seconds":timed,"final_val_loss":final_loss,"metric_stats":metric_stats,"trainable_parameters":sum(p.numel() for p in params),"trace":trace}
 del opt,mom,metric,preconditioner,params,model;gc.collect()
 if device.type=="cuda":torch.cuda.empty_cache()
 return rec

def main():
 a=parse();start=time.monotonic();deadline=start+a.max_minutes*60
 import torch
 from datasets import load_dataset
 from transformers import AutoModelForCausalLM,AutoTokenizer,GPT2Config,GPT2LMHeadModel
 from peft import LoraConfig,get_peft_model
 if a.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("Select a GPU runtime")
 device=torch.device(a.device if torch.cuda.is_available() else "cpu")
 print(f"protocol={PROTOCOL} device={device} model={a.model} hard_timeout={a.max_minutes:.0f}m",flush=True)
 if device.type=="cuda":print("GPU:",torch.cuda.get_device_name(0),flush=True)
 # Exercise the exact PEFT GPT-2 Conv1D injection route before any large model
 # or dataset download. This catches optional-backend incompatibilities early.
 print("[preflight] local tiny-GPT2 PEFT/LoRA injection",flush=True)
 tiny=GPT2LMHeadModel(GPT2Config(vocab_size=128,n_positions=32,n_ctx=32,n_embd=32,n_layer=1,n_head=1))
 tiny=get_peft_model(tiny,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.,bias="none",task_type="CAUSAL_LM",target_modules=["c_attn"]))
 if not any(p.requires_grad for p in tiny.parameters()):raise RuntimeError("PEFT smoke preflight produced no trainable LoRA parameters")
 del tiny;gc.collect();print("[preflight] PEFT/LoRA injection passed",flush=True)
 tokenizer=AutoTokenizer.from_pretrained(a.model);tokenizer.pad_token=tokenizer.eos_token;tokenizer.model_max_length=10**12
 train_text,val_text,data_source=load_text_dataset(load_dataset);train_blocks=blocks(torch,tokenizer,train_text,a.seq_len,12000);val_blocks=blocks(torch,tokenizer,val_text,a.seq_len,512)
 print(f"[data] source={data_source} train_blocks={len(train_blocks)} val_blocks={len(val_blocks)} seq={a.seq_len}",flush=True)
 AdamGrid=[2e-4,5e-4];PicardGrid=[.3,.6,1.,2.];exponents={"picard_a0":0.,"picard_a025":.25,"picard_a05":.5,"picard_a1":1.};pilot={};chosen={}
 cand=[]
 for lr in AdamGrid:
  print(f"[pilot] adamw lr={lr}",flush=True);r=run_arm(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,train_blocks,val_blocks,a.model,device,"adamw",lr,0.,PILOT_SEED,a.pilot_steps,a.batch_size,-math.inf,deadline,max(25,a.pilot_steps//2));finite=r["stable"] and math.isfinite(r["final_val_loss"]);cand.append((not finite,r["final_val_loss"] if finite else math.inf,lr,r));print(f"[pilot done] finite={finite} loss={r['final_val_loss']:.6f}",flush=True)
 failed,score,lr,selected_adam=min(cand,key=lambda q:q[:3])
 if failed:raise RuntimeError("no finite AdamW pilot candidate")
 chosen["adamw"]=lr;pilot["adamw"]={"lr":lr,"final_val_loss":score,"grid":AdamGrid};target=selected_adam["final_val_loss"]
 geometry_choices=[]
 for kind,alpha in exponents.items():
  cand=[];diagnostics=[]
  for lr in PicardGrid:
   print(f"[pilot] {kind} alpha={alpha} lr={lr}",flush=True);r=run_arm(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,train_blocks,val_blocks,a.model,device,kind,lr,alpha,PILOT_SEED,a.pilot_steps,a.batch_size,-math.inf,deadline,max(25,a.pilot_steps//2));finite=r["stable"] and math.isfinite(r["final_val_loss"]);cand.append((not finite,r["final_val_loss"] if finite else math.inf,lr,r));diagnostics.append({"lr":lr,"stable":r["stable"],"final_val_loss":r["final_val_loss"],"max_update_ratio":r["max_update_to_parameter_norm_ratio"]});print(f"[pilot done] stable={r['stable']} loss={r['final_val_loss']:.6f} max_update_ratio={r['max_update_to_parameter_norm_ratio']:.3e}",flush=True)
  failed,score,lr,r=min(cand,key=lambda q:q[:3]);pilot[kind]={"alpha":alpha,"lr":None if failed else lr,"final_val_loss":None if failed else score,"grid":PicardGrid,"candidates":diagnostics};geometry_choices.append((failed,score,lr,kind,alpha,r))
 failed,score,lr,selected_geometry,selected_alpha,selected_picard=min(geometry_choices,key=lambda q:q[:3])
 if failed:raise RuntimeError("no stable Picard metric-exponent pilot candidate")
 chosen[selected_geometry]=lr;pilot["selected_geometry"]={"kind":selected_geometry,"alpha":selected_alpha,"lr":lr,"rule":"lowest stable finite pilot validation loss"}
 print(f"[freeze-development] adamw_lr={chosen['adamw']} selected={selected_geometry} alpha={selected_alpha} lr={lr} target={target:.6f}",flush=True)
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);records=[]
 for i,seed in enumerate(EVAL_SEEDS,1):
  for kind,lr,alpha in (("adamw",chosen["adamw"],0.),(selected_geometry,chosen[selected_geometry],selected_alpha)):
   print(f"[eval {i}/{len(EVAL_SEEDS)}] seed={seed} {kind}",flush=True);r=run_arm(torch,AutoModelForCausalLM,LoraConfig,get_peft_model,train_blocks,val_blocks,a.model,device,kind,lr,alpha,seed,a.steps,a.batch_size,target,deadline);records.append(r);(out/f"{kind}_{seed}.json").write_text(json.dumps(r,indent=2)+"\n")
 pairs=[]
 for seed in EVAL_SEEDS:
  aa=next(r for r in records if r["seed"]==seed and r["kind"]=="adamw");pp=next(r for r in records if r["seed"]==seed and r["kind"]==selected_geometry);valid=aa["time_to_target"] is not None and pp["time_to_target"] is not None and pp["stable"]
  pairs.append({"seed":seed,"valid":valid,"shared_warm_val_loss_delta":pp["shared_warm_val_loss"]-aa["shared_warm_val_loss"],"time_to_equal_loss_speedup_fraction":(aa["time_to_target"]-pp["time_to_target"])/aa["time_to_target"] if valid else None,"fixed_budget_speedup_fraction":(aa["timed_train_seconds"]-pp["timed_train_seconds"])/aa["timed_train_seconds"],"steps_reduction_fraction":(aa["steps_to_target"]-pp["steps_to_target"])/aa["steps_to_target"] if valid else None,"final_val_loss_delta":pp["final_val_loss"]-aa["final_val_loss"],"metric_max_to_min_ratio":{"adamw":aa["metric_stats"]["max_to_min_ratio"],"picard":pp["metric_stats"]["max_to_min_ratio"]}})
 valid=all(p["valid"] for p in pairs);eq=statistics.median(p["time_to_equal_loss_speedup_fraction"] for p in pairs) if valid else None;fixed=statistics.median(p["fixed_budget_speedup_fraction"] for p in pairs);loss_delta=statistics.median(p["final_val_loss_delta"] for p in pairs)
 gates={"real_gpt2_lora_model":True,"same_lora_parameterization":True,"shared_warm_start_identical":all(abs(p["shared_warm_val_loss_delta"])<=1e-10 for p in pairs),"four_metric_exponents_piloted":set(exponents.values())=={0.,.25,.5,1.},"selected_picard_stable":all(next(r for r in records if r["seed"]==s and r["kind"]==selected_geometry)["stable"] for s in EVAL_SEEDS),"two_new_eval_seeds":len(pairs)==2,"both_arms_reach_development_target":valid,"median_time_to_equal_loss_speedup_positive":valid and eq>0,"median_final_loss_delta_at_most_0_02":loss_delta<=.02,"runtime_within_hard_limit":time.monotonic()<=deadline};signal=all(gates.values())
 summary={"protocol":PROTOCOL,"mode":"metric_exponent_final_development_screen","model":a.model,"data_source":data_source,"sequence_length":a.seq_len,"batch_size":a.batch_size,"shared_warm_start":{"steps":WARM_STEPS,"optimizer":"AdamW","lr":WARM_LR,"excluded_from_comparison_timing":True},"pilot":pilot,"selected_geometry":selected_geometry,"selected_metric_exponent":selected_alpha,"development_target":target,"eval_seeds":EVAL_SEEDS,"pairs":pairs,"median_time_to_equal_loss_speedup_fraction":eq,"median_fixed_budget_speedup_fraction_diagnostic":fixed,"median_final_val_loss_delta":loss_delta,"wall_seconds":time.monotonic()-start,"gates":gates,"scientific_status":"GPT2_LORA_PICARD_V0_2_6_R3_METRIC_EXPONENT_POSITIVE_SIGNAL" if signal else "GPT2_LORA_PICARD_V0_2_6_R3_DIRECT_TRANSFER_NOT_SUPPORTED","claim_boundary":"Final development screen of metric exponents 0, 1/4, 1/2, and 1 after a shared 50-step AdamW warm-start; Tiny Shakespeare fallback is logged when WikiText is unavailable; two evaluation seeds after same-run pilot; fixed-budget speed is diagnostic only; not confirmatory or universal evidence."}
 (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return 0 if signal else 2

if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception:
  traceback.print_exc();raise
