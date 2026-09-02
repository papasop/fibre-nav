#!/usr/bin/env python3
import argparse,json,math,statistics,time
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--adapter",required=True);ap.add_argument("--data",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 torch.manual_seed(20260902);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);rows=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
 tok=AutoTokenizer.from_pretrained(a.adapter,trust_remote_code=True);tok.pad_token=tok.eos_token;base_name="Qwen/Qwen3-0.6B"
 def load(adapter=False):
  m=AutoModelForCausalLM.from_pretrained(base_name,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
  return PeftModel.from_pretrained(m,a.adapter).eval() if adapter else m
 def avg_logp(model,prompt,answer):
  prefix=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  p=tok(prefix,return_tensors="pt").input_ids.to(model.device);full=tok(prefix+answer+tok.eos_token,return_tensors="pt").input_ids.to(model.device)
  with torch.inference_mode(): logits=model(full).logits[:,:-1].float();targets=full[:,1:];lp=torch.log_softmax(logits,-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
  start=max(p.shape[1]-1,0);return float(lp[:,start:].mean())
 def generate(model,prompt):
  text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False);x=tok(text,return_tensors="pt").to(model.device)
  with torch.inference_mode(): y=model.generate(**x,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
  return tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip()
 records=[];models={}
 for arm,is_adapter in (("base",False),("personal_adapter",True)):
  model=load(is_adapter);models[arm]=[]
  for r in rows:
   pref=avg_logp(model,r["prompt"],r["preferred"]);rej=avg_logp(model,r["prompt"],r["rejected"]);gen=generate(model,r["prompt"])
   rec={"id":r["id"],"concept":r["concept"],"arm":arm,"preferred_avg_logp":pref,"rejected_avg_logp":rej,"preference_margin":pref-rej,"prefers_declared_answer":pref>rej,"generation":gen};records.append(rec);models[arm].append(rec);print(f"[{arm}] {r['id']} margin={pref-rej:.6f}",flush=True)
  del model
  if torch.cuda.is_available():torch.cuda.empty_cache()
 base=models["base"];adapt=models["personal_adapter"];deltas=[y["preference_margin"]-x["preference_margin"] for x,y in zip(base,adapt)]
 gates={"three_frozen_records_exact":len(rows)==3,"all_finite":all(math.isfinite(x["preference_margin"]) for x in records),"adapter_margin_improves_at_least_2_of_3":sum(d>0 for d in deltas)>=2,"median_margin_delta_positive":statistics.median(deltas)>0,"adapter_prefers_declared_at_least_2_of_3":sum(x["prefers_declared_answer"] for x in adapt)>=2}
 passed=all(gates.values());summary={"protocol":"FIBRE_QWEN3_0P6B_BOOTSTRAP_FROZEN_EVAL_R20D","mode":"three_record_frozen_development_audit","records":len(rows),"margin_deltas_adapter_minus_base":deltas,"median_margin_delta":statistics.median(deltas),"base_declared_preference_count":sum(x["prefers_declared_answer"] for x in base),"adapter_declared_preference_count":sum(x["prefers_declared_answer"] for x in adapt),"gates":gates,"scientific_status":"R20D_BOOTSTRAP_FROZEN_SIGNAL_SUPPORTED" if passed else "R20D_BOOTSTRAP_FROZEN_SIGNAL_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Three held-out conversation-derived records only. This audits preference likelihood and exports free generations; it is not independent human evaluation, broad generalization, moving-fibre superiority, safety, continual learning, or deployment readiness."}
 (out/"records.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
