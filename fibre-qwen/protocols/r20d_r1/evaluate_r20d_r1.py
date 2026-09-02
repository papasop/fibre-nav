#!/usr/bin/env python3
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path

RUBRIC={
 "p18":{"groups":[["一阶","切向"],["局部","方向"],["不自动","不能保证"]],"forbid":["数字表示","Digital Representation"]},
 "p19":{"groups":[["保留","记录"],["诊断","定位","原因"],["冻结","新种子","停止扩张"]],"forbid":["删除失败"]},
 "p20":{"groups":[["核验","证据"],["边界","范围"],["下一步","行动"]],"forbid":["迎合"]},
}
def health(ids):
 n=len(ids);u=len(set(ids))/max(n,1);grams=[tuple(ids[i:i+4]) for i in range(max(0,n-3))];rep=(len(grams)-len(set(grams)))/max(len(grams),1)
 run=best=1
 for i in range(1,n):run=run+1 if ids[i]==ids[i-1] else 1;best=max(best,run)
 return {"generated_tokens":n,"unique_token_fraction":u,"repeated_4gram_fraction":rep,"max_identical_token_run":best,"healthy":20<=n<=160 and u>=.35 and rep<=.15 and best<=4}
def semantic(rid,text):
 q=RUBRIC[rid];hits=[any(x in text for x in g) for g in q["groups"]];forbidden=[x for x in q["forbid"] if x in text];return {"group_hits":hits,"forbidden_hits":forbidden,"pass":sum(hits)>=2 and not forbidden}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--adapter",required=True);ap.add_argument("--data",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 torch.manual_seed(20260903);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);rows=[json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
 tok=AutoTokenizer.from_pretrained(a.adapter,trust_remote_code=True);tok.pad_token=tok.eos_token
 def load(adapt):
  m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
  return PeftModel.from_pretrained(m,a.adapter).eval() if adapt else m
 def prefix(p):return tok.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def logp(m,p,answer):
  pre=prefix(p);pn=tok(pre,return_tensors="pt").input_ids.shape[1];x=tok(pre+answer+tok.eos_token,return_tensors="pt").input_ids.to(m.device)
  with torch.inference_mode():z=torch.log_softmax(m(x).logits[:,:-1].float(),-1).gather(-1,x[:,1:].unsqueeze(-1)).squeeze(-1)
  return float(z[:,max(0,pn-1):].mean())
 def generate(m,p):
  x=tok(prefix(p),return_tensors="pt").to(m.device)
  with torch.inference_mode():y=m.generate(**x,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
  ids=y[0,x.input_ids.shape[1]:].tolist();return tok.decode(ids,skip_special_tokens=True).strip(),ids
 rec=[];by={}
 for arm,adapt in (("base",False),("stable_adapter",True)):
  m=load(adapt);by[arm]=[]
  for r in rows:
   pp=logp(m,r["prompt"],r["preferred"]);rp=logp(m,r["prompt"],r["rejected"]);text,ids=generate(m,r["prompt"]);z={"id":r["id"],"arm":arm,"preference_margin":pp-rp,"prefers_declared":pp>rp,"generation":text,"health":health(ids),"semantic":semantic(r["id"],text)};rec.append(z);by[arm].append(z);print(f"[{arm}] {r['id']} margin={z['preference_margin']:.6f} health={z['health']['healthy']} semantic={z['semantic']['pass']}",flush=True)
  del m
  if torch.cuda.is_available():torch.cuda.empty_cache()
 base=by["base"];ad=by["stable_adapter"];delta=[y["preference_margin"]-x["preference_margin"] for x,y in zip(base,ad)]
 gates={"three_frozen_records_exact":len(rows)==3,"all_numeric_finite":all(math.isfinite(x["preference_margin"]) for x in rec),"adapter_margin_improves_at_least_2_of_3":sum(x>0 for x in delta)>=2,"median_margin_delta_positive":statistics.median(delta)>0,"adapter_prefers_declared_at_least_2_of_3":sum(x["prefers_declared"] for x in ad)>=2,"all_adapter_generations_healthy":all(x["health"]["healthy"] for x in ad),"adapter_semantic_pass_at_least_2_of_3":sum(x["semantic"]["pass"] for x in ad)>=2,"no_adapter_generation_health_regression":statistics.median(x["health"]["unique_token_fraction"] for x in ad)>=statistics.median(x["health"]["unique_token_fraction"] for x in base)-.05}
 passed=all(gates.values());summary={"protocol":"FIBRE_QWEN3_STABLE_BOOTSTRAP_DUAL_GATE_R20D_R1","mode":"post_collapse_stability_development_audit","margin_deltas":delta,"median_margin_delta":statistics.median(delta),"base_declared_count":sum(x["prefers_declared"] for x in base),"adapter_declared_count":sum(x["prefers_declared"] for x in ad),"base_healthy_count":sum(x["health"]["healthy"] for x in base),"adapter_healthy_count":sum(x["health"]["healthy"] for x in ad),"base_semantic_count":sum(x["semantic"]["pass"] for x in base),"adapter_semantic_count":sum(x["semantic"]["pass"] for x in ad),"gates":gates,"scientific_status":"R20D_R1_STABLE_BOOTSTRAP_DUAL_SIGNAL_SUPPORTED" if passed else "R20D_R1_STABLE_BOOTSTRAP_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Post-collapse development audit on the same three held-out conversation-derived records, with a rubric added after R20d. Not independent confirmation, broad personalization, moving-fibre evidence, safety certification, or deployment readiness."}
 (out/"records.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
