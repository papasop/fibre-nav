#!/usr/bin/env python3
import argparse,json,math,statistics
from pathlib import Path
def health(ids):
 n=len(ids);g=[tuple(ids[i:i+4]) for i in range(max(0,n-3))];rep=(len(g)-len(set(g)))/max(1,len(g));return {"tokens":n,"unique_fraction":len(set(ids))/max(1,n),"repeated_4gram_fraction":rep,"pass":15<=n<=192 and len(set(ids))/max(1,n)>=.3 and rep<=.18}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--profile",required=True);ap.add_argument("--eval",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(2101);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);profile=Path(a.profile).read_text();rows=[json.loads(x) for x in Path(a.eval).read_text().splitlines() if x.strip()]
 tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",trust_remote_code=True);tok.pad_token=tok.eos_token;m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
 def pre(r,use):
  msg=([{"role":"system","content":profile}] if use else [])+[{"role":"user","content":r["prompt"]}];return tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def lp(prefix,ans):
  pn=tok(prefix,return_tensors="pt").input_ids.shape[1];x=tok(prefix+ans+tok.eos_token,return_tensors="pt").input_ids.to(m.device)
  with torch.inference_mode():z=torch.log_softmax(m(x).logits[:,:-1].float(),-1).gather(-1,x[:,1:].unsqueeze(-1)).squeeze(-1)
  return float(z[:,max(0,pn-1):].mean())
 rec=[];by={}
 for arm,use in (("bare",False),("research_profile",True)):
  by[arm]=[]
  for r in rows:
   p=pre(r,use);margin=lp(p,r["preferred"])-lp(p,r["rejected"]);x=tok(p,return_tensors="pt").to(m.device)
   with torch.inference_mode():y=m.generate(**x,max_new_tokens=192,do_sample=False,pad_token_id=tok.eos_token_id)
   ids=y[0,x.input_ids.shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=True).strip();sem=sum(k in text for k in r["keywords"])>=2 and not any(k in text for k in r["forbid"]);q={"id":r["id"],"arm":arm,"margin":margin,"prefers_declared":margin>0,"generation":text,"health":health(ids),"semantic_pass":sem};rec.append(q);by[arm].append(q);print(f"[{arm}] {r['id']} margin={margin:.4f} health={q['health']['pass']} semantic={sem}",flush=True)
 d=[b["margin"]-a["margin"] for a,b in zip(by["bare"],by["research_profile"])];prof=by["research_profile"]
 gates={"ten_new_development_items":len(rows)==10,"all_finite":all(math.isfinite(x["margin"]) for x in rec),"margin_improves_at_least_7_of_10":sum(x>0 for x in d)>=7,"median_margin_delta_positive":statistics.median(d)>0,"profile_prefers_declared_at_least_8_of_10":sum(x["prefers_declared"] for x in prof)>=8,"profile_health_at_least_9_of_10":sum(x["health"]["pass"] for x in prof)>=9,"profile_semantic_at_least_7_of_10":sum(x["semantic_pass"] for x in prof)>=7}
 ok=all(gates.values());summary={"protocol":"FIBRE_QWEN_RESEARCH_CONSTITUTION_R21A_DEVELOPMENT","margin_deltas":d,"median_margin_delta":statistics.median(d),"bare_declared":sum(x["prefers_declared"] for x in by["bare"]),"profile_declared":sum(x["prefers_declared"] for x in prof),"bare_semantic":sum(x["semantic_pass"] for x in by["bare"]),"profile_semantic":sum(x["semantic_pass"] for x in prof),"profile_healthy":sum(x["health"]["pass"] for x in prof),"gates":gates,"scientific_status":"R21A_RESEARCH_PROFILE_CANDIDATE_SUPPORTED" if ok else "R21A_RESEARCH_PROFILE_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Development comparison of bare versus constitution-conditioned Qwen3-0.6B on ten authored items. The profile contains domain knowledge; this is not weight learning, independent confirmation, moving-fibre superiority, or broad personal-model validation."}
 (out/"records.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");(out/"research_constitution.txt").write_text(profile);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if ok else 2
if __name__=="__main__":raise SystemExit(main())
