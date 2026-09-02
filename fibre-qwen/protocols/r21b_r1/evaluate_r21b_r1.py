#!/usr/bin/env python3
import argparse,json,math,statistics
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cards",required=True);ap.add_argument("--long",required=True);ap.add_argument("--eval",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rows=[json.loads(x) for x in Path(a.eval).read_text().splitlines() if x.strip()];cards=json.loads(Path(a.cards).read_text());long=Path(a.long).read_text();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",trust_remote_code=True);tok.pad_token=tok.eos_token;m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
 def select(q):
  z=sorted(((sum(k.casefold() in q.casefold() for k in c["keywords"]),i,c) for i,c in enumerate(cards)),key=lambda x:(-x[0],x[1]));return [([x[2] for x in z if x[0]>0] or [cards[-1]])[0]]
 def system(arm,r):
  if arm=="bare":return "",[]
  if arm=="long_constitution":return long,[]
  chosen=select(r["prompt"]);text="你是严谨研究助手。只使用下面一条相关规则。不要复述提示，不要讨论无关概念。必须输出四个完整句子：第一句直接回答；第二、三句解释理由；第四句说明不能推出什么。总长度至少60个中文字。\n"+"\n".join(f"相关规则：{c['rule']}\n回答必须体现：{c['positive']}" for c in chosen);return text,[c["id"] for c in chosen]
 def pre(arm,r):
  s,ids=system(arm,r);msg=([{"role":"system","content":s}] if s else [])+[{"role":"user","content":r["prompt"]}];return tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False),ids
 def lp(p,ans):
  n=tok(p,return_tensors="pt").input_ids.shape[1];x=tok(p+ans+tok.eos_token,return_tensors="pt").input_ids.to(m.device)
  with torch.inference_mode():z=torch.log_softmax(m(x).logits[:,:-1].float(),-1).gather(-1,x[:,1:].unsqueeze(-1)).squeeze(-1)
  return float(z[:,max(0,n-1):].mean())
 rec=[];by={}
 for arm in ("bare","long_constitution","targeted_cards"):
  by[arm]=[]
  for r in rows:
   p,ids=pre(arm,r);margin=lp(p,r["preferred"])-lp(p,r["rejected"]);x=tok(p,return_tensors="pt").to(m.device)
   with torch.inference_mode():y=m.generate(**x,min_new_tokens=48,max_new_tokens=160,do_sample=False,repetition_penalty=1.05,pad_token_id=tok.eos_token_id)
   text=tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True).strip();t=y[0,x.input_ids.shape[1]:].tolist();grams=[tuple(t[i:i+4]) for i in range(max(0,len(t)-3))];health=15<=len(t)<=160 and len(set(t))/max(1,len(t))>=.3 and (len(grams)-len(set(grams)))/max(1,len(grams))<=.18;sem=sum(k in text for k in r["keywords"])>=2 and not any(k in text for k in r["forbid"]);z={"id":r["id"],"arm":arm,"routed_cards":ids,"margin":margin,"prefers_declared":margin>0,"generation":text,"health":health,"semantic":sem};rec.append(z);by[arm].append(z);print(f"[{arm}] {r['id']} cards={ids} margin={margin:.3f} semantic={sem}",flush=True)
 base=by["bare"];target=by["targeted_cards"];delta=[b["margin"]-a["margin"] for a,b in zip(base,target)];g={"ten_items":len(rows)==10,"all_finite":all(math.isfinite(x["margin"]) for x in rec),"target_margin_improves_7_of_10":sum(x>0 for x in delta)>=7,"median_delta_positive":statistics.median(delta)>0,"target_prefers_declared_8_of_10":sum(x["prefers_declared"] for x in target)>=8,"target_health_9_of_10":sum(x["health"] for x in target)>=9,"target_semantic_7_of_10":sum(x["semantic"] for x in target)>=7,"target_semantic_beats_long":sum(x["semantic"] for x in target)>sum(x["semantic"] for x in by["long_constitution"])};ok=all(g.values())
 summary={"protocol":"FIBRE_QWEN_TARGETED_RULE_RETRIEVAL_R21B_DIAGNOSTIC","mode":"same_item_post_r21a_mechanism_diagnostic","median_target_margin_delta":statistics.median(delta),"improved_items":sum(x>0 for x in delta),"counts":{arm:{"declared":sum(x["prefers_declared"] for x in v),"semantic":sum(x["semantic"] for x in v),"healthy":sum(x["health"] for x in v)} for arm,v in by.items()},"gates":g,"scientific_status":"R21B_TARGETED_PROFILE_CANDIDATE_SUPPORTED" if ok else "R21B_TARGETED_PROFILE_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Same authored items reused after R21a to diagnose targeted versus full-context prompting. Development only; not untouched confirmation, weight learning, moving-fibre evidence, or broad personalization."};(out/"records.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");(out/"rule_cards.json").write_text(json.dumps(cards,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if ok else 2
if __name__=="__main__":raise SystemExit(main())
