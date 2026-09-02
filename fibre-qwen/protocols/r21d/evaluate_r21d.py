#!/usr/bin/env python3
import argparse,csv,gc,hashlib,json,math,re,statistics,time
from pathlib import Path

PROTOCOL="FIBRE_QWEN_PROSPECTIVE_NEW_ITEM_ROUTED_CAPACITY_R21D"
SMALL="Qwen/Qwen3-0.6B"; LARGE="Qwen/Qwen3-8B"
ARMS=((SMALL,"router"),(LARGE,"bare"),(LARGE,"router"))
NEGATORS=("不","不能","无法","并非","不是","不得","未","没有","避免","禁止","不可","不应","不足")

def compact(s): return re.sub(r"[\s`*_，。；：、,.!?！？()（）\[\]{}\"']+","",s.casefold())
def contains(text,alts):
 z=compact(text); return any(compact(a) in z for a in alts)
def affirmative_hits(text,phrases):
 z=compact(text);hits=[]
 for phrase in phrases:
  p=compact(phrase);start=0
  while True:
   i=z.find(p,start)
   if i<0:break
   prefix=z[max(0,i-12):i]
   if not any(n in prefix for n in NEGATORS):hits.append(phrase)
   start=i+max(1,len(p))
 return hits
def semantic(text,c):
 conclusion=contains(text,c["conclusion"]);core=[contains(text,g) for g in c["core"]];boundary=contains(text,c["boundary"]);bad=affirmative_hits(text,c.get("forbidden",[]))
 return conclusion and all(core) and boundary and not bad,{"conclusion":conclusion,"core":core,"boundary":boundary,"affirmative_forbidden_hits":bad}
def route(prompt,cards):
 q=compact(prompt);hits=[]
 for c in cards:
  matched=[p for p in c["patterns"] if compact(p) in q]
  if matched:hits.append((max(map(lambda x:len(compact(x)),matched)),c["id"],matched))
 hits.sort(key=lambda x:(-x[0],x[1]));return (hits[0][1] if hits else "assistant"),hits
def healthy(ids):
 grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
 return 12<=len(ids)<=192 and len(set(ids))/max(1,len(ids))>=.25 and (len(grams)-len(set(grams)))/max(1,len(grams))<=.2

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--outdir",required=True);ap.add_argument("--eval",default=str(Path(__file__).with_name("eval.jsonl")));ap.add_argument("--cards",default=str(Path(__file__).with_name("rule_cards.json")));a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 eval_bytes=Path(a.eval).read_bytes();card_bytes=Path(a.cards).read_bytes();rows=[json.loads(x) for x in eval_bytes.decode().splitlines() if x.strip()];cards=json.loads(card_bytes);cmap={x["id"]:x for x in cards};out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(21021);records=[];started=time.time()
 for model_name in (SMALL,LARGE):
  print(f"[model] loading {model_name}",flush=True);tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True);tok.pad_token=tok.eos_token
  model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
  for mn,style in [x for x in ARMS if x[0]==model_name]:
   arm=f"{model_name.split('/')[-1]}_{style}"
   for row in rows:
    cid,audit=route(row["prompt"],cards)
    sys=""
    if style=="router":
     c=cmap[cid];sys=("你是严谨的中文研究助手。先给结论，再给两项理由和适用边界；明确区分已证事实与愿景，不得编造实验数字。\n研究规则："+c["rule"])
    msgs=([{"role":"system","content":sys}] if sys else [])+[{"role":"user","content":row["prompt"]}]
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);x=tok(prompt,return_tensors="pt").to(model.device)
    with torch.inference_mode():y=model.generate(**x,min_new_tokens=28,max_new_tokens=192,do_sample=False,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
    ids=y[0,x.input_ids.shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=True).strip();ok,sa=semantic(text,row["contract"])
    rec={"id":row["id"],"arm":arm,"model":model_name,"style":style,"expected_card":row["card"],"routed_card":cid if style=="router" else None,"router_exact":cid==row["card"] if style=="router" else None,"router_audit":audit if style=="router" else None,"generation":text,"semantic_v2":ok,"semantic_v2_audit":sa,"healthy":healthy(ids),"tokens":len(ids)}
    records.append(rec);print(f"[{arm}] {row['id']} route={rec['routed_card'] or '-'} semantic={ok} healthy={rec['healthy']}",flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 def arm(name):return [r for r in records if r["arm"]==name]
 s=arm("Qwen3-0.6B_router");b=arm("Qwen3-8B_bare");r=arm("Qwen3-8B_router")
 counts=lambda v:{"semantic":sum(x["semantic_v2"] for x in v),"healthy":sum(x["healthy"] for x in v)}
 improve_small=sum(x["semantic_v2"] and not y["semantic_v2"] for x,y in zip(r,s));improve_bare=sum(x["semantic_v2"] and not y["semantic_v2"] for x,y in zip(r,b))
 auto={"twenty_new_items_exact":len(rows)==20,"sixty_records_exact":len(records)==60,"all_records_finite":all(isinstance(x["tokens"],int) for x in records),"router_exact_20_of_20":sum(x["router_exact"] is True for x in r)==20 and sum(x["router_exact"] is True for x in s)==20,"large_router_semantic_at_least_14_of_20":sum(x["semantic_v2"] for x in r)>=14,"large_router_healthy_at_least_18_of_20":sum(x["healthy"] for x in r)>=18,"large_router_beats_small_on_at_least_6_items":improve_small>=6,"large_router_beats_large_bare_on_at_least_4_items":improve_bare>=4}
 candidate=all(auto.values())
 summary={"protocol":PROTOCOL,"mode":"prospectively_frozen_new_item_automatic_audit","models":[SMALL,LARGE],"arms":["Qwen3-0.6B_router","Qwen3-8B_bare","Qwen3-8B_router"],"items":len(rows),"eval_sha256":hashlib.sha256(eval_bytes).hexdigest(),"cards_sha256":hashlib.sha256(card_bytes).hexdigest(),"counts":{"small_router":counts(s),"large_bare":counts(b),"large_router":counts(r)},"large_router_unique_wins":{"versus_small_router":improve_small,"versus_large_bare":improve_bare},"automatic_gates":auto,"human_review_status":"PENDING_BLINDED_REVIEW","scientific_status":"R21D_AUTOMATIC_CANDIDATE_SUPPORTED_HUMAN_PENDING" if candidate else "R21D_AUTOMATIC_AUDIT_INCONCLUSIVE_FAIL_CLOSED","wall_seconds":time.time()-started,"claim_boundary":"Prospectively frozen automatic audit on twenty newly authored research-assistant items. The v2 contracts were frozen before these outputs, but they are authored lexical contracts rather than blinded human labels. No weight learning occurs. This is not broad personalization, moving-fibre superiority, safety certification, continual learning, or deployment readiness."}
 (out/"records.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 blinded=[]
 for row in rows:
  for rec in [x for x in records if x["id"]==row["id"]]:
   code=hashlib.sha256((row["id"]+"|"+rec["arm"]+"|r21d").encode()).hexdigest()[:12];blinded.append((hashlib.sha256(code.encode()).hexdigest(),code,row["id"],row["prompt"],rec["generation"]))
 blinded.sort()
 with (out/"human_review.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.writer(f);w.writerow(["blind_code","item_id","prompt","generation","semantic_pass_0_or_1","healthy_pass_0_or_1","reviewer_note"]);[w.writerow(x[1:]+("","","")) for x in blinded]
 key={hashlib.sha256((x["id"]+"|"+x["arm"]+"|r21d").encode()).hexdigest()[:12]:x["arm"] for x in records};(out/"blind_key.json").write_text(json.dumps(key,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if candidate else 2
if __name__=="__main__":raise SystemExit(main())

