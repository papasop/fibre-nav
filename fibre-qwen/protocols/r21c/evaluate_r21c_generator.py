#!/usr/bin/env python3
import argparse, gc, json, math, re, statistics, time
from pathlib import Path

PROTOCOL="FIBRE_QWEN_ORACLE_GENERATOR_CAPACITY_R21C_DEVELOPMENT"
MODELS=("Qwen/Qwen3-0.6B","Qwen/Qwen3-8B")

def norm(s): return re.sub(r"[\s`*_，。；：、,.!?！？()（）\[\]{}\"']+","",s.casefold())
def semantic(text,contract):
 z=norm(text); gp=[any(norm(t) in z for t in group) for group in contract["required_groups"]]; bad=[t for t in contract.get("forbidden",[]) if norm(t) in z]
 return all(gp) and not bad,{"required_group_pass":gp,"forbidden_hits":bad}
def health(ids):
 grams=[tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3))]
 return 10<=len(ids)<=180 and len(set(ids))/max(1,len(ids))>=.26 and (len(grams)-len(set(grams)))/max(1,len(grams))<=.2

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--cards",required=True);ap.add_argument("--eval",required=True);ap.add_argument("--outdir",required=True);ap.add_argument("--models",nargs="*",default=list(MODELS));a=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 rows=[json.loads(x) for x in Path(a.eval).read_text().splitlines() if x.strip()];cards={c["id"]:c for c in json.loads(Path(a.cards).read_text())};out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 records=[];started=time.time()
 for model_name in a.models:
  print(f"[model] loading {model_name}",flush=True);tok=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True);tok.pad_token=tok.eos_token
  m=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True).eval()
  for style in ("oracle_free","oracle_slots"):
   for row in rows:
    c=cards[row["expected_card"]]
    if style=="oracle_free":
     sys=("你是严谨研究助手。只依据给定规则回答。先给结论，再给理由和边界。不得编造实验数字，也不得复述提示。\n"
          f"规则：{c['rule']}\n关键事实：{c['facts']}\n禁止主张：{c['prohibited']}")
    else:
     sys=("你是严谨研究助手。只依据给定规则回答。必须严格输出四行，每行只能写一个完整句子，并保留行首标签：\n"
          "结论：直接回答问题。\n理由一：写第一个关键事实。\n理由二：写第二个关键事实。\n边界：明确不能推出什么。\n"
          "不得编造数字，不得复述格式说明。\n"
          f"规则：{c['rule']}\n关键事实：{c['facts']}\n禁止主张：{c['prohibited']}")
    chat=[{"role":"system","content":sys},{"role":"user","content":row["prompt"]}]
    p=tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,enable_thinking=False);x=tok(p,return_tensors="pt").to(m.device)
    with torch.inference_mode(): y=m.generate(**x,min_new_tokens=28,max_new_tokens=180,do_sample=False,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
    ids=y[0,x.input_ids.shape[1]:].tolist();text=tok.decode(ids,skip_special_tokens=True).strip();sem,audit=semantic(text,row["contract"])
    slot_ok=True
    if style=="oracle_slots": slot_ok=all(label in text for label in ("结论：","理由一：","理由二：","边界："))
    rec={"model":model_name,"style":style,"id":row["id"],"oracle_card":row["expected_card"],"generation":text,"semantic":sem,"semantic_audit":audit,"health":health(ids),"slot_format":slot_ok,"tokens":len(ids)}
    records.append(rec);print(f"[{model_name.split('/')[-1]} {style}] {row['id']} semantic={sem} health={rec['health']} format={slot_ok}",flush=True)
  del m;gc.collect();torch.cuda.empty_cache()
 by={}
 for model_name in a.models:
  for style in ("oracle_free","oracle_slots"):
   v=[r for r in records if r["model"]==model_name and r["style"]==style];by[f"{model_name}|{style}"]={"semantic":sum(r["semantic"] for r in v),"healthy":sum(r["health"] for r in v),"slot_format":sum(r["slot_format"] for r in v)}
 small_free=by[f"{a.models[0]}|oracle_free"];small_slots=by[f"{a.models[0]}|oracle_slots"]
 large_slots=by[f"{a.models[-1]}|oracle_slots"]
 gates={"ten_same_development_items":len(rows)==10,"all_records_present":len(records)==len(a.models)*20,"all_generations_healthy":all(r["health"] for r in records),"small_slots_format_at_least_9_of_10":small_slots["slot_format"]>=9,"small_slots_semantic_beats_small_free":small_slots["semantic"]>small_free["semantic"],"large_slots_semantic_at_least_7_of_10":large_slots["semantic"]>=7}
 if large_slots["semantic"]>=7 and small_slots["semantic"]<7:diagnosis="QWEN_0P6B_CAPACITY_LIMIT_SUPPORTED"
 elif small_slots["semantic"]>=7:diagnosis="STRUCTURED_DECODING_CANDIDATE_SUPPORTED"
 elif large_slots["semantic"]<7:diagnosis="CARD_OR_EVALUATION_REPRESENTATION_LIMITING"
 else:diagnosis="GENERATOR_DIAGNOSTIC_INCONCLUSIVE"
 summary={"protocol":PROTOCOL,"mode":"same_item_oracle_retrieval_generator_mechanism_diagnostic","models":a.models,"items":len(rows),"counts":by,"gates":gates,"diagnosis":diagnosis,"wall_seconds":time.time()-started,"scientific_status":"R21C_GENERATOR_CANDIDATE_NOMINATED" if diagnosis in ("QWEN_0P6B_CAPACITY_LIMIT_SUPPORTED","STRUCTURED_DECODING_CANDIDATE_SUPPORTED") else "R21C_GENERATOR_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Same ten authored development items with declared oracle retrieval. Qwen3-8B is a capacity control, not a personalized model. The oracle is non-deployable and the slot prompt is development-selected. This diagnoses generator capacity and output structure only; it is not untouched confirmation, weight learning, broad personalization, safety certification, moving-fibre evidence, or deployment readiness."}
 (out/"records.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if summary["scientific_status"].endswith("NOMINATED") else 2
if __name__=="__main__":raise SystemExit(main())
