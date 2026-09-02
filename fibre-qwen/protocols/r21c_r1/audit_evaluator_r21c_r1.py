#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

PROTOCOL="FIBRE_QWEN_R21C_R1_SEMANTIC_EVALUATOR_AUDIT"
EXPECTED_RECORDS_SHA256="e0e815134da012a3f4f5a164206f4095fcb6d7732858bfc4ef194767f71c9c4e"
NEGATORS=("不","不能","无法","并非","不是","不得","未","没有","避免","禁止","不可","不应")

def compact(s): return re.sub(r"[\s`*_，。；：、,.!?！？()（）\[\]{}\"']+","",s.casefold())
def present(text, alternatives):
 z=compact(text);return any(compact(x) in z for x in alternatives)
def affirmative_forbidden(text, phrases):
 z=compact(text);hits=[]
 for phrase in phrases:
  p=compact(phrase);start=0
  while True:
   i=z.find(p,start)
   if i<0:break
   prefix=z[max(0,i-10):i]
   if not any(n in prefix for n in NEGATORS):hits.append(phrase)
   start=i+len(p)
 return hits
def score(text,c):
 conclusion=present(text,c["conclusion"]);core=[present(text,g) for g in c["core_groups"]];boundary=present(text,c["boundary"]);bad=affirmative_forbidden(text,c.get("forbidden_affirmative",[]))
 # Exact rule: correct conclusion, every declared core fact, one boundary, no affirmative prohibited claim.
 ok=conclusion and all(core) and boundary and not bad
 return ok,{"conclusion":conclusion,"core_groups":core,"boundary":boundary,"affirmative_forbidden_hits":bad}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--records",default=str(Path(__file__).with_name("frozen_records.json")));ap.add_argument("--contracts",default=str(Path(__file__).with_name("contracts_v2.json")));ap.add_argument("--outdir",required=True);a=ap.parse_args()
 rb=Path(a.records).read_bytes();rows=json.loads(rb);contracts=json.loads(Path(a.contracts).read_text());out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 actual=hashlib.sha256(rb).hexdigest();rescored=[]
 for r in rows:
  ok,audit=score(r["generation"],contracts[r["id"]]);z=dict(r);z["semantic_v1"]=r["semantic"];z["semantic_v2"]=ok;z["semantic_v2_audit"]=audit
  z["slot_format_v2"]=(all(x in r["generation"] for x in ("结论：","理由一：","理由二：","边界：")) if r["style"]=="oracle_slots" else None);rescored.append(z)
  print(f"[{r['model'].split('/')[-1]} {r['style']}] {r['id']} v1={r['semantic']} v2={ok} audit={audit}",flush=True)
 def count(model,style,key):return sum(bool(x[key]) for x in rescored if x["model"]==model and x["style"]==style)
 models=sorted({x["model"] for x in rows},key=lambda x:float(re.search(r'(\d+(?:\.\d+)?)B$',x).group(1)))
 counts={f"{m}|{s}":{"semantic_v1":count(m,s,"semantic_v1"),"semantic_v2":count(m,s,"semantic_v2"),"healthy":count(m,s,"health"),"slot_format_v2":(sum(x["slot_format_v2"] is True for x in rescored if x["model"]==m and x["style"]==s) if s=="oracle_slots" else None)} for m in models for s in ("oracle_free","oracle_slots")}
 tests=[
  ("无法自动保证任意有限步提升",["自动保证任意"],False),
  ("ker DR 自动保证任意有限步提升",["自动保证任意"],True),
  ("不能只凭一分钟无输出认定死锁",["一分钟无输出认定死锁"],False),
  ("一分钟无输出就能认定死锁",["认定死锁"],True),
 ]
 meta=[{"text":t,"expected_affirmative":e,"observed_affirmative":bool(affirmative_forbidden(t,p))} for t,p,e in tests]
 gates={"frozen_records_sha256_exact":actual==EXPECTED_RECORDS_SHA256,"forty_records_exact":len(rows)==40,"two_models_two_styles_exact":len(counts)==4,"negation_metamorphic_tests_all_pass":all(x["expected_affirmative"]==x["observed_affirmative"] for x in meta),"free_slot_format_is_null":all(x["slot_format_v2"] is None for x in rescored if x["style"]=="oracle_free")}
 summary={"protocol":PROTOCOL,"mode":"post_r21c_offline_evaluator_validity_audit","frozen_records_sha256":actual,"counts":counts,"metamorphic_tests":meta,"gates":gates,"scientific_status":"R21C_V1_EVALUATOR_MECHANICALLY_INVALIDATED" if all(gates.values()) else "R21C_R1_EVALUATOR_AUDIT_FAILED","interpretation_boundary":"V2 corrects declared lexical and polarity defects and rescoring is reported transparently. Contracts were revised after inspecting R21c outputs, so v2 scores are development diagnostics, not blinded human labels or untouched confirmation. They must not be used alone to nominate a deployable model."}
 (out/"rescored_records.json").write_text(json.dumps(rescored,ensure_ascii=False,indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if all(gates.values()) else 2
if __name__=="__main__":raise SystemExit(main())
