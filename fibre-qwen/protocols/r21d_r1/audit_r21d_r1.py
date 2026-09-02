#!/usr/bin/env python3
import argparse,csv,hashlib,json,statistics
from collections import Counter
from pathlib import Path

PROTOCOL="FIBRE_QWEN_R21D_R1_FROZEN_OUTPUT_EVALUATOR_AUDIT"
RECORDS_SHA="9316dda03c440f067b4b1a1cf2475c6e448158f03750f71ccfaa39fb8c6b055d"
EVAL_SHA="02a278eb00e6055eff2a65e2dd18a5b149a1625a05e778e5caf7103105f4700a"
CARDS_SHA="261a00407574e7d07ca5e3e99c46a7450d84e7124921387c42c3756b8924d46c"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();here=Path(__file__).parent
 ap.add_argument("--records",default=str(here/"records.json"));ap.add_argument("--eval",default=str(here/"eval.jsonl"));ap.add_argument("--cards",default=str(here/"rule_cards.json"));ap.add_argument("--outdir",required=True);a=ap.parse_args()
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);records=json.loads(Path(a.records).read_text());items=[json.loads(x) for x in Path(a.eval).read_text().splitlines() if x.strip()];prompts={x["id"]:x["prompt"] for x in items}
 hashes={"records":sha(a.records),"eval":sha(a.eval),"cards":sha(a.cards)}
 arms=sorted({x["arm"] for x in records});by={k:[x for x in records if x["arm"]==k] for k in arms}
 component_fail=Counter();core_fail=Counter()
 for r in records:
  q=r["semantic_v2_audit"]
  if not q["conclusion"]:component_fail["conclusion"]+=1
  if not q["boundary"]:component_fail["boundary"]+=1
  for i,v in enumerate(q["core"]):
   if not v:core_fail[str(i)]+=1
 route_misses=[{"arm":r["arm"],"id":r["id"],"expected":r["expected_card"],"observed":r["routed_card"]} for r in records if r["style"]=="router" and r["router_exact"] is False]
 # A diagnostic is mechanically saturated when every arm receives the same all-zero semantic count.
 semantics={k:sum(x["semantic_v2"] for x in v) for k,v in by.items()};saturated=len(set(semantics.values()))==1 and next(iter(semantics.values())) in (0,len(items))
 pairs=[];key={}
 comparisons=[("Qwen3-8B_router","Qwen3-0.6B_router","capacity_plus_routing_vs_small"),("Qwen3-8B_router","Qwen3-8B_bare","routing_increment_at_8b")]
 for left_arm,right_arm,label in comparisons:
  L={x["id"]:x for x in by[left_arm]};R={x["id"]:x for x in by[right_arm]}
  for iid in sorted(prompts):
   token=hashlib.sha256(f"r21d-r1|{label}|{iid}".encode()).hexdigest();swap=int(token[-1],16)%2==1
   aa,bb=(R[iid],L[iid]) if swap else (L[iid],R[iid]);code=token[:12]
   pairs.append({"blind_code":code,"comparison":label,"item_id":iid,"prompt":prompts[iid],"answer_A":aa["generation"],"answer_B":bb["generation"],"winner_A_B_TIE":"","A_correct_0_1":"","B_correct_0_1":"","A_boundary_0_1":"","B_boundary_0_1":"","A_fabricates_0_1":"","B_fabricates_0_1":"","reviewer_note":""})
   key[code]={"A":aa["arm"],"B":bb["arm"],"comparison":label,"item_id":iid}
 fields=list(pairs[0]);
 with (out/"human_pairwise_review.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairs)
 (out/"blind_key.json").write_text(json.dumps(key,ensure_ascii=False,indent=2)+"\n")
 gates={"records_sha_exact":hashes["records"]==RECORDS_SHA,"eval_sha_exact":hashes["eval"]==EVAL_SHA,"cards_sha_exact":hashes["cards"]==CARDS_SHA,"twenty_items_exact":len(items)==20,"sixty_records_exact":len(records)==60,"three_arms_exact":len(arms)==3,"forty_blinded_pairs_exact":len(pairs)==40,"r21d_semantic_evaluator_saturated":saturated}
 summary={"protocol":PROTOCOL,"mode":"post_r21d_frozen_output_mechanical_diagnostic","hashes":hashes,"arms":arms,"semantic_counts_reproduced":semantics,"healthy_counts":{k:sum(x["healthy"] for x in v) for k,v in by.items()},"router_exact_counts":{k:sum(x["router_exact"] is True for x in v) for k,v in by.items() if v[0]["style"]=="router"},"router_misses":route_misses,"lexical_failure_components":dict(component_fail),"lexical_core_failure_positions":dict(core_fail),"gates":gates,"diagnosis":"R21D_LEXICAL_EVALUATOR_SATURATED_AND_ROUTER_COVERAGE_INCOMPLETE" if all(gates.values()) else "R21D_R1_AUDIT_PREFLIGHT_FAILED","human_review_status":"PENDING_40_BLINDED_PAIRWISE_JUDGMENTS","scientific_status":"R21D_R1_MEASUREMENT_FAILURE_LOCALIZED" if all(gates.values()) else "R21D_R1_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Post-output diagnostic on the frozen R21d records. It localizes measurement and routing defects and prepares blinded review, but cannot repair R21d into confirmation, establish an 8B ordering, validate personalization, demonstrate weight learning or moving-fibre superiority, certify safety, or justify deployment."}
 (out/"audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n");print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if all(gates.values()) else 2
if __name__=="__main__":raise SystemExit(main())

