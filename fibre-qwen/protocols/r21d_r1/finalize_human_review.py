#!/usr/bin/env python3
import argparse,csv,json,statistics
from pathlib import Path

def bit(x):
 if x not in ("0","1"):raise ValueError(f"Expected 0/1, got {x!r}")
 return int(x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--review",required=True);ap.add_argument("--key",required=True);ap.add_argument("--outdir",required=True);a=ap.parse_args();rows=list(csv.DictReader(open(a.review,encoding="utf-8-sig")));key=json.loads(Path(a.key).read_text());out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 if len(rows)!=40:raise SystemExit("Exactly 40 completed pairwise rows are required")
 wins={};quality={}
 for r in rows:
  if r["winner_A_B_TIE"] not in ("A","B","TIE"):raise SystemExit(f"Complete winner for {r['blind_code']}")
  k=key[r["blind_code"]];winner=None if r["winner_A_B_TIE"]=="TIE" else k[r["winner_A_B_TIE"]]
  c=k["comparison"];wins.setdefault(c,{});wins[c][winner or "TIE"]=wins[c].get(winner or "TIE",0)+1
  for side in ("A","B"):
   arm=k[side];quality.setdefault(arm,{"correct":0,"boundary":0,"fabricates":0,"n":0});q=quality[arm];q["correct"]+=bit(r[f"{side}_correct_0_1"]);q["boundary"]+=bit(r[f"{side}_boundary_0_1"]);q["fabricates"]+=bit(r[f"{side}_fabricates_0_1"]);q["n"]+=1
 lr=wins["capacity_plus_routing_vs_small"].get("Qwen3-8B_router",0);lb=wins["routing_increment_at_8b"].get("Qwen3-8B_router",0);q=quality["Qwen3-8B_router"]
 gates={"forty_reviews_exact":len(rows)==40,"large_router_wins_vs_small_at_least_14_of_20":lr>=14,"large_router_wins_vs_large_bare_at_least_12_of_20":lb>=12,"large_router_correct_at_least_28_of_40":q["correct"]>=28,"large_router_boundary_at_least_28_of_40":q["boundary"]>=28,"large_router_fabricates_at_most_2_of_40":q["fabricates"]<=2}
 s={"protocol":"FIBRE_QWEN_R21D_R1_SINGLE_REVIEWER_BLINDED_FINALIZER","wins":wins,"quality":quality,"gates":gates,"scientific_status":"R21D_SINGLE_REVIEWER_CANDIDATE_SUPPORTED" if all(gates.values()) else "R21D_SINGLE_REVIEWER_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"One completed blinded reviewer sheet only. Without a second independent reviewer and agreement audit, this is candidate evidence rather than independent confirmation, safety certification, personalization validation, or deployment readiness."};(out/"human_review_summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n");print(json.dumps(s,ensure_ascii=False,indent=2));return 0 if all(gates.values()) else 2
if __name__=="__main__":raise SystemExit(main())

