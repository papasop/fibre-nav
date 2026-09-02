#!/usr/bin/env python3
import argparse, json, math, re, statistics
from pathlib import Path

PROTOCOL = "FIBRE_QWEN_DETERMINISTIC_ROUTER_ORACLE_R21B_R2_DIAGNOSTIC"
MODEL = "Qwen/Qwen3-0.6B"

def norm(s):
    return re.sub(r"[\s`*_，。；：、,.!?！？()（）]+", "", s.casefold())

def route(prompt, cards):
    """Auditable priority router. No eval id or answer text is visible here."""
    q = norm(prompt)
    hits = []
    for c in cards:
        matched = [p for p in c["patterns"] if norm(p) in q]
        score = max([len(norm(p)) for p in matched] or [0])
        if score:
            hits.append((score, c["priority"], c["id"], matched))
    hits.sort(key=lambda x: (-x[0], x[1], x[2]))
    if not hits:
        return "assistant", {"reason": "fallback", "candidates": []}
    return hits[0][2], {"reason": "longest_exact_pattern_then_priority", "candidates": hits}

def semantic(text, contract):
    z = norm(text)
    groups = contract["required_groups"]
    group_pass = [any(norm(term) in z for term in group) for group in groups]
    forbidden = [term for term in contract.get("forbidden", []) if norm(term) in z]
    return all(group_pass) and not forbidden, {"required_group_pass": group_pass, "forbidden_hits": forbidden}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True); ap.add_argument("--long", required=True)
    ap.add_argument("--eval", required=True); ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(x) for x in Path(a.eval).read_text().splitlines() if x.strip()]
    cards = json.loads(Path(a.cards).read_text()); cmap = {c["id"]: c for c in cards}
    long = Path(a.long).read_text(); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    def sys_prompt(arm, row):
        if arm == "bare": return "", [], {}
        if arm == "long_constitution": return long, [], {}
        if arm == "deterministic_router": cid, audit = route(row["prompt"], cards)
        elif arm == "retrieval_oracle": cid, audit = row["expected_card"], {"reason":"declared_oracle_upper_bound"}
        else: raise ValueError(arm)
        c = cmap[cid]
        s = ("你是严谨研究助手。仅依据所给研究规则回答，不得编造实验数字。"
             "输出四个完整句子：结论；两个理由；适用边界。不要复述规则。\n"
             f"研究规则：{c['rule']}\n必须覆盖的事实：{c['facts']}\n禁止主张：{c['prohibited']}")
        return s, [cid], audit

    def prompt(arm, row):
        s, ids, audit = sys_prompt(arm, row)
        msg = ([{"role":"system","content":s}] if s else []) + [{"role":"user","content":row["prompt"]}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False), ids, audit

    def mean_logp(p, answer):
        n = tok(p, return_tensors="pt").input_ids.shape[1]
        x = tok(p + answer + tok.eos_token, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            lp = torch.log_softmax(model(x).logits[:, :-1].float(), -1).gather(-1, x[:,1:].unsqueeze(-1)).squeeze(-1)
        return float(lp[:, max(0,n-1):].mean())

    records=[]; by={}; arms=("bare","long_constitution","deterministic_router","retrieval_oracle")
    for arm in arms:
        by[arm]=[]
        for row in rows:
            p, ids, audit = prompt(arm,row)
            margin = mean_logp(p,row["preferred"])-mean_logp(p,row["rejected"])
            x=tok(p,return_tensors="pt").to(model.device)
            with torch.inference_mode():
                y=model.generate(**x,min_new_tokens=36,max_new_tokens=144,do_sample=False,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
            ids_out=y[0,x.input_ids.shape[1]:].tolist(); text=tok.decode(ids_out,skip_special_tokens=True).strip()
            grams=[tuple(ids_out[i:i+4]) for i in range(max(0,len(ids_out)-3))]
            health=(12<=len(ids_out)<=144 and len(set(ids_out))/max(1,len(ids_out))>=.28 and
                    (len(grams)-len(set(grams)))/max(1,len(grams))<=.18)
            sem,sem_audit=semantic(text,row["contract"])
            routed_correct=(not ids) or ids[0]==row["expected_card"]
            rec={"id":row["id"],"arm":arm,"expected_card":row["expected_card"],"routed_cards":ids,
                 "routed_correct":routed_correct,"router_audit":audit,"margin":margin,
                 "prefers_declared":margin>0,"generation":text,"health":health,"semantic":sem,
                 "semantic_audit":sem_audit}
            records.append(rec);by[arm].append(rec)
            print(f"[{arm}] {row['id']} route={ids or '-'} correct={routed_correct} margin={margin:.3f} semantic={sem} health={health}",flush=True)

    base=by["bare"]; det=by["deterministic_router"]; oracle=by["retrieval_oracle"]
    deltas=[d["margin"]-b["margin"] for b,d in zip(base,det)]
    counts=lambda v:{"declared":sum(x["prefers_declared"] for x in v),"semantic":sum(x["semantic"] for x in v),"healthy":sum(x["health"] for x in v)}
    gates={
      "ten_same_development_items":len(rows)==10,
      "all_records_finite":all(math.isfinite(x["margin"]) for x in records),
      "deterministic_router_exact_10_of_10":sum(x["routed_correct"] for x in det)==10,
      "oracle_semantic_at_least_7_of_10":sum(x["semantic"] for x in oracle)>=7,
      "deterministic_semantic_at_least_7_of_10":sum(x["semantic"] for x in det)>=7,
      "deterministic_healthy_at_least_9_of_10":sum(x["health"] for x in det)>=9,
      "deterministic_prefers_declared_at_least_8_of_10":sum(x["prefers_declared"] for x in det)>=8,
      "median_margin_delta_positive":statistics.median(deltas)>0,
    }
    ok=all(gates.values())
    if sum(x["routed_correct"] for x in det)<10: diagnosis="ROUTER_REMAINS_LIMITING"
    elif sum(x["semantic"] for x in oracle)<7: diagnosis="GENERATOR_OR_CARD_REPRESENTATION_LIMITING"
    elif sum(x["semantic"] for x in det)<7: diagnosis="ROUTER_GENERATOR_INTERACTION_LIMITING"
    else: diagnosis="ROUTED_PROFILE_CANDIDATE_NOMINATED"
    summary={"protocol":PROTOCOL,"mode":"same_item_post_r21b_r1_router_oracle_diagnostic","model":MODEL,
      "items":len(rows),"median_deterministic_margin_delta":statistics.median(deltas),
      "deterministic_improved_items":sum(x>0 for x in deltas),"counts":{k:counts(v) for k,v in by.items()},
      "router_exact_count":sum(x["routed_correct"] for x in det),"diagnosis":diagnosis,"gates":gates,
      "scientific_status":"R21B_R2_ROUTED_PROFILE_CANDIDATE_SUPPORTED" if ok else "R21B_R2_INCONCLUSIVE_FAIL_CLOSED",
      "claim_boundary":"Same ten authored development items reused after R21a/R21b. The oracle arm diagnoses retrieval versus generation; it is not an implementable baseline, untouched confirmation, weight learning, moving-fibre evidence, broad personalization, safety certification, or deployment readiness."}
    (out/"records.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
    (out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (out/"rule_cards.json").write_text(json.dumps(cards,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if ok else 2

if __name__=="__main__": raise SystemExit(main())
