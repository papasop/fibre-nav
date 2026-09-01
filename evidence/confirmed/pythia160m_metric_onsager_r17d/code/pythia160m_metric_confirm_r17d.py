#!/usr/bin/env python3
"""Frozen five-seed confirmation of the R17c metric-Onsager candidate."""
from __future__ import annotations
import argparse, json, math, statistics, subprocess, sys, time
from pathlib import Path

PROTOCOL = "PYTHIA160M_SST2_AGNEWS_METRIC_ONSAGER_R17D_CONFIRMATORY"
SEEDS = (55211, 55219, 55229, 55243, 55259)
BUDGET = 0.004543482202852718
MULTIPLIER = 1.15
ARMS = ("current_metric_m115", "current_projected_adamw", "source_frozen_metric_m100")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="pythia_r17d_results")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir).resolve(); out.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("pythia160m_metric_confirm_r17d_seed.py")
    seeds = SEEDS[:1] if args.quick else SEEDS
    started = time.time(); summaries = []
    for index, seed in enumerate(seeds, 1):
        seed_out = out / f"seed_{seed}"
        cmd = [sys.executable, str(worker), "--device", args.device,
               "--seed", str(seed), "--outdir", str(seed_out)]
        if args.quick: cmd.append("--quick")
        print(f"[seed {index}/{len(seeds)}] {seed}", flush=True)
        code = subprocess.run(cmd).returncode
        path = seed_out / "run_summary.json"
        if not path.exists(): raise RuntimeError(f"seed {seed} produced no summary; exit={code}")
        if code not in (0, 2): raise RuntimeError(f"seed {seed} worker failed; exit={code}")
        summaries.append(json.loads(path.read_text()))

    pairs=[]; all_numeric=True; all_records=[]
    for summary in summaries:
        by={r["arm"]:r for r in summary["records"]}
        cand, adam, source=(by[a] for a in ARMS)
        adam_margin=adam["final_validation_loss"]-cand["final_validation_loss"]
        source_margin=source["final_validation_loss"]-cand["final_validation_loss"]
        accuracy_delta=cand["final_validation_accuracy"]-adam["final_validation_accuracy"]
        numeric=all(summary["numerical_gates"].values())
        pairs.append({
            "seed":summary["seed"],
            "adamw_minus_metric_onsager_loss":adam_margin,
            "source_minus_metric_onsager_loss":source_margin,
            "metric_onsager_minus_adamw_accuracy":accuracy_delta,
            "metric_response_budget_utilization":cand["maximum_global_response_drift"]/BUDGET,
            "worker_numerical_gates_pass":numeric,
            "supports_frozen_gate":numeric and adam_margin>0 and source_margin>0 and accuracy_delta>=-.005,
        })
        all_numeric &= numeric; all_records.extend(summary["records"])
    supporting=sum(p["supports_frozen_gate"] for p in pairs)
    med_adam=statistics.median(p["adamw_minus_metric_onsager_loss"] for p in pairs)
    med_source=statistics.median(p["source_minus_metric_onsager_loss"] for p in pairs)
    med_acc=statistics.median(p["metric_onsager_minus_adamw_accuracy"] for p in pairs)
    gates={
        "five_untouched_confirmation_seeds":len(summaries)==5,
        "frozen_seed_list_exact":tuple(s["seed"] for s in summaries)==SEEDS,
        "frozen_multiplier_1_15_exact":all(s["selected_multiplier"]==MULTIPLIER for s in summaries),
        "three_frozen_arms_exact":all(tuple(s["arms"])==ARMS for s in summaries),
        "one_frozen_budget_exact":all(s["frozen_global_response_budget"]==BUDGET for s in summaries),
        "all_worker_numerical_gates":all_numeric,
        "all_records_finite":all(math.isfinite(r["final_validation_loss"]) for r in all_records),
        "at_least_four_of_five_seeds_support":supporting>=4,
        "median_adamw_minus_metric_onsager_loss_positive":med_adam>0,
        "median_source_minus_metric_onsager_loss_positive":med_source>0,
        "median_accuracy_noninferior_0_5pp":med_acc>=-.005,
    }
    passed=all(gates.values()) and not args.quick
    aggregate={
        "protocol":PROTOCOL,
        "mode":"quick_nonclaim" if args.quick else "untouched_seed_confirmatory",
        "development_reference":"R17C_BUDGET_MATCHED_METRIC_ONSAGER_CANDIDATE_SELECTED",
        "model":"EleutherAI/pythia-160m", "pretrained":True,
        "learning_target_L":"GLUE/SST-2 prompted binary sentiment loss",
        "response_map_R":"four AG News topic-margin coordinates on frozen disjoint inputs",
        "r_l_separation":"different datasets, prompts, labels, verbalizers and declared functionals",
        "frozen_configuration":{"multiplier":MULTIPLIER,"budget":BUDGET,"steps":80,
            "warm_steps":20,"step_norm":.025,"chart_dim":32,"lora_rank":4,"layers":2,"arms":list(ARMS)},
        "seeds":list(seeds), "pairs":pairs, "supporting_seed_count":supporting,
        "medians":{"adamw_minus_metric_onsager_loss":med_adam,
            "source_minus_metric_onsager_loss":med_source,
            "metric_onsager_minus_adamw_accuracy":med_acc},
        "gates":gates,
        "scientific_status":("R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED"
            if passed else ("R17D_QUICK_NONCLAIM" if args.quick else "R17D_METRIC_ONSAGER_INCONCLUSIVE_FAIL_CLOSED")),
        "wall_seconds":time.time()-started,
        "claim_boundary":"Frozen five-seed confirmation within pretrained Pythia-160M, SST-2 learning, an independent AG News response map, one 32-dimensional LoRA chart and one global response budget. It confirms or rejects only the nominated metric-constrained Onsager versus projected AdamW and source-frozen ordering; it is not a continuous-action theorem, universal optimizer result, Principle-R theorem, or physical law.",
    }
    (out/"run_summary.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    print(json.dumps(aggregate,indent=2))
    return 0 if (passed or args.quick) else 2


if __name__=="__main__": raise SystemExit(main())
