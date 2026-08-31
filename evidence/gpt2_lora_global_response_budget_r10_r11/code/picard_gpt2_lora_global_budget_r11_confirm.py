#!/usr/bin/env python3
"""Frozen untouched-seed confirmation of the r10 global-budget candidate."""
from __future__ import annotations
import argparse,json,math,statistics,time
from pathlib import Path
import picard_gpt2_lora_dynamic_r7 as core
import picard_gpt2_lora_global_budget_r10 as r10

PROTOCOL="GPT2_LORA_GLOBAL_RESPONSE_BUDGET_PARETO_V0_3_R11_CONFIRMATORY"
SEEDS=(27211,27217,27229,27241,27253)
BUDGETS=r10.BUDGETS;KERNELS=r10.KERNELS

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--outdir",default="global_budget_r11_results");ap.add_argument("--data-root",default="data");ap.add_argument("--steps",type=int,default=180);ap.add_argument("--warm-steps",type=int,default=50);ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--seq-len",type=int,default=64);ap.add_argument("--chart-dim",type=int,default=24);ap.add_argument("--lora-rank",type=int,default=4);ap.add_argument("--layers",type=int,default=2);ap.add_argument("--lr",type=float,default=0.035);ap.add_argument("--warm-lr",type=float,default=0.02);ap.add_argument("--max-backtracks",type=int,default=14);ap.add_argument("--eval-interval",type=int,default=10);ap.add_argument("--val-blocks",type=int,default=32);ap.add_argument("--quick",action="store_true");args=ap.parse_args()
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device);seeds=SEEDS[:1] if args.quick else SEEDS
    if args.quick:args.steps=40;args.warm_steps=10;args.val_blocks=16
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);raw=core.get_data(Path(args.data_root)/"tinyshakespeare.txt");train,val=core.make_blocks(raw,args.seq_len,torch);print(f"protocol={PROTOCOL} device={device} seeds={len(seeds)} budgets={BUDGETS}",flush=True);records=[];started=time.time()
    for si,seed in enumerate(seeds,1):
        core.seed_all(seed,torch);cfg=GPT2Config(vocab_size=256,n_positions=args.seq_len,n_ctx=args.seq_len,n_embd=256,n_layer=6,n_head=8,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.);model=GPT2LMHeadModel(cfg).to(device);coord,_=core.inject_chart_lora(model,args.chart_dim,args.lora_rank,args.layers,torch);model.to(device)
        batches=core.frozen_indices(len(train[0]),args.warm_steps+args.steps,args.batch_size,seed+11,torch);ids=[torch.arange(0,args.batch_size),torch.arange(args.batch_size,2*args.batch_size)];anchors=[(train[0],train[1],i) for i in ids];warm=core.warm_start(model,coord,train[0],train[1],batches,device,args.warm_steps,args.warm_lr,torch);print(f"[seed {si}/{len(seeds)}] {seed} warm start complete",flush=True)
        for budget in BUDGETS:
            for kernel in KERNELS:records.append(r10.run_arm(model,coord,warm,kernel,budget,train,val,batches,anchors,args,device,torch,seed,out))
    by={(r["seed"],r["kernel"],r["global_response_budget"]):r for r in records};pairs=[]
    for budget in BUDGETS:
        contrasts=[by[s,"source",budget]["final_validation_loss"]-by[s,"current",budget]["final_validation_loss"] for s in seeds];pairs.append({"budget":budget,"source_minus_current_loss_by_seed":contrasts,"median_source_minus_current_loss":statistics.median(contrasts),"current_positive_seeds":sum(x>0 for x in contrasts),"supports_frozen_gate":statistics.median(contrasts)>0 and sum(x>0 for x in contrasts)>=4})
    numerical={"five_untouched_confirmation_seeds":len(seeds)==5,"all_runs_finite":all(math.isfinite(r["final_validation_loss"]) for r in records),"all_global_budgets_respected":all(r["max_global_response_drift"]<=r["global_response_budget"]*(1+1e-8) for r in records),"float64_projectors_eligible":max(r["max_projector_leakage_at_construction"] for r in records)<=1e-10,"projector_idempotence_at_most_1e_10":max(r["max_projector_idempotence"] for r in records)<=1e-10,"response_rank_constant":all(len(r["response_ranks"])==1 for r in records)}
    supporting=sum(p["supports_frozen_gate"] for p in pairs);passed=(not args.quick) and all(numerical.values()) and supporting>=3
    summary={"protocol":PROTOCOL,"mode":"quick_nonclaim" if args.quick else "untouched_seed_confirmatory","development_reference":"R10_GLOBAL_BUDGET_PARETO_DIAGNOSTIC_COMPLETE","frozen_configuration":{"budgets":BUDGETS,"steps":180,"warm_steps":50,"chart_dim":24,"lora_rank":4,"layers":2,"lr":0.035,"warm_lr":0.02,"max_backtracks":14},"seeds":seeds,"pairs":pairs,"supporting_budget_count":supporting,"median_loss":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["final_validation_loss"] for s in seeds) for b in BUDGETS for k in KERNELS},"median_max_drift":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["max_global_response_drift"] for s in seeds) for b in BUDGETS for k in KERNELS},"median_zero_step_fraction":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["zero_step_fraction"] for s in seeds) for b in BUDGETS for k in KERNELS},"gates":{**numerical,"at_least_three_of_four_budgets_support":supporting>=3},"scientific_status":"R11_CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_CONFIRMED" if passed else ("R11_QUICK_NONCLAIM" if args.quick else "R11_CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_INCONCLUSIVE_FAIL_CLOSED"),"wall_seconds":time.time()-started,"claim_boundary":"Confirmation within a deterministic compact randomly initialized GPT-2 byte language model and shared 24-dimensional LoRA chart. It establishes a current-versus-source kernel Pareto ordering under the four frozen global response balls only; it is not pretrained GPT-2, AdamW, speed, semantic transfer, universal optimizer superiority, or a global Picard theorem."}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return 0 if passed or args.quick else 2
if __name__=="__main__":raise SystemExit(main())
