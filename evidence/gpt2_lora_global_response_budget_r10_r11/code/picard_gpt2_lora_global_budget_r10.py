#!/usr/bin/env python3
"""r10 current-vs-source Pareto audit under matched global response balls."""
from __future__ import annotations
import argparse,json,math,statistics,time
from pathlib import Path
import picard_gpt2_lora_dynamic_r7 as core

PROTOCOL="GPT2_LORA_GLOBAL_RESPONSE_BUDGET_PARETO_V0_3_R10_DEVELOPMENT"
BUDGETS=(5e-5,1e-4,2e-4,5e-4)
KERNELS=("current","source")

def response_values(model,anchors,device,torch):
    with torch.no_grad(): return torch.tensor([float(core.batch_loss(model,xb,yb,idx,device,torch)) for xb,yb,idx in anchors],dtype=torch.float64,device=device)

def arm_name(kernel,budget): return f"{kernel}_B{budget:.0e}".replace("-","m")

def run_arm(model,coord,warm,kernel,budget,train,val,batches,anchors,args,device,torch,seed,out):
    xb,yb=train;xv,yv=val;coord.data.copy_(warm);model.train();J0=core.response_jacobian(model,coord,anchors,device,torch);N0,d0=core.null_basis(J0,torch);R0=response_values(model,anchors,device,torch)
    trace=[];max_drift=max_leak=max_idem=0.;lengths=[];backtracks=[];zero=0;ranks=[];core.sync(torch,device);t0=time.perf_counter()
    for step in range(args.steps):
        J=core.response_jacobian(model,coord,anchors,device,torch)
        if kernel=="current":N,dg=core.null_basis(J,torch)
        else:N,dg=N0,d0
        ranks.append(dg["rank"]);max_leak=max(max_leak,dg["linear_leakage"]);max_idem=max(max_idem,dg["idempotence"])
        loss=core.batch_loss(model,xb,yb,batches[args.warm_steps+step],device,torch);g=core.grad_vec(loss,coord,torch).detach().double();d=-(N@(N.T@g));d=d/d.norm().clamp_min(1e-30)
        old=coord.detach().clone();length=args.lr;accepted=False;used=0;Rnew=response_values(model,anchors,device,torch)
        for bt in range(args.max_backtracks+1):
            coord.data.copy_(old);coord.data.add_(d.to(coord.dtype),alpha=length);Rnew=response_values(model,anchors,device,torch)
            if float((Rnew-R0).norm())<=budget*(1+1e-8):accepted=True;used=bt;break
            length*=0.5
        if not accepted:coord.data.copy_(old);Rnew=response_values(model,anchors,device,torch);length=0.;used=args.max_backtracks+1;zero+=1
        drift=float((Rnew-R0).norm());max_drift=max(max_drift,drift);lengths.append(length);backtracks.append(used)
        if (step+1)%args.eval_interval==0 or step==args.steps-1:
            vl=core.evaluate(model,xv,yv,device,torch,args.val_blocks,args.batch_size);trace.append({"step":step+1,"validation_loss":vl,"global_response_drift":drift,"step_length":length})
            print(f"[{kernel} B={budget:.0e}] step={step+1}/{args.steps} val={vl:.6f} drift={drift:.2e} len={length:.2e}",flush=True)
    core.sync(torch,device);elapsed=time.perf_counter()-t0;name=arm_name(kernel,budget)
    rec={"seed":seed,"arm":name,"kernel":kernel,"global_response_budget":budget,"final_validation_loss":trace[-1]["validation_loss"],"best_validation_loss":min(x["validation_loss"] for x in trace),"max_global_response_drift":max_drift,"budget_residual":budget-max_drift,"median_step_length":statistics.median(lengths),"zero_step_fraction":zero/args.steps,"median_backtracks":statistics.median(backtracks),"max_projector_leakage_at_construction":max_leak,"max_projector_idempotence":max_idem,"response_ranks":sorted(set(ranks)),"timed_seconds":elapsed,"trace":trace}
    (out/f"{name}_{seed}.json").write_text(json.dumps(rec,indent=2)+"\n");return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda");ap.add_argument("--outdir",default="global_budget_r10_results");ap.add_argument("--data-root",default="data");ap.add_argument("--steps",type=int,default=180);ap.add_argument("--warm-steps",type=int,default=50);ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--seq-len",type=int,default=64);ap.add_argument("--chart-dim",type=int,default=24);ap.add_argument("--lora-rank",type=int,default=4);ap.add_argument("--layers",type=int,default=2);ap.add_argument("--lr",type=float,default=0.035);ap.add_argument("--warm-lr",type=float,default=0.02);ap.add_argument("--max-backtracks",type=int,default=14);ap.add_argument("--eval-interval",type=int,default=10);ap.add_argument("--val-blocks",type=int,default=32);ap.add_argument("--quick",action="store_true");args=ap.parse_args()
    import torch
    from transformers import GPT2Config,GPT2LMHeadModel
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device);seeds=core.SEEDS[:1] if args.quick else core.SEEDS
    if args.quick:args.steps=40;args.warm_steps=10;args.val_blocks=16
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);raw=core.get_data(Path(args.data_root)/"tinyshakespeare.txt");train,val=core.make_blocks(raw,args.seq_len,torch);print(f"protocol={PROTOCOL} device={device} seeds={len(seeds)} budgets={BUDGETS}",flush=True);records=[];started=time.time()
    for si,seed in enumerate(seeds,1):
        core.seed_all(seed,torch);cfg=GPT2Config(vocab_size=256,n_positions=args.seq_len,n_ctx=args.seq_len,n_embd=256,n_layer=6,n_head=8,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.);model=GPT2LMHeadModel(cfg).to(device);coord,_=core.inject_chart_lora(model,args.chart_dim,args.lora_rank,args.layers,torch);model.to(device)
        batches=core.frozen_indices(len(train[0]),args.warm_steps+args.steps,args.batch_size,seed+11,torch);ids=[torch.arange(0,args.batch_size),torch.arange(args.batch_size,2*args.batch_size)];anchors=[(train[0],train[1],i) for i in ids];warm=core.warm_start(model,coord,train[0],train[1],batches,device,args.warm_steps,args.warm_lr,torch);print(f"[seed {si}/{len(seeds)}] {seed} warm start complete",flush=True)
        for budget in BUDGETS:
            for kernel in KERNELS:records.append(run_arm(model,coord,warm,kernel,budget,train,val,batches,anchors,args,device,torch,seed,out))
    by={(r["seed"],r["kernel"],r["global_response_budget"]):r for r in records};pairs=[]
    for budget in BUDGETS:
        contrasts=[by[s,"source",budget]["final_validation_loss"]-by[s,"current",budget]["final_validation_loss"] for s in seeds]
        pairs.append({"budget":budget,"source_minus_current_loss_by_seed":contrasts,"median_source_minus_current_loss":statistics.median(contrasts),"current_positive_seeds":sum(x>0 for x in contrasts)})
    numerical={"all_runs_finite":all(math.isfinite(r["final_validation_loss"]) for r in records),"all_global_budgets_respected":all(r["max_global_response_drift"]<=r["global_response_budget"]*(1+1e-8) for r in records),"float64_projectors_eligible":max(r["max_projector_leakage_at_construction"] for r in records)<=1e-10,"projector_idempotence_at_most_1e_10":max(r["max_projector_idempotence"] for r in records)<=1e-10,"response_rank_constant":all(len(r["response_ranks"])==1 for r in records)}
    supporting=sum(p["median_source_minus_current_loss"]>0 and p["current_positive_seeds"]>=4 for p in pairs);signal=supporting>=3
    outcome="CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_CANDIDATE" if signal else "CURRENT_KERNEL_GLOBAL_BUDGET_ADVANTAGE_NOT_SUPPORTED"
    if not all(numerical.values()):outcome="NUMERICALLY_INELIGIBLE"
    summary={"protocol":PROTOCOL,"mode":"quick_nonclaim" if args.quick else "same_seed_global_budget_pareto_development","r7_r9_seed_reuse_intentional":True,"budgets":BUDGETS,"seeds":seeds,"pairs":pairs,"median_loss":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["final_validation_loss"] for s in seeds) for b in BUDGETS for k in KERNELS},"median_max_drift":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["max_global_response_drift"] for s in seeds) for b in BUDGETS for k in KERNELS},"median_zero_step_fraction":{f"{k}_{b:.0e}":statistics.median(by[s,k,b]["zero_step_fraction"] for s in seeds) for b in BUDGETS for k in KERNELS},"numerical_gates":numerical,"supporting_budget_count":supporting,"diagnostic_outcome":outcome,"scientific_status":"R10_GLOBAL_BUDGET_PARETO_DIAGNOSTIC_COMPLETE" if all(numerical.values()) and not args.quick else "R10_QUICK_NONCLAIM_OR_INELIGIBLE","wall_seconds":time.time()-started,"claim_boundary":"Same-seed development Pareto audit under matched global response balls. A positive outcome only freezes a candidate for untouched-seed confirmation."}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return 0 if all(numerical.values()) else 2
if __name__=="__main__":raise SystemExit(main())
