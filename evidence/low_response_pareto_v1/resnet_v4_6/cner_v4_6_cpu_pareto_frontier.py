#!/usr/bin/env python3
"""Prospective CPU audit of the low-response stability-plasticity Pareto frontier."""
from __future__ import annotations
import argparse, csv, json, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import cner_resnet18_cifar10_real_optimizer_path_v4_4_r1 as base
import cner_resnet18_cifar10_stability_plasticity_v4_5_r2 as r2

PROTOCOL="CNER_RESNET18_CIFAR10_LOW_RESPONSE_PARETO_V4_6_CPU"
SEEDS=list(range(85741,85749))
REQUIRED=6
ADAPT_STEPS=32
AUDIT_EVERY=4
BUDGET_FRACTIONS=[0.02,0.05,0.10,0.20]
ALPHAS=[0.0,0.015625,0.03125,0.0625,0.125,0.25,0.5,0.75,1.0]
MIN_NODE_WIN_FRACTION=5/7
MIN_POSITIVE_NODE_FRACTION=5/7
MIN_ROTATION=0.01
HARD_LIMIT=7200
ARMS=("actual","current_kernel","source_kernel","far_time_kernel","permuted_kernel")
CONTROLS=("actual","source_kernel","far_time_kernel","permuted_kernel")

def arm_curve(theta,direction,anchors,x_cal,y_cal,x_eval,y_eval):
    base_cal=r2.loss_value(theta,x_cal,y_cal); base_eval=r2.loss_value(theta,x_eval,y_eval)
    points=[]
    for alpha in ALPHAS:
        q=theta+alpha*direction
        points.append({"alpha":alpha,"response_cost":r2.response_cost(theta,q,anchors),
                       "calibration_utility":base_cal-r2.loss_value(q,x_cal,y_cal),
                       "heldout_utility":base_eval-r2.loss_value(q,x_eval,y_eval)})
    return points

def choose_on_calibration(curve,budget):
    feasible=[p for p in curve if p["response_cost"]<=budget+1e-12]
    # Held-out utility is never inspected. Smaller alpha breaks exact ties.
    return max(feasible,key=lambda p:(p["calibration_utility"],-p["alpha"]))

def auc(values):
    return float(np.trapz(np.asarray(values,dtype=float),np.asarray(BUDGET_FRACTIONS,dtype=float))/(BUDGET_FRACTIONS[-1]-BUDGET_FRACTIONS[0]))

def run_seed(seed,data,device,started):
    sx,sy,anchors,dx,dy,xcal,ycal,xeval,yeval=[z.to(device) for z in data]
    theta=base.init_theta(seed,device); base.fit_source(theta,sx,sy,seed)
    source_acc=base.accuracy(theta,sx[:1600],sy[:1600])
    pre_eval_loss=r2.loss_value(theta,xeval,yeval)
    j0,row0,sigma0,rank0=base.jacobian(theta,anchors)
    opt=torch.optim.AdamW([theta],lr=base.ADAPT_LR,weight_decay=base.ADAPT_WEIGHT_DECAY)
    schedule=base.batch_indices(seed+2,len(dx),ADAPT_STEPS,192); snaps=[]
    for step,idx_cpu in enumerate(schedule):
        if time.time()-started>HARD_LIMIT: raise TimeoutError("two-hour hard limit exceeded")
        idx=idx_cpu.to(device); before=theta.detach().clone()
        loss=F.cross_entropy(base.forward(theta,dx[idx]),dy[idx]); opt.zero_grad(); loss.backward(); opt.step()
        delta=theta.detach()-before
        if step%AUDIT_EVERY==0:
            jt,rowt,sigmat,rankt=base.jacobian(before,anchors)
            snaps.append({"step":step,"theta":before,"delta":delta.detach().clone(),"row":rowt,
                          "rank":rankt,"rotation":base.row_rotation(rowt,row0)})
    post_eval_loss=r2.loss_value(theta,xeval,yeval)
    nodes=[]; n=len(snaps)
    for i,s in enumerate(snaps):
        th,delta,rowt=s["theta"],s["delta"],s["row"]; norm=delta.norm()
        far=snaps[(i+n//2)%n]["row"]; perm=r2.permuted_row(rowt,seed*1000+s["step"])
        directions={"actual":delta,
                    "current_kernel":r2.normalized(r2.kernel_project(rowt,delta),norm),
                    "source_kernel":r2.normalized(r2.kernel_project(row0,delta),norm),
                    "far_time_kernel":r2.normalized(r2.kernel_project(far,delta),norm),
                    "permuted_kernel":r2.normalized(r2.kernel_project(perm,delta),norm)}
        th=th.double(); a64=anchors.double(); xc=xcal.double(); xe=xeval.double()
        curves={k:arm_curve(th,v.double(),a64,xc,ycal,xe,yeval) for k,v in directions.items()}
        actual_full=next(p for p in curves["actual"] if p["alpha"]==1.0)
        budgets=[f*actual_full["response_cost"] for f in BUDGET_FRACTIONS]
        selected={arm:[choose_on_calibration(curves[arm],b) for b in budgets] for arm in ARMS}
        arm_aucs={arm:auc([p["heldout_utility"] for p in selected[arm]]) for arm in ARMS}
        node={"step":s["step"],"row_space_rotation":s["rotation"],"update_norm":float(norm.item()),
              "actual_full_response_cost":actual_full["response_cost"]}
        for arm in ARMS:
            node[f"{arm}_heldout_auc"]=arm_aucs[arm]
            node[f"{arm}_selected_alphas"]=[p["alpha"] for p in selected[arm]]
            node[f"{arm}_heldout_utilities"]=[p["heldout_utility"] for p in selected[arm]]
            node[f"{arm}_response_costs"]=[p["response_cost"] for p in selected[arm]]
        for control in CONTROLS:
            node[f"current_minus_{control}_auc"]=arm_aucs["current_kernel"]-arm_aucs[control]
        nodes.append(node)
    primary=[x for x in nodes if x["step"]>0]
    vals=lambda key:np.asarray([x[key] for x in primary],dtype=float)
    summary={"source_accuracy":source_acc,"pre_heldout_loss":pre_eval_loss,"post_heldout_loss":post_eval_loss,
             "heldout_loss_gain":pre_eval_loss-post_eval_loss,"audited_nodes_total":len(nodes),
             "primary_noninitial_nodes":len(primary),"maximum_row_space_rotation":max(x["row_space_rotation"] for x in nodes),
             "fraction_current_positive_auc":float(np.mean(vals("current_kernel_heldout_auc")>0))}
    for control in CONTROLS:
        diff=vals(f"current_minus_{control}_auc")
        summary[f"fraction_current_auc_beats_{control}"]=float(np.mean(diff>0))
        summary[f"median_current_minus_{control}_auc"]=float(np.median(diff))
    gates={"source_accuracy":source_acc>=base.MIN_SOURCE_ACC,
           "real_adaptation_improves_heldout":pre_eval_loss-post_eval_loss>0,
           "nontrivial_rotation":summary["maximum_row_space_rotation"]>=MIN_ROTATION,
           "current_positive_low_budget_auc":summary["fraction_current_positive_auc"]>=MIN_POSITIVE_NODE_FRACTION}
    for control in CONTROLS:
        gates[f"current_low_budget_auc_beats_{control}"]=(summary[f"fraction_current_auc_beats_{control}"]>=MIN_NODE_WIN_FRACTION and summary[f"median_current_minus_{control}_auc"]>0)
    return {"seed":seed,"summary":summary,"gates":gates,"supported":all(gates.values()),"nodes":nodes}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="cner_v4_6_cpu_pareto_results")
    args,unknown=p.parse_known_args()
    if unknown: print("[notice] ignored notebook arguments:",unknown,flush=True)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); device=torch.device("cpu")
    torch.set_num_threads(max(1,min(8,torch.get_num_threads()))); base.SOURCE_STEPS=60
    started=time.time(); print(f"[preflight] {PROTOCOL} seeds={SEEDS} device=cpu",flush=True)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    src=transforms.Compose([transforms.Resize(112),transforms.ToTensor(),norm])
    shift=transforms.Compose([transforms.Resize(112),transforms.GaussianBlur(7,1.4),transforms.ToTensor(),norm])
    root=Path("data"); train=datasets.CIFAR10(root,train=True,download=True,transform=src); shifted=datasets.CIFAR10(root,train=False,download=True,transform=shift)
    backbone=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); backbone.fc=torch.nn.Identity(); backbone.to(device).eval()
    for q in backbone.parameters():q.requires_grad_(False)
    sx,sy=base.cache(backbone,train,range(1600),device); anchors,_=base.cache(backbone,train,range(1600,1612),device,12)
    dx,dy=base.cache(backbone,shifted,range(600),device); xcal,ycal=base.cache(backbone,shifted,range(600,900),device); xeval,yeval=base.cache(backbone,shifted,range(900,1200),device)
    del backbone; data=(sx,sy,anchors,dx,dy,xcal,ycal,xeval,yeval); records=[]
    for i,seed in enumerate(SEEDS,1):
        print(f"[seed {i}/8] {seed}",flush=True); rec=run_seed(seed,data,device,started); records.append(rec)
        (out/f"seed_{seed}.json").write_text(json.dumps(rec,indent=2)); print(json.dumps(rec["summary"],indent=2),flush=True)
    supporting=sum(x["supported"] for x in records); decision="LOW_RESPONSE_PARETO_ADVANTAGE_SUPPORTED" if supporting>=REQUIRED else "LOW_RESPONSE_PARETO_ADVANTAGE_NOT_SUPPORTED"
    report={"protocol":PROTOCOL,"prospective":True,"decision":decision,"supporting_seeds":supporting,"required":REQUIRED,"attempted":len(records),"elapsed_seconds":time.time()-started,"records":records,"claim_boundary":"CPU reduced-workload local low-response Pareto audit in the declared frozen-backbone ResNet-18 adapter+head chart; not A100/full-backbone/cross-architecture confirmation or a universal law."}
    protocol={"protocol":PROTOCOL,"prospective":True,"prior_seed_cohorts_excluded":[83741,list(range(84741,84749))],"seeds":SEEDS,"required":REQUIRED,"budget_fractions":BUDGET_FRACTIONS,"alpha_grid":ALPHAS,"alpha_selected_on":"300-example calibration split only","heldout_split":"disjoint 300 examples; never used for alpha selection","primary_metric":"mean trapezoidal held-out utility over normalized low-response budgets","controls":list(CONTROLS),"initial_node_excluded":True,"min_node_win_fraction":MIN_NODE_WIN_FRACTION,"hard_limit_seconds":HARD_LIMIT}
    (out/"report.json").write_text(json.dumps(report,indent=2)); (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    rows=[(r["seed"],n) for r in records for n in r["nodes"]]
    with (out/"node_metrics.csv").open("w",newline="") as f:
        scalar=[k for k,v in rows[0][1].items() if not isinstance(v,list)]; w=csv.DictWriter(f,fieldnames=["seed"]+scalar); w.writeheader()
        for seed,node in rows:w.writerow({"seed":seed,**{k:node[k] for k in scalar}})
    archive=shutil.make_archive(str(out),"zip",out.parent,out.name); print("="*88); print(json.dumps(report,indent=2),flush=True); print("RESULT ZIP:",archive,flush=True)

if __name__=="__main__":main()
