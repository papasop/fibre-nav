#!/usr/bin/env python3
"""Prospective ResNet-18/CIFAR-10 moving-fibre dual-scaling confirmation v4.1b."""
from __future__ import annotations

import argparse, hashlib, json, math, random, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROTOCOL = "CNER_RESNET18_CIFAR10_MOVING_FIBRE_DUAL_SCALING_CONFIRM_V4_1B"
SEEDS = list(range(68726, 68742))
RADII = [0.08, 0.04, 0.02, 0.01]
ANCHORS = 16
CHART_DIM = 8
ADAPTER_DIM = 8
MAX_STEPS = 48
TARGET_REDUCTION = 0.03
RIDGE = 1e-7
RCOND = 1e-9


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def layout(dim=512):
    sizes = [ADAPTER_DIM*dim, ADAPTER_DIM, dim*ADAPTER_DIM, dim, 10*dim, 10]
    cuts=np.cumsum([0]+sizes)
    return sizes,cuts


def init_theta(seed, device):
    seed_all(seed); dim=512; _,c=layout(dim)
    g=torch.Generator(device=device).manual_seed(seed+7001)
    parts=[
      torch.randn(ADAPTER_DIM,dim,generator=g,device=device)*0.01,
      torch.zeros(ADAPTER_DIM,device=device),
      torch.randn(dim,ADAPTER_DIM,generator=g,device=device)*0.01,
      torch.zeros(dim,device=device),
      torch.randn(10,dim,generator=g,device=device)*0.02,
      torch.zeros(10,device=device)]
    return torch.cat([x.reshape(-1) for x in parts])


def unpack(theta):
    _,c=layout(); d=512
    return (theta[c[0]:c[1]].reshape(ADAPTER_DIM,d), theta[c[1]:c[2]],
            theta[c[2]:c[3]].reshape(d,ADAPTER_DIM), theta[c[3]:c[4]],
            theta[c[4]:c[5]].reshape(10,d), theta[c[5]:c[6]])


def logits(theta,x):
    wd,bd,wu,bu,wh,bh=unpack(theta)
    z=x + torch.tanh(x@wd.T+bd)@wu.T+bu
    return z@wh.T+bh


def response(theta,x):
    z=logits(theta,x); return (z-z.mean(-1,keepdim=True)).reshape(-1)


def loss(theta,x,y): return F.cross_entropy(logits(theta,x),y)


def grad_loss(theta,x,y):
    z=theta.detach().requires_grad_(True)
    return torch.autograd.grad(loss(z,x,y),z)[0].detach()


def jacobian(theta,x):
    z=theta.detach().requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda q: response(q,x),z,vectorize=True).detach()


def rowspace(J):
    _,s,vh=torch.linalg.svd(J,full_matrices=False)
    rank=int((s>s.max().clamp_min(1e-30)*RCOND).sum())
    return vh[:rank].T,rank,s


def project(v,row): return v-row@(row.T@v) if row.numel() else v


def orthonormalize(B,row):
    for _ in range(2):
        B=project(B,row); B,_=torch.linalg.qr(B,mode="reduced")
    return B


def align(B,ref):
    u,_,vh=torch.linalg.svd(B.T@ref,full_matrices=False)
    return B@(u@vh)


def initial_chart(theta,J,dev_x,dev_y,seed):
    row,rank,s=rowspace(J); g=grad_loss(theta,dev_x,dev_y); cols=[]
    q=project(-g,row); cols.append(q/q.norm().clamp_min(1e-30))
    gen=torch.Generator(device=theta.device).manual_seed(seed+92001)
    while len(cols)<CHART_DIM:
        q=project(torch.randn(theta.numel(),generator=gen,device=theta.device,dtype=theta.dtype),row)
        for b in cols:q-=b*torch.dot(b,q)
        if q.norm()>1e-10: cols.append(q/q.norm())
    B=orthonormalize(torch.stack(cols,1),row)
    return B,row,{"rank":rank,"null_dimension":theta.numel()-rank,
      "kernel_residual":float(torch.linalg.norm(J@B)/torch.linalg.norm(J).clamp_min(1e-30))}


def transported_chart(J,previous,arm,seed,step):
    row,rank,_=rowspace(J); candidate=previous
    if arm=="transport_shuffled":
        gen=torch.Generator(device=J.device).manual_seed(seed*100000+step)
        perm=torch.randperm(previous.shape[0],generator=gen,device=J.device)
        signs=torch.where(torch.rand(previous.shape[0],generator=gen,device=J.device)>0.5,1.,-1.).to(J.dtype)
        candidate=previous[perm]*signs[:,None]
    B=orthonormalize(candidate,row); B=align(B,previous)
    sv=torch.linalg.svdvals(previous.T@B).clamp(0,1)
    return B,row,{"rank":rank,"null_dimension":J.shape[1]-rank,
      "kernel_residual":float(torch.linalg.norm(J@B)/torch.linalg.norm(J).clamp_min(1e-30)),
      "max_principal_angle":float(torch.acos(sv).max())}


def retract(trial,target,anchor_x):
    theta=trial.detach(); total=torch.zeros_like(theta); initial=None
    for _ in range(2):
        e=response(theta,anchor_x)-target
        if initial is None: initial=float(e.norm())
        J=jacobian(theta,anchor_x)
        A=J@J.T+RIDGE*torch.eye(J.shape[0],device=J.device,dtype=J.dtype)
        corr=-J.T@torch.linalg.solve(A,e); theta=(theta+corr).detach(); total+=corr
    rel=float((response(theta,anchor_x)-target).norm()/target.norm().clamp_min(1e-30))
    return theta,{"initial_response_error":initial,"final_relative_response_error":rel,
                  "correction_norm":float(total.norm())}


def kl_cost(trial,retracted,x):
    a=F.log_softmax(logits(trial,x),-1); b=F.softmax(logits(retracted,x),-1)
    kl=F.kl_div(a,b,reduction="batchmean")
    return float(torch.sqrt((2*kl).clamp_min(0)))


def fit_head(theta,x,y):
    z=theta.detach().clone().requires_grad_(True); opt=torch.optim.AdamW([z],lr=3e-3,weight_decay=1e-4)
    for _ in range(140):
        idx=torch.randint(0,len(x),(min(256,len(x)),),device=x.device)
        v=loss(z,x[idx],y[idx]); opt.zero_grad(); v.backward(); opt.step()
    return z.detach()


def cosine(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))


def run_path(source,B0,J0,arm,radius,data,seed):
    dev_x,dev_y,anchor_x,metric_x=data
    theta=source.clone(); B=B0.clone(); target=response(source,anchor_x).detach()
    # The complete online construction, including stopping, is dev-only.
    initial_dev=loss(source,dev_x,dev_y).item(); goal_dev=initial_dev*(1-TARGET_REDUCTION)
    costs=[]; residuals=[]; rel_errors=[]; angles=[]
    for step in range(MAX_STEPS):
        J=jacobian(theta,anchor_x)
        if arm=="fixed":
            B=B0; row,rank,_=rowspace(J)
            meta={"rank":rank,"null_dimension":theta.numel()-rank,
                  "kernel_residual":float(torch.linalg.norm(J@B)/torch.linalg.norm(J).clamp_min(1e-30)),
                  "max_principal_angle":0.0}
        else: B,row,meta=transported_chart(J,B,arm,seed,step)
        g=grad_loss(theta,dev_x,dev_y); direction=-(B@(B.T@g)); direction/=direction.norm().clamp_min(1e-30)
        trial=theta+radius*direction
        new_theta,ret=retract(trial,target,anchor_x)
        costs.append(kl_cost(trial,new_theta,metric_x)); residuals.append(meta["kernel_residual"])
        rel_errors.append(ret["final_relative_response_error"]); angles.append(meta["max_principal_angle"])
        theta=new_theta
        if loss(theta,dev_x,dev_y).item()<=goal_dev: break
    final_dev=loss(theta,dev_x,dev_y).item()
    dev_reduction=max(0.0,(initial_dev-final_dev)/max(abs(initial_dev),1e-30))
    cost_per_reduction=float(sum(costs))/max(dev_reduction,1e-12)
    record={"arm":arm,"radius":radius,"steps":len(costs),"dev_endpoint_reached":final_dev<=goal_dev,
      "initial_dev_loss":initial_dev,"final_dev_loss":final_dev,"target_dev_loss":goal_dev,
      "achieved_dev_loss_reduction_fraction":dev_reduction,
      "cost_per_unit_loss_reduction":cost_per_reduction,
      "total_realizability_cost":float(sum(costs)),"max_kernel_residual":max(residuals),
      "max_final_relative_response_error":max(rel_errors),"max_principal_angle":max(angles)}
    return record,theta


def slope(paths,arm):
    rows=sorted([r for r in paths if r["arm"]==arm],key=lambda q:q["radius"])
    x=np.log([r["radius"] for r in rows]); y=np.log([max(r["total_realizability_cost"],1e-30) for r in rows])
    a,b=np.polyfit(x,y,1); pred=a*x+b
    r2=1-float(((y-pred)**2).sum()/max(((y-y.mean())**2).sum(),1e-30))
    return float(a),r2


def run_seed(seed,data,device):
    source_x,source_y,dev_x,dev_y,confirm_x,confirm_y,anchor_x,metric_x=data
    theta=fit_head(init_theta(seed,device).double(),source_x,source_y)
    J0=jacobian(theta,anchor_x); B0,_,source_meta=initial_chart(theta,J0,dev_x,dev_y,seed)
    shared=(dev_x,dev_y,anchor_x,metric_x)
    paths=[]; frozen=[]
    for a in ("moving","fixed","transport_shuffled"):
        for h in RADII:
            record,endpoint=run_path(theta,B0,J0,a,h,shared,seed)
            paths.append(record); frozen.append(endpoint)
    # The first confirm access occurs only after all 12 paths are frozen.
    initial_confirm=loss(theta,confirm_x,confirm_y).item()
    for record,endpoint in zip(paths,frozen):
        final_confirm=loss(endpoint,confirm_x,confirm_y).item()
        record["initial_confirm_loss"]=initial_confirm
        record["final_confirm_loss"]=final_confirm
        record["heldout_confirm_reduction_fraction"]=(initial_confirm-final_confirm)/max(abs(initial_confirm),1e-30)
    g_dev=grad_loss(theta,dev_x,dev_y); g_confirm=grad_loss(theta,confirm_x,confirm_y)
    p_dev=B0@(B0.T@g_dev); p_confirm=B0@(B0.T@g_confirm)
    transfer_diagnostic={"raw_gradient_cosine":cosine(g_dev,g_confirm),
      "projected_gradient_cosine":cosine(p_dev,p_confirm),
      "dev_direction_confirm_directional_derivative":float(torch.dot(g_confirm,-p_dev/p_dev.norm().clamp_min(1e-30))),
      "confirm_access_boundary":"First confirm access occurs after all 12 paths are frozen; confirm never selects a direction, step, endpoint, radius, arm, or geometric gate."}
    fits={a:{"alpha":slope(paths,a)[0],"r2":slope(paths,a)[1]} for a in ("moving","fixed","transport_shuffled")}
    at_min={r["arm"]:r for r in paths if r["radius"]==min(RADII)}
    ratios={"fixed_over_moving":at_min["fixed"]["total_realizability_cost"]/max(at_min["moving"]["total_realizability_cost"],1e-30)}
    moving_rows=[r for r in paths if r["arm"]=="moving"]
    fixed_rows=[r for r in paths if r["arm"]=="fixed"]
    shuffled_rows=[r for r in paths if r["arm"]=="transport_shuffled"]
    moving_fixed_eligible=all(r["dev_endpoint_reached"] and r["max_final_relative_response_error"]<=2e-3 for r in moving_rows+fixed_rows)
    transport_step_ratios=[(MAX_STEPS+1 if not s["dev_endpoint_reached"] else s["steps"])/m["steps"] for m,s in zip(moving_rows,shuffled_rows)]
    transport_progress_ratios=[m["achieved_dev_loss_reduction_fraction"]/max(s["achieved_dev_loss_reduction_fraction"],1e-12) for m,s in zip(moving_rows,shuffled_rows)]
    transport={"moving_dev_endpoint_count":sum(r["dev_endpoint_reached"] for r in moving_rows),
      "shuffled_dev_endpoint_count":sum(r["dev_endpoint_reached"] for r in shuffled_rows),
      "moving_faster_at_all_radii":all(m["dev_endpoint_reached"] and (not s["dev_endpoint_reached"] or m["steps"]<s["steps"]) for m,s in zip(moving_rows,shuffled_rows)),
      "median_censored_step_ratio":float(np.median(transport_step_ratios)),
      "median_progress_ratio":float(np.median(transport_progress_ratios))}
    gates={"moving_fixed_eligible":moving_fixed_eligible,"moving_alpha":fits["moving"]["alpha"]>=0.40,
      "alpha_gap":fits["moving"]["alpha"]-fits["fixed"]["alpha"]>=0.40,
      "fixed_cost_ratio":ratios["fixed_over_moving"]>=2.0,
      "transport_efficiency":transport["moving_faster_at_all_radii"] and transport["median_censored_step_ratio"]>=2.0,
      "transport_progress":transport["median_progress_ratio"]>=1.5}
    heldout={"projected_gradient_cosine_gate":transfer_diagnostic["projected_gradient_cosine"]>=0.95,
      "moving_confirm_nonworsening":all(r["heldout_confirm_reduction_fraction"]>=0 for r in moving_rows)}
    return {"seed":seed,"source":source_meta,"gradient_transfer_diagnostic":transfer_diagnostic,
            "fits":fits,"smallest_radius_ratios":ratios,
            "transport_comparison":transport,"geometric_gates":gates,
            "heldout_generalization_gates":heldout,
            "geometric_candidate":all(gates.values()),"heldout_generalization_pass":all(heldout.values()),"paths":paths}


def prepare_data(device):
    tf=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=datasets.CIFAR10("/content/data",train=True,download=True,transform=tf)
    test=datasets.CIFAR10("/content/data",train=False,download=True,transform=tf)
    weights=models.ResNet18_Weights.IMAGENET1K_V1; net=models.resnet18(weights=weights); net.fc=torch.nn.Identity(); net.to(device).eval()
    def cache(ds,idx):
        out=[]; yy=[]
        with torch.no_grad():
            for x,y in DataLoader(Subset(ds,idx),batch_size=128,num_workers=2,pin_memory=True): out.append(net(x.to(device)).cpu()); yy.append(y)
        return torch.cat(out).double().to(device),torch.cat(yy).to(device)
    rng=np.random.default_rng(41001); tr=rng.permutation(len(train)); te=rng.permutation(len(test))
    sx,sy=cache(train,tr[:2048]); dx,dy=cache(train,tr[2048:2560]); ax,_=cache(train,tr[2560:2576]); mx,_=cache(train,tr[2576:2640]); cx,cy=cache(test,te[:512])
    return sx,sy,dx,dy,cx,cy,ax,mx


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="cner_resnet18_cifar10_moving_fibre_dual_scaling_confirm_v4_1b_results"); args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"[preflight] {device=}")
    data=prepare_data(device); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,s in enumerate(SEEDS,1):
        print(f"[v4.1b seed {i}/{len(SEEDS)}] {s}",flush=True); row=run_seed(s,data,device); rows.append(row)
        (out/f"seed_{s}.json").write_text(json.dumps(row,indent=2))
    candidates=sum(r["geometric_candidate"] for r in rows)
    heldout=sum(r["heldout_generalization_pass"] for r in rows)
    passed=candidates>=12
    report={"protocol":PROTOCOL,"scientific_status":"RESNET_MOVING_FIBRE_V41B_DUAL_SCALING_CONFIRMED" if passed else "RESNET_MOVING_FIBRE_V41B_DUAL_SCALING_NOT_CONFIRMED",
      "seeds":16,"geometric_candidate_seeds":candidates,"required_geometric_candidates":12,
      "heldout_generalization_pass_seeds":heldout,"confirmation_gate_pass":passed,"per_seed":rows,
      "frozen_primary_claim":"At a common dev-defined endpoint, current-kernel transport has a realizability-cost exponent at least 0.40 larger than a source-fixed chart, a smallest-radius fixed/moving cost ratio at least 2, and superior shuffled-transport efficiency/progress.",
      "causal_boundary":"All 12 paths per seed are generated and frozen using dev data before the first confirm access. Held-out generalization is reported separately and cannot change the geometric confirmation decision.",
      "claim_boundary":"Sixteen-new-seed prospective confirmation in a trainable terminal residual adapter plus complete classifier over frozen ImageNet ResNet-18 features; not full layer4/backbone transport, F16 action ordering, arbitrary-path global variation, or a universal learning law."}
    frozen_protocol={"protocol":PROTOCOL,"seeds":SEEDS,"radii":RADII,"anchors":ANCHORS,
      "chart_dim":CHART_DIM,"maximum_steps":MAX_STEPS,"dev_loss_reduction_fraction":TARGET_REDUCTION,
      "required_geometric_candidates":12,
      "geometric_thresholds":{"maximum_response_retraction_error":2e-3,"minimum_moving_alpha":0.40,
        "minimum_alpha_gap":0.40,"minimum_smallest_radius_cost_ratio":2.0,
        "minimum_censored_step_ratio":2.0,"minimum_progress_ratio":1.5},
      "heldout_descriptive_thresholds":{"minimum_projected_gradient_cosine":0.95,
        "moving_confirm_nonworsening":True},
      "causal_boundary":"All 12 dev paths freeze before first confirm access."}
    (out/"report.json").write_text(json.dumps(report,indent=2)); (out/"protocol.json").write_text(json.dumps(frozen_protocol,indent=2))
    print("="*96); print(json.dumps({k:v for k,v in report.items() if k!="per_seed"},indent=2))
    shutil.make_archive(str(out),"zip",out.parent,out.name); print(f"Results ZIP: {out}.zip")

if __name__=="__main__": main()
