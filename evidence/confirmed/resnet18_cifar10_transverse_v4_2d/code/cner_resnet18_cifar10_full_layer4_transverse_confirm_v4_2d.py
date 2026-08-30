#!/usr/bin/env python3
"""Prospective float64 confirmation of transverse amplification v4.2d."""
from __future__ import annotations

import argparse, json, math, random, shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp, vmap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

PROTOCOL="CNER_RESNET18_CIFAR10_FULL_LAYER4_TRANSVERSE_CONFIRM_V4_2D"
SEEDS=[76742,76743,76744,76745,76746,76747,76748,76749,
       76750,76751,76752,76753,76754,76755,76756,76757]
RADII=[0.004,0.002,0.001]
MICRO_SCALES=[1.0,0.25,0.0625,0.015625]
ANCHORS=8
CHART_DIM=8
MAX_STEPS=24
TARGET_REDUCTION=0.03
RIDGE=1e-5
CG_STEPS=48
CG_TOL=2e-3
KERNEL_GATE=1e-3


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class FunctionalLayer4:
    def __init__(self,model):
        self.model=model
        self.names=[n for n,p in model.named_parameters() if n.startswith("layer4.") or n.startswith("fc.")]
        params=OrderedDict((n,p.detach()) for n,p in model.named_parameters())
        self.frozen=OrderedDict((n,p) for n,p in params.items() if n not in self.names)
        self.buffers=OrderedDict((n,b.detach()) for n,b in model.named_buffers())
        self.shapes=[params[n].shape for n in self.names]
        self.sizes=[params[n].numel() for n in self.names]
        self.cuts=np.cumsum([0]+self.sizes).tolist()
        self.theta=torch.cat([params[n].reshape(-1) for n in self.names]).detach()

    def unpack(self,theta):
        train=OrderedDict((n,theta[self.cuts[i]:self.cuts[i+1]].view(s)) for i,(n,s) in enumerate(zip(self.names,self.shapes)))
        params=OrderedDict(self.frozen); params.update(train)
        return params

    def logits(self,theta,x):
        return functional_call(self.model,(self.unpack(theta),self.buffers),(x,))

    def response(self,theta,x):
        z=self.logits(theta,x); return (z-z.mean(-1,keepdim=True)).reshape(-1)

    def loss(self,theta,x,y): return F.cross_entropy(self.logits(theta,x),y)


def grad_loss(fun,theta,x,y):
    q=theta.detach().requires_grad_(True)
    return torch.autograd.grad(fun.loss(q,x,y),q)[0].detach()


def response_ops(fun,theta,anchor_x):
    f=lambda q: fun.response(q,anchor_x)
    value,pullback=vjp(f,theta)
    def J(V):
        if V.ndim==1: return jvp(f,(theta,),(V,))[1]
        return vmap(lambda v:jvp(f,(theta,),(v,))[1])(V.T).T
    def JT(U):
        if U.ndim==1: return pullback(U)[0]
        return vmap(lambda u:pullback(u)[0])(U.T).T
    return value,J,JT


def block_cg(A,B,maxiter=CG_STEPS,tol=CG_TOL):
    if B.ndim==1: B=B[:,None]
    X=torch.zeros_like(B); R=B.clone(); P=R.clone()
    rr=(R*R).sum(0); bnorm=torch.sqrt((B*B).sum(0)).clamp_min(1e-30)
    used=0
    for used in range(1,maxiter+1):
        AP=A(P); den=(P*AP).sum(0).clamp_min(1e-30); alpha=rr/den
        X=X+P*alpha; R=R-AP*alpha
        new=(R*R).sum(0)
        if float((torch.sqrt(new)/bnorm).max())<=tol: rr=new; break
        beta=new/rr.clamp_min(1e-30); P=R+P*beta; rr=new
    rel=torch.sqrt(rr)/bnorm
    return X,{"iterations":used,"maximum_relative_residual":float(rel.max())}


def project(fun,theta,anchor_x,V):
    _,J,JT=response_ops(fun,theta,anchor_x)
    Y=J(V)
    A=lambda U:J(JT(U))+RIDGE*U
    X,cg=block_cg(A,Y)
    out=V-JT(X)
    residual=float(torch.linalg.norm(J(out))/torch.linalg.norm(out).clamp_min(1e-30))
    return out,{**cg,"kernel_residual":residual}


def orthonormalize(fun,theta,anchor_x,B,max_rounds=4,target=5e-4):
    metas=[]
    actual=math.inf
    for _ in range(max_rounds):
        B,m=project(fun,theta,anchor_x,B); metas.append(m)
        B=torch.linalg.qr(B,mode="reduced").Q
        _,J,_=response_ops(fun,theta,anchor_x)
        actual=float(torch.linalg.norm(J(B))/torch.linalg.norm(B).clamp_min(1e-30))
        if actual<=target: break
    return B,{"kernel_residual":actual,"projection_rounds":len(metas),
              "maximum_cg_relative_residual":max(m["maximum_relative_residual"] for m in metas),
              "maximum_cg_iterations":max(m["iterations"] for m in metas)}


def align(B,ref):
    u,_,vh=torch.linalg.svd(B.T@ref,full_matrices=False)
    return B@(u@vh)


def initial_chart(fun,theta,anchor_x,dev_x,dev_y,seed):
    g=grad_loss(fun,theta,dev_x,dev_y)
    gen=torch.Generator(device=theta.device).manual_seed(seed+42001)
    raw=torch.randn(theta.numel(),CHART_DIM,generator=gen,device=theta.device,dtype=theta.dtype)
    raw[:,0]=-g/g.norm().clamp_min(1e-30)
    return orthonormalize(fun,theta,anchor_x,raw,max_rounds=8,target=5e-4)


def retract(fun,trial,target,anchor_x):
    theta=trial.detach(); initial=None; cgmax=0.0; total=torch.zeros_like(theta)
    for _ in range(2):
        value,J,JT=response_ops(fun,theta,anchor_x); error=value-target
        if initial is None: initial=float(error.norm())
        A=lambda U:J(JT(U))+RIDGE*U
        x,cg=block_cg(A,-error); correction=JT(x[:,0])
        theta=(theta+correction).detach(); total+=correction
        cgmax=max(cgmax,cg["maximum_relative_residual"])
    final=float((fun.response(theta,anchor_x)-target).norm()/target.norm().clamp_min(1e-30))
    return theta,{"initial_response_error":initial,"final_relative_response_error":final,
      "correction_norm":float(total.norm()),"maximum_cg_relative_residual":cgmax}


def cost_measures(fun,trial,retracted,x):
    with torch.no_grad():
        logp=F.log_softmax(fun.logits(trial,x).double(),-1)
        logq=F.log_softmax(fun.logits(retracted,x).double(),-1); q=logq.exp()
        kl=(q*(logq-logp)).sum(-1).mean().clamp_min(0)
        return {"kl64":float(kl),"kl64_distance":float(torch.sqrt(2*kl))}


def participation(coeff):
    power=coeff.square(); return float(power.sum().square()/power.square().sum().clamp_min(1e-30))


def run_path(fun,source,B0,arm,radius,data):
    dev_x,dev_y,anchor_x,metric_x=data
    theta=source.clone(); Bmoving=B0.clone(); target=fun.response(source,anchor_x).detach()
    initial=float(fun.loss(source,dev_x,dev_y)); goal=initial*(1-TARGET_REDUCTION)
    costs=[]; leaks=[]; corrections=[]; kernels=[]; cg=[]; response_errors=[]; angles=[]; step_rows=[]
    for step in range(MAX_STEPS):
        old=Bmoving; Bmoving,meta=orthonormalize(fun,theta,anchor_x,Bmoving,max_rounds=4,target=KERNEL_GATE); Bmoving=align(Bmoving,old)
        sv=torch.linalg.svdvals(old.T@Bmoving).clamp(0,1); angle=float(torch.acos(sv).max()); angles.append(angle)
        _,J,_=response_ops(fun,theta,anchor_x)
        moving_chart_res=float(torch.linalg.norm(J(Bmoving))/torch.linalg.norm(Bmoving).clamp_min(1e-30))
        fixed_chart_res=float(torch.linalg.norm(J(B0))/torch.linalg.norm(B0).clamp_min(1e-30))
        g=grad_loss(fun,theta,dev_x,dev_y); cm=Bmoving.T@g; cf=B0.T@g
        dm=-(Bmoving@cm); df=-(B0@cf)
        if float(dm.norm())<1e-12 or float(df.norm())<1e-12: break
        dm=dm/dm.norm(); df=df/df.norm()
        jdm=J(dm); jdf=J(df); rho_m=float(torch.linalg.norm(jdm)); rho_f=float(torch.linalg.norm(jdf))
        direction_cos=float(torch.dot(dm,df).clamp(-1,1))
        delta=df-dm; delta_norm=float(delta.norm()); jdelta=J(delta); jdelta_norm=float(jdelta.norm())
        jdelta_reference=jdf-jdm
        jvp_linearity_error=float((jdelta-jdelta_reference).norm()/jdelta_reference.norm().clamp_min(1e-30))
        transverse_gain=jdelta_norm/max(delta_norm,1e-30)
        gain_contrast=transverse_gain/max(rho_m,1e-30)
        finite_ladder=[]
        with torch.no_grad():
            for scale in MICRO_SCALES:
                eps=radius*scale
                finite_difference=float((fun.response(theta+eps*df,anchor_x)-fun.response(theta+eps*dm,anchor_x)).norm())
                finite_ladder.append({"scale":scale,"epsilon":eps,
                  "response_difference":finite_difference,
                  "finite_over_linear_prediction":finite_difference/max(eps*jdelta_norm,1e-300)})
        active={"step":step,"moving_chart_kernel_residual":moving_chart_res,
          "fixed_chart_kernel_residual":fixed_chart_res,
          "chart_residual_ratio_fixed_over_moving":fixed_chart_res/max(moving_chart_res,1e-30),
          "moving_direction_response_residual":rho_m,"fixed_direction_response_residual":rho_f,
          "active_residual_ratio_fixed_over_moving":rho_f/max(rho_m,1e-30),
          "moving_fixed_direction_cosine":direction_cos,
          "transverse_parameter_norm":delta_norm,"transverse_response_norm":jdelta_norm,
          "transverse_response_gain":transverse_gain,"transverse_over_moving_gain":gain_contrast,
          "jvp_linearity_relative_error":jvp_linearity_error,
          "finite_radius_ladder":finite_ladder,
          "moving_coefficient_participation":participation(cm),"fixed_coefficient_participation":participation(cf)}
        B=Bmoving if arm=="moving" else B0; direction=dm if arm=="moving" else df
        kernels.append(moving_chart_res if arm=="moving" else fixed_chart_res)
        cg.append(meta["maximum_cg_relative_residual"])
        trial=theta+radius*direction
        preleak=float((fun.response(trial,anchor_x)-target).norm()/target.norm().clamp_min(1e-30))
        new,ret=retract(fun,trial,target,anchor_x); measure=cost_measures(fun,trial,new,metric_x)
        costs.append(measure["kl64_distance"]); leaks.append(preleak); corrections.append(ret["correction_norm"])
        step_rows.append({**active,"kl64":measure["kl64"],"kl64_distance":measure["kl64_distance"],
          "pre_retraction_relative_response_leakage":preleak,"retraction_correction_norm":ret["correction_norm"]})
        response_errors.append(ret["final_relative_response_error"]); cg.append(ret["maximum_cg_relative_residual"])
        theta=new
        if float(fun.loss(theta,dev_x,dev_y))<=goal: break
    final=float(fun.loss(theta,dev_x,dev_y)); reduction=(initial-final)/max(abs(initial),1e-30)
    return {"arm":arm,"radius":radius,"steps":len(costs),"endpoint_reached":final<=goal,
      "initial_dev_loss":initial,"final_dev_loss":final,"target_dev_loss":goal,
      "dev_reduction_fraction":reduction,"total_realizability_cost":float(sum(costs)),
      "total_kl64_distance":float(sum(costs)),"total_pre_retraction_response_leakage":float(sum(leaks)),
      "total_retraction_correction_norm":float(sum(corrections)),"step_cost_diagnostics":step_rows,
      "max_kernel_residual":max(kernels,default=math.inf),
      "max_cg_relative_residual":max(cg,default=math.inf),
      "max_response_retraction_error":max(response_errors,default=math.inf),
      "max_principal_angle":max(angles,default=0.0),"active_direction_diagnostics":step_rows}


def slope(paths,arm,field):
    rows=sorted((p for p in paths if p["arm"]==arm),key=lambda p:p["radius"])
    x=np.log([p["radius"] for p in rows]); y=np.log([max(p[field],1e-30) for p in rows])
    a,b=np.polyfit(x,y,1); pred=a*x+b
    r2=1-float(((y-pred)**2).sum()/max(((y-y.mean())**2).sum(),1e-30))
    return {"alpha":float(a),"r2":r2}


def fit_source(model,loader,device,seed):
    seed_all(seed)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.layer4.parameters(): p.requires_grad_(True)
    for p in model.fc.parameters(): p.requires_grad_(True)
    model.eval(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    it=iter(loader)
    for _ in range(60):
        try:x,y=next(it)
        except StopIteration: it=iter(loader); x,y=next(it)
        x=x.to(device); y=y.to(device); z=model(x); v=F.cross_entropy(z,y)
        opt.zero_grad(); v.backward(); opt.step()
    return model


def run_seed(seed,base_state,data,device):
    source_loader,dev,confirm,anchor_x,metric_x=data
    model=models.resnet18(weights=None); model.fc=torch.nn.Linear(model.fc.in_features,10)
    model.load_state_dict(base_state); model.to(device); fit_source(model,source_loader,device,seed)
    # Source fitting remains economical float32; every geometric and response
    # audit operation below is performed after a complete float64 conversion.
    model=model.double()
    fun=FunctionalLayer4(model); theta=fun.theta
    dev_x,dev_y=(dev[0].double(),dev[1]); confirm_x,confirm_y=(confirm[0].double(),confirm[1])
    anchor_x=anchor_x.double(); metric_x=metric_x.double()
    B0,source_meta=initial_chart(fun,theta,anchor_x,dev_x,dev_y,seed)
    paths=[run_path(fun,theta,B0,"moving",h,(dev_x,dev_y,anchor_x,metric_x)) for h in RADII]
    # First confirm access occurs after all three paths are frozen.
    source_confirm=float(fun.loss(theta,confirm_x,confirm_y))
    zero_floor=cost_measures(fun,theta,theta,metric_x)
    moving=paths
    min_moving=next(p for p in moving if p["radius"]==min(RADII))["steps"]
    active=[q for p in moving for q in p["active_direction_diagnostics"]]
    transverse=[q for q in active if q["transverse_parameter_norm"]>=1e-5]
    if not transverse: raise RuntimeError("No nonzero transverse active-direction differences were observed")
    finest=[q["finite_radius_ladder"][-1]["finite_over_linear_prediction"] for q in transverse]
    coarsest=[q["finite_radius_ladder"][0]["finite_over_linear_prediction"] for q in transverse]
    convergence=[abs(math.log(max(a,1e-300)))<=abs(math.log(max(b,1e-300))) for a,b in zip(finest,coarsest)]
    active_summary={"median_chart_residual_ratio_fixed_over_moving":float(np.median([q["chart_residual_ratio_fixed_over_moving"] for q in active])),
      "median_active_residual_ratio_fixed_over_moving":float(np.median([q["active_residual_ratio_fixed_over_moving"] for q in transverse])),
      "median_moving_fixed_direction_cosine":float(np.median([q["moving_fixed_direction_cosine"] for q in transverse])),
      "median_transverse_parameter_norm":float(np.median([q["transverse_parameter_norm"] for q in transverse])),
      "median_transverse_response_gain":float(np.median([q["transverse_response_gain"] for q in transverse])),
      "median_transverse_over_moving_gain":float(np.median([q["transverse_over_moving_gain"] for q in transverse])),
      "median_finest_finite_over_linear_prediction":float(np.median(finest)),
      "median_coarsest_finite_over_linear_prediction":float(np.median(coarsest)),
      "fraction_states_improving_toward_linearity":float(np.mean(convergence)),
      "maximum_jvp_linearity_relative_error":float(max(q["jvp_linearity_relative_error"] for q in transverse)),
      "audited_moving_state_steps":len(active),"audited_nonzero_transverse_steps":len(transverse)}
    gates={"source_chart_feasible":source_meta["kernel_residual"]<=2e-3,
      "moving_numerically_feasible":all(p["max_kernel_residual"]<=2e-3 for p in moving),
      "moving_endpoints":all(p["endpoint_reached"] for p in moving),
      "multistep_identifiable":min_moving>=3,
      "chart_level_separation":active_summary["median_chart_residual_ratio_fixed_over_moving"]>=50,
      "near_collinearity":active_summary["median_moving_fixed_direction_cosine"]>=0.999,
      "active_response_separation":active_summary["median_active_residual_ratio_fixed_over_moving"]>=50,
      "transverse_gain_contrast":active_summary["median_transverse_over_moving_gain"]>=1000,
      "finite_radius_validation":0.9<=active_summary["median_finest_finite_over_linear_prediction"]<=1.1,
      "radius_convergence":active_summary["fraction_states_improving_toward_linearity"]>=0.75,
      "jvp_identity":active_summary["maximum_jvp_linearity_relative_error"]<=1e-8}
    return {"seed":seed,"trainable_parameters":theta.numel(),"source_chart":source_meta,
      "source_confirm_loss_post_freeze":source_confirm,"zero_step_cost_floor":zero_floor,
      "transverse_amplification_summary":active_summary,"gates":gates,
      "candidate":all(gates.values()),"paths":paths}


def prepare(device):
    tf=transforms.Compose([transforms.Resize(112),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=datasets.CIFAR10("/content/data",train=True,download=True,transform=tf)
    test=datasets.CIFAR10("/content/data",train=False,download=True,transform=tf)
    rng=np.random.default_rng(42001); tr=rng.permutation(len(train)); te=rng.permutation(len(test))
    loader=DataLoader(Subset(train,tr[:1024]),batch_size=64,shuffle=True,num_workers=2,pin_memory=True)
    def batch(ds,idx):
        x,y=next(iter(DataLoader(Subset(ds,idx),batch_size=len(idx),num_workers=2)))
        return x.to(device),y.to(device)
    dev=batch(train,tr[1024:1152]); anchor_x,_=batch(train,tr[1152:1160]); metric_x,_=batch(train,tr[1160:1192]); confirm=batch(test,te[:128])
    pretrained=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); pretrained.fc=torch.nn.Linear(pretrained.fc.in_features,10)
    return pretrained.state_dict(),(loader,dev,confirm,anchor_x,metric_x)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="cner_resnet18_cifar10_full_layer4_transverse_confirm_v4_2d_results"); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("v4.2d requires CUDA; use a Colab A100 runtime")
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    print(f"[preflight] device={device} torch={torch.__version__} matrix_free=True audit_dtype=float64 tf32=False",flush=True)
    base,data=prepare(device); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); rows=[]
    for i,seed in enumerate(SEEDS,1):
        print(f"[v4.2d seed {i}/{len(SEEDS)}] {seed}",flush=True)
        row=run_seed(seed,base,data,device); rows.append(row); (out/f"seed_{seed}.json").write_text(json.dumps(row,indent=2))
        torch.cuda.empty_cache()
    count=sum(r["candidate"] for r in rows); passed=count>=12
    report={"protocol":PROTOCOL,"scientific_status":"FULL_LAYER4_TRANSVERSE_AMPLIFICATION_V42D_CONFIRMED" if passed else "FULL_LAYER4_TRANSVERSE_AMPLIFICATION_V42D_NOT_CONFIRMED",
      "seeds":16,"candidate_seeds":count,"required_candidate_seeds":12,"confirmation_gate_pass":passed,"per_seed":rows,
      "primary_hypothesis":"Under the unchanged float64, TF32-off r1 measurement protocol, at least 12 of 16 prospective seeds jointly satisfy JVP additivity, finite-radius convergence, near-collinearity and predeclared transverse response-amplification gates.",
      "method_boundary":"Matrix-free JVP/VJP block-CG projection of an eight-dimensional transported chart; the complete response Jacobian and complete kernel are never constructed.",
      "causal_boundary":"All three moving paths per seed are generated and frozen from dev data before the first confirm access.",
      "claim_boundary":"Sixteen-new-seed prospective float64 confirmation in an eight-dimensional transported chart with trainable ResNet-18 layer4 plus fc on CIFAR-10; not a cost advantage, an asymptotic singular-spectrum theorem, a complete kernel bundle, global variation, LLM transfer, or a universal law."}
    (out/"report.json").write_text(json.dumps(report,indent=2)); (out/"protocol.json").write_text(json.dumps({"protocol":PROTOCOL,"seeds":SEEDS,"radii":RADII,"micro_scales":MICRO_SCALES,"audit_dtype":"float64","tf32":False,"anchors":ANCHORS,"chart_dim":CHART_DIM,"max_steps":MAX_STEPS,"target_reduction":TARGET_REDUCTION,"ridge":RIDGE,"cg_steps":CG_STEPS,"cg_tol":CG_TOL},indent=2))
    print("="*96); print(json.dumps({k:v for k,v in report.items() if k!="per_seed"},indent=2)); shutil.make_archive(str(out),"zip",out.parent,out.name); print(f"Results ZIP: {out}.zip")


if __name__=="__main__": main()
