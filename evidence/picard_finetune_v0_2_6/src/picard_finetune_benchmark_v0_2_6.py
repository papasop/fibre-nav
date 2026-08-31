#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, time, urllib.request
from pathlib import Path
import numpy as np

PROTOCOL="GEOMETRIC_INTRINSIC_PICARD_FINETUNE_V0_2_6_REPEATED_TIMING_CONFIRMATORY"
EVAL_SEEDS=[22229,22247,22259,22271,22277]
FROZEN_LR={"adamw":.003,"picard_cached":.12}
TIMING_REPEATS=5
# Frozen before this protocol; no v0.2.6 tuning or target selection.
FROZEN_TARGET=1.6318350233280339
CIFAR_MD5="c58f30108f718f92721af3b95e74349a"
CIFAR_MIRROR="https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

def parse():
 p=argparse.ArgumentParser();p.add_argument("--outdir",default="picard_v0_2_results");p.add_argument("--data-root",default="/content/data");p.add_argument("--device",default="cuda");p.add_argument("--quick",action="store_true");a,u=p.parse_known_args()
 if u:print("[notice] ignored",u,flush=True)
 return a
def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def prepare_archive(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/"cifar-10-python.tar.gz"
 if p.exists() and md5(p)==CIFAR_MD5:return
 print("[prepare] fetching verified CIFAR-10 mirror (excluded from timing)",flush=True)
 urllib.request.urlretrieve(CIFAR_MIRROR,p)
 if md5(p)!=CIFAR_MD5:raise RuntimeError("CIFAR-10 mirror MD5 mismatch")
def seed_all(s,torch):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def sync(torch,d):
 if d.type=="cuda":torch.cuda.synchronize()

def features(torch,tv,d,root,quick):
 tf=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 tr=tv.datasets.CIFAR10(root,train=True,download=True,transform=tf);va=tv.datasets.CIFAR10(root,train=False,download=True,transform=tf)
 ntr,nva=(6000,2000) if quick else (20000,5000);tr=torch.utils.data.Subset(tr,range(ntr));va=torch.utils.data.Subset(va,range(nva))
 net=tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT);net.fc=torch.nn.Identity();net.eval().to(d)
 def ex(ds,label):
  dl=torch.utils.data.DataLoader(ds,batch_size=1024,shuffle=False,num_workers=4,persistent_workers=True,prefetch_factor=4,pin_memory=True);xx=[];yy=[]
  with torch.inference_mode():
   for i,(x,y) in enumerate(dl,1):
    xx.append(net(x.to(d,non_blocking=True)));yy.append(y.to(d,non_blocking=True))
    if i==1 or i%5==0 or i==len(dl):print(f"[features:{label}] {i}/{len(dl)}",flush=True)
  return torch.cat(xx),torch.cat(yy)
 x,y=ex(tr,"train");xv,yv=ex(va,"validation");return x[:-1000],y[:-1000],x[-1000:],y[-1000:],xv,yv

def intrinsic_chart(torch,anchor):
 # Construct and certify the response kernel in float64. Training remains
 # directly in these intrinsic coordinates, so neither arm projects per step.
 anchor=anchor.double();_,s,vh=torch.linalg.svd(anchor,full_matrices=True);tol=s.max()*max(anchor.shape)*torch.finfo(anchor.dtype).eps;rank=int((s>tol).sum())
 T=vh[rank:].T.contiguous();leak=float((anchor@T).abs().max())
 return T,rank,leak
def evalm(torch,z,y,a):
 with torch.no_grad():
  lg=z@a;return float(torch.nn.functional.cross_entropy(lg,y)),float((lg.argmax(1)==y).float().mean())
def grad(torch,z,y,a,wd=1e-3):
 lg=z@a;p=lg.softmax(1);oh=torch.nn.functional.one_hot(y,10).to(lg.dtype);return z.T@(p-oh)/len(y)+wd*a

def train(torch,z,y,zv,yv,kind,lr,seed,d,steps,bs,target,metric,record_trace=True):
 seed_all(seed,torch);a=torch.zeros(z.shape[1],10,device=d,dtype=z.dtype);m=torch.zeros_like(a);v=torch.zeros_like(a);order=torch.randperm(len(y),device=d);timed=0.;hit=None;hit_step=None;trace=[]
 if not (z.device==a.device==metric.device and z.dtype==a.dtype==metric.dtype):
  raise RuntimeError(f"intrinsic state mismatch: z={z.device}/{z.dtype}, a={a.device}/{a.dtype}, metric={metric.device}/{metric.dtype}")
 # The chart and its inverse diagonal metric are frozen. Constructing this
 # preconditioner is setup, not an optimizer step, so it is outside timing.
 preconditioner=metric.reciprocal()[:,None] if kind=="picard_cached" else None
 if preconditioner is not None and not bool(torch.isfinite(preconditioner).all()):
  raise RuntimeError("cached inverse metric is nonfinite")
 # Untimed warm-up kernels on a disposable state.
 for _ in range(20): _=z[:bs]@a
 sync(torch,d)
 for step in range(steps):
  if step*bs+bs>len(order):order=torch.randperm(len(y),device=d)
  ix=order[(step*bs)%max(bs,len(order)-bs):(step*bs)%max(bs,len(order)-bs)+bs]
  sync(torch,d);t=time.perf_counter();g=grad(torch,z[ix],y[ix],a)
  if kind=="adamw":
   m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001);a.addcdiv_(m/(1-.9**(step+1)),(v/(1-.999**(step+1))).sqrt().add_(1e-8),value=-lr)
  elif kind=="picard_cached":
   direction=g*preconditioner;m.mul_(.85).add_(direction,alpha=.15);a.add_(m,alpha=-lr)
  else:raise ValueError(f"unknown optimizer arm: {kind}")
  sync(torch,d);timed+=time.perf_counter()-t
  if (step+1)%100==0 or step==steps-1:
   vl,ac=evalm(torch,zv,yv,a)
   if record_trace:trace.append({"step":step+1,"train_kernel_seconds":timed,"val_loss":vl,"val_acc":ac})
   if hit is None and vl<=target:hit=timed;hit_step=step+1
 vl,ac=evalm(torch,zv,yv,a);return {"kind":kind,"seed":seed,"lr":lr,"time_to_target":hit,"steps_to_target":hit_step,"timed_train_seconds":timed,"final_loss":vl,"final_acc":ac,"trace":trace}

def repeated_train(torch,z,y,zv,yv,kind,lr,seed,d,steps,bs,target,metric,repeats):
 runs=[]
 for j in range(repeats):
  r=train(torch,z,y,zv,yv,kind,lr,seed,d,steps,bs,target,metric,j==0);runs.append(r)
  print(f"[repeat {j+1}/{repeats}] hit={r['time_to_target']} timed={r['timed_train_seconds']:.3f}s",flush=True)
 first=runs[0];all_hit=all(r["time_to_target"] is not None for r in runs);same_steps=len({r["steps_to_target"] for r in runs})==1;loss_span=max(r["final_loss"] for r in runs)-min(r["final_loss"] for r in runs);acc_span=max(r["final_acc"] for r in runs)-min(r["final_acc"] for r in runs);full_times=[r["timed_train_seconds"] for r in runs];hit_times=[r["time_to_target"] for r in runs] if all_hit else []
 return {"kind":kind,"seed":seed,"lr":lr,"time_to_target":statistics.median(hit_times) if all_hit else None,"steps_to_target":first["steps_to_target"] if same_steps else None,"timed_train_seconds":statistics.median(full_times),"aggregate_target_measurement_seconds":sum(hit_times) if all_hit else None,"aggregate_full_measurement_seconds":sum(full_times),"target_time_repeat_cv":statistics.pstdev(hit_times)/statistics.mean(hit_times) if all_hit else None,"full_time_repeat_cv":statistics.pstdev(full_times)/statistics.mean(full_times),"final_loss":first["final_loss"],"final_acc":first["final_acc"],"endpoint_repeat_loss_span":loss_span,"endpoint_repeat_accuracy_span":acc_span,"repeat_hit_steps_identical":same_steps,"timing_repeats":[{"repeat":j+1,"time_to_target":r["time_to_target"],"steps_to_target":r["steps_to_target"],"timed_train_seconds":r["timed_train_seconds"]} for j,r in enumerate(runs)],"trace":first["trace"]}

def main():
 a=parse();import torch,torchvision as tv
 d=torch.device(a.device if torch.cuda.is_available() else "cpu");torch.set_float32_matmul_precision("high")
 prepare_archive(a.data_root);print(f"protocol={PROTOCOL} device={d}",flush=True);x,y,xc,yc,xv,yv=features(torch,tv,d,a.data_root,a.quick)
 # Fixed random down-map, then exact structured response kernel.
 gen=torch.Generator(device=d).manual_seed(7001);D=torch.randn(512,24,generator=gen,device=d)/math.sqrt(512);z=x@D;zc=xc@D;zv=xv@D
 T,rank,leak=intrinsic_chart(torch,zc[:4]);z=z.double()@T;zc=zc.double()@T;zv=zv.double()@T;metric=(z.square().mean(0)+1e-3).detach()
 steps=500 if a.quick else 30000;bs=256;repeats=2 if a.quick else TIMING_REPEATS
 if not (z.dtype==zc.dtype==zv.dtype==metric.dtype==torch.float64 and z.device==zc.device==zv.device==metric.device and z.device.type==d.type):
  raise RuntimeError("float64 intrinsic chart preflight failed")
 if not math.isfinite(leak) or leak>1e-10:raise RuntimeError(f"response-kernel preflight leakage failed: {leak}")
 print("[preflight] float64 dtype/device/leakage passed; exercising both optimizer arms",flush=True)
 for smoke_kind,smoke_lr in (("adamw",FROZEN_LR["adamw"]),("picard_cached",FROZEN_LR["picard_cached"])):
  smoke=train(torch,z,y,zc,yc,smoke_kind,smoke_lr,11939,d,2,bs,-math.inf,metric,False)
  if not (math.isfinite(smoke["final_loss"]) and math.isfinite(smoke["final_acc"])):raise RuntimeError(f"{smoke_kind} smoke step nonfinite")
 print("[preflight] AdamW and cached-Picard smoke steps passed",flush=True)
 target=FROZEN_TARGET;best=FROZEN_LR.copy();frozen_config={"source":"v0.2.5 fair-tuning protocol","learning_rates":best,"target":target,"timing_repeats":repeats,"selection_or_tuning_in_v0_2_6":False}
 print(f"[freeze] no tuning; lr={best} target={target:.6f} repeats={repeats}",flush=True)
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);records=[]
 for i,s in enumerate(EVAL_SEEDS[:2] if a.quick else EVAL_SEEDS,1):
  for kind in ("adamw","picard_cached"):
   lr=best[kind]
   print(f"[eval {i}] seed={s} {kind}",flush=True);r=repeated_train(torch,z,y,zv,yv,kind,lr,s,d,steps,bs,target,metric,repeats);records.append(r);print(f"[done] median_hit={r['time_to_target']} median_timed={r['timed_train_seconds']:.3f}s aggregate_hit={r['aggregate_target_measurement_seconds']} loss={r['final_loss']:.6f} acc={r['final_acc']:.4f}",flush=True)
   (out/f"{kind}_{s}.json").write_text(json.dumps(r,indent=2)+"\n")
 pairs=[]
 for s in (EVAL_SEEDS[:2] if a.quick else EVAL_SEEDS):
  aa=next(r for r in records if r["seed"]==s and r["kind"]=="adamw");pp=next(r for r in records if r["seed"]==s and r["kind"]=="picard_cached");valid=aa["time_to_target"] is not None and pp["time_to_target"] is not None
  pairs.append({"seed":s,"valid":valid,"speedup_fraction":(aa["time_to_target"]-pp["time_to_target"])/aa["time_to_target"] if valid else None,"step_reduction_fraction":(aa["steps_to_target"]-pp["steps_to_target"])/aa["steps_to_target"] if valid and aa["steps_to_target"] is not None and pp["steps_to_target"] is not None else None,"fixed_budget_speedup_fraction":(aa["timed_train_seconds"]-pp["timed_train_seconds"])/aa["timed_train_seconds"],"per_step_kernel_seconds":{"adamw":aa["timed_train_seconds"]/steps,"picard_cached":pp["timed_train_seconds"]/steps},"aggregate_target_measurement_seconds":{"adamw":aa["aggregate_target_measurement_seconds"],"picard_cached":pp["aggregate_target_measurement_seconds"]},"final_loss_delta":pp["final_loss"]-aa["final_loss"],"accuracy_delta":pp["final_acc"]-aa["final_acc"]})
 full=not a.quick;valid=all(p["valid"] for p in pairs);med=statistics.median(p["speedup_fraction"] for p in pairs) if valid else None;duration=statistics.median(r["timed_train_seconds"] for r in records);adam_hit=statistics.median(r["time_to_target"] for r in records if r["kind"]=="adamw" and r["time_to_target"] is not None) if valid else None
 fixed=[p["fixed_budget_speedup_fraction"] for p in pairs] if full else []
 fixed_med=statistics.median(fixed) if fixed else None
 step_med=statistics.median(p["step_reduction_fraction"] for p in pairs) if valid and all(p["step_reduction_fraction"] is not None for p in pairs) else None
 loss_delta_med=statistics.median(p["final_loss_delta"] for p in pairs)
 aggregate_adam=statistics.median(r["aggregate_target_measurement_seconds"] for r in records if r["kind"]=="adamw" and r["aggregate_target_measurement_seconds"] is not None) if valid else None;repeat_exact=all(r["repeat_hit_steps_identical"] and r["endpoint_repeat_loss_span"]<=1e-12 and r["endpoint_repeat_accuracy_span"]<=1e-12 for r in records)
 gates={"five_new_frozen_eval_seeds":full,"no_v0_2_6_tuning_or_target_selection":not frozen_config["selection_or_tuning_in_v0_2_6"],"five_timing_repeats_per_seed_arm":full and repeats==5,"repeat_trajectories_numerically_identical":repeat_exact,"all_repeat_timing_cv_at_most_10pct":valid and all(r["target_time_repeat_cv"]<=.10 and r["full_time_repeat_cv"]<=.10 for r in records),"all_both_arms_reach_strict_target":valid,"median_time_to_equal_loss_speedup_at_least_10pct":valid and med>=.10,"fixed_budget_speedup_at_least_10pct":full and fixed_med>=.10,"accuracy_noninferior_0_5pp":all(p["accuracy_delta"]>=-.005 for p in pairs),"median_endpoint_loss_delta_at_most_0_002":loss_delta_med<=.002,"every_endpoint_loss_delta_at_most_0_003":all(p["final_loss_delta"]<=.003 for p in pairs),"median_aggregate_adamw_target_measurement_at_least_2s":valid and aggregate_adam>=2,"median_full_train_repeat_measurement_at_least_10s":statistics.median(r["aggregate_full_measurement_seconds"] for r in records)>=10,"float64_response_leakage_at_most_1e_10":leak<=1e-10};passed=all(gates.values())
 summary={"protocol":PROTOCOL,"mode":"quick_nonclaim" if a.quick else "repeated_timing_confirmatory_new_seed_evaluation","intrinsic_dimension":T.shape[1],"response_rank":rank,"float64_response_leakage":leak,"frozen_config":frozen_config,"pairs":pairs,"median_time_to_equal_loss_speedup_fraction":med,"median_step_reduction_fraction":step_med,"fixed_budget_speedup_fractions":fixed,"median_fixed_budget_speedup_fraction":fixed_med,"median_endpoint_loss_delta":loss_delta_med,"median_timed_train_seconds_per_repeat":duration,"median_adamw_target_seconds_per_repeat":adam_hit,"median_aggregate_adamw_target_measurement_seconds":aggregate_adam,"gates":gates,"scientific_status":"PICARD_V0_2_6_REPEATED_TIMING_DUAL_10PCT_SPEEDUP_SUPPORTED" if passed else "PICARD_V0_2_6_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"No retuning: v0.2.5 learning rates and target are frozen; each new-seed arm repeats the identical trajectory five times and uses median synchronized kernel timing. Frozen-feature ResNet-18/CIFAR-10 only, not a universal optimizer claim."}
 (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
