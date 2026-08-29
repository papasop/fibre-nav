#!/usr/bin/env python3
"""GPTW-v2: prospective GPT-2/LoRA current-fibre versus source-fixed replay audit."""
from __future__ import annotations
import argparse,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

PROTOCOL="CNER_GPT2_LORA_MOVING_FIBRE_CURRENT_VS_FIXED_V2"
MODEL_IDS=["openai-community/gpt2","gpt2"]
SEEDS=[26742,26743,26744,26745,26746,26747]
REQUIRED_INSTANCES=4; LORA_RANK=2; LAYERS=[10,11]; RESPONSE_DIM=16
AUDIT_RADII=[0.20,0.10,0.05]; PATH_STEP=0.20; INTERIOR_NODES=3; RETRACTION_ROUNDS=2
MAX_CURRENT_KERNEL_RESIDUAL=1e-5; MIN_ACTIVE_RATIO=10.0; MIN_FINE_COST_RATIO=2.0
MIN_CURRENT_COST_SLOPE=1.5; MAX_FIXED_COST_SLOPE=1.4; REQUIRED_NODES=2
MAX_PATH_RESPONSE_ERROR=2e-4; WALL_CLOCK_LIMIT_SECONDS=3300

ANCHORS=["The capital of France is","Water freezes at","A triangle has","The opposite of hot is"]
RESPONSE_TOKENS=[" Paris"," zero"," three"," cold"]
TASK=[
 ("Codeword dax means the color of snow. Snow is"," white"),("Codeword wug means the color of grass. Grass is"," green"),
 ("Codeword zorp means the opposite of down. The answer is"," up"),("Codeword blick means the opposite of night. The answer is"," day"),
 ("Remember: dax maps to white. Therefore dax is"," white"),("Remember: wug maps to green. Therefore wug is"," green"),
 ("Remember: zorp maps to up. Therefore zorp is"," up"),("Remember: blick maps to day. Therefore blick is"," day")]

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class NativeLoRAConv1D(torch.nn.Module):
    def __init__(self,base,rank=2,alpha=4):
        super().__init__(); self.base=base; self.scale=alpha/rank
        for p in base.parameters(): p.requires_grad_(False)
        self.A=torch.nn.Parameter(torch.empty(int(base.weight.shape[0]),rank),requires_grad=False)
        self.B=torch.nn.Parameter(torch.zeros(rank,int(base.weight.shape[1]))); torch.nn.init.normal_(self.A,std=0.02)
    def forward(self,x): return self.base(x)+(x@self.A@self.B)*self.scale

def build(device):
    errors=[]
    for mid in MODEL_IDS:
        try:
            tok=AutoTokenizer.from_pretrained(mid); model=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float32); break
        except Exception as exc: errors.append(f"{mid}: {type(exc).__name__}: {exc}")
    else: raise RuntimeError("GPT-2 load failed:\n"+"\n".join(errors))
    tok.pad_token=tok.eos_token
    for p in model.parameters(): p.requires_grad_(False)
    for layer in LAYERS: model.transformer.h[layer].attn.c_attn=NativeLoRAConv1D(model.transformer.h[layer].attn.c_attn,LORA_RANK,4)
    model.to(device=device,dtype=torch.float64).eval(); params=[model.transformer.h[i].attn.c_attn.B for i in LAYERS]
    return mid,tok,model,params

def flat(params): return torch.cat([p.detach().reshape(-1) for p in params])
def assign(params,v):
    k=0
    with torch.no_grad():
        for p in params: n=p.numel(); p.copy_(v[k:k+n].view_as(p)); k+=n
def grads(y,params,retain=False):
    gs=torch.autograd.grad(y,params,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
def unit(v): return v/v.norm().clamp_min(1e-30)
def token_id(tok,s):
    ids=tok.encode(s,add_special_tokens=False)
    if not ids: raise RuntimeError(f"empty target tokenization: {s!r}")
    return ids[-1]
def logits(model,tok,prompt,device): return model(input_ids=tok(prompt,return_tensors="pt").input_ids.to(device)).logits[0,-1]
def loss(model,tok,device):
    rows=[]
    for prompt,target in TASK:
        rows.append(torch.nn.functional.cross_entropy(logits(model,tok,prompt,device)[None,:],torch.tensor([token_id(tok,target)],device=device)))
    return torch.stack(rows).mean()
def response(model,tok,device):
    vals=[]
    for prompt,target in zip(ANCHORS,RESPONSE_TOKENS):
        z=logits(model,tok,prompt,device); tid=token_id(tok,target); vals.extend([z[tid],z.topk(3).values.mean(),z.mean(),z.std()])
    return torch.stack(vals)[:RESPONSE_DIM]
def jacobian(model,tok,params,device):
    r=response(model,tok,device); return r.detach(),torch.stack([grads(r[i],params,retain=i+1<len(r)) for i in range(len(r))])
def solve_normal(J,error):
    gram=J@J.T; scale=float(torch.trace(gram)/len(gram)); eye=torch.eye(len(gram),device=J.device,dtype=J.dtype)
    return -J.T@torch.linalg.solve(gram+(1e-8*max(scale,1e-12))*eye,error)
def project(v,J): return v+solve_normal(J,J@v)
def direction(model,tok,params,J,device): return unit(project(-grads(loss(model,tok,device),params),J))

def finite_cost(model,tok,params,theta,r_target,J,v,h,device):
    assign(params,theta+h*v); err=response(model,tok,device).detach()-r_target; assign(params,theta)
    corr=solve_normal(J,err)
    return {"radius":h,"active_response_residual":float(err.norm()/r_target.norm().clamp_min(1e-30)),
            "linear_newton_correction_norm":float(corr.norm()),"correction_to_step_ratio":float(corr.norm()/(h*v.norm()).clamp_min(1e-30))}

def retract(model,tok,params,theta,r_source,device):
    q=theta.clone(); trace=[]
    for _ in range(RETRACTION_ROUNDS):
        assign(params,q); r,J=jacobian(model,tok,params,device); err=r-r_source; corr=solve_normal(J,err); q=q+corr
        trace.append({"relative_error_before":float(err.norm()/r_source.norm().clamp_min(1e-30)),"correction_norm":float(corr.norm())})
    assign(params,q); final=float((response(model,tok,device).detach()-r_source).norm()/r_source.norm().clamp_min(1e-30)); return q,trace,final

def slope(rows):
    x=np.log([x["radius"] for x in rows]); y=np.log([max(x["linear_newton_correction_norm"],1e-30) for x in rows])
    p=np.polyfit(x,y,1); pred=np.polyval(p,x); ssr=float(np.sum((y-pred)**2)); sst=float(np.sum((y-y.mean())**2))
    return float(p[0]),float(1-ssr/sst) if sst>0 else 1.0

def run_seed(seed,device):
    seed_all(seed); mid,tok,model,params=build(device); theta=flat(params); assign(params,theta); r_source,J0=jacobian(model,tok,params,device)
    v_fixed=direction(model,tok,params,J0,device); nodes=[]; retractions=[]
    # First advance creates a genuine interior state; source node itself is excluded.
    theta,retr,err=retract(model,tok,params,theta+PATH_STEP*v_fixed,r_source,device); retractions.append({"advance":0,"trace":retr,"final_relative_error":err})
    for node in range(INTERIOR_NODES):
        assign(params,theta); r_node,J=jacobian(model,tok,params,device); v_cur=direction(model,tok,params,J,device)
        cur=[finite_cost(model,tok,params,theta,r_node,J,v_cur,h,device) for h in AUDIT_RADII]
        fix=[finite_cost(model,tok,params,theta,r_node,J,v_fixed,h,device) for h in AUDIT_RADII]
        cs,cr2=slope(cur); fs,fr2=slope(fix); cur_kr=float((J@v_cur).norm()/J.norm().clamp_min(1e-30)); fix_active=float((J@v_fixed).norm()/J.norm().clamp_min(1e-30)); active_ratio=fix_active/max(cur_kr,1e-30)
        fine_ratio=fix[-1]["linear_newton_correction_norm"]/max(cur[-1]["linear_newton_correction_norm"],1e-30)
        angle=float(torch.acos(torch.dot(v_cur,v_fixed).abs().clamp(0,1)))
        gate=bool(cur_kr<=MAX_CURRENT_KERNEL_RESIDUAL and active_ratio>=MIN_ACTIVE_RATIO and fine_ratio>=MIN_FINE_COST_RATIO and
                  cs>=MIN_CURRENT_COST_SLOPE and fs<=MAX_FIXED_COST_SLOPE and all(a["linear_newton_correction_norm"]<b["linear_newton_correction_norm"] for a,b in zip(cur,fix)))
        nodes.append({"node":node+1,"principal_angle_radians":angle,"current_kernel_relative_residual":cur_kr,"fixed_active_relative_residual":fix_active,
          "fixed_to_current_active_ratio":active_ratio,"current_costs":cur,"fixed_costs":fix,"current_cost_slope":cs,"current_cost_r2":cr2,
          "fixed_cost_slope":fs,"fixed_cost_r2":fr2,"fine_fixed_to_current_cost_ratio":fine_ratio,"node_gate":gate})
        theta,retr,err=retract(model,tok,params,theta+PATH_STEP*v_cur,r_source,device); retractions.append({"advance":node+1,"trace":retr,"final_relative_error":err})
    passed=sum(x["node_gate"] for x in nodes); max_path=max(x["final_relative_error"] for x in retractions); supported=passed>=REQUIRED_NODES and max_path<=MAX_PATH_RESPONSE_ERROR
    out={"seed":seed,"model":mid,"nodes":nodes,"retractions":retractions,"passing_nodes":passed,"required_nodes":REQUIRED_NODES,
         "maximum_path_response_error":max_path,"instance_supported":supported}; del model; torch.cuda.empty_cache(); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="cner_gpt2_lora_moving_fibre_current_vs_fixed_v2_results"); args,_=ap.parse_known_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPTW-v2 requires CUDA; select an A100 runtime")
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); rows=[]
    for i,s in enumerate(SEEDS):
        if time.monotonic()-started>WALL_CLOCK_LIMIT_SECONDS: raise TimeoutError("GPTW-v2 exceeded 55-minute hard limit")
        print(f"[GPTW-v2 prospective {i+1}/{len(SEEDS)}] seed={s}",flush=True); row=run_seed(s,device); rows.append(row)
        (out/f"seed_{s}.json").write_text(json.dumps(row,indent=2)); print(json.dumps({"seed":s,"passing_nodes":row["passing_nodes"],"supported":row["instance_supported"]}),flush=True)
    elapsed=time.monotonic()-started; count=sum(x["instance_supported"] for x in rows); passed=count>=REQUIRED_INSTANCES
    report={"protocol":PROTOCOL,"scientific_status":"PROSPECTIVE_GPT2_LORA_MOVING_FIBRE_AUDIT_COMPLETED",
      "decision":"GPT2_LORA_CURRENT_FIBRE_REALIZABILITY_ADVANTAGE_SUPPORTED" if passed else "GPT2_LORA_CURRENT_FIBRE_REALIZABILITY_ADVANTAGE_NOT_SUPPORTED",
      "seeds":SEEDS,"supporting_instances":count,"required_instances":REQUIRED_INSTANCES,"elapsed_seconds":elapsed,"wall_clock_limit_seconds":WALL_CLOCK_LIMIT_SECONDS,
      "frozen_gates":{"audit_radii":AUDIT_RADII,"path_step":PATH_STEP,"interior_nodes":INTERIOR_NODES,"maximum_current_kernel_residual":MAX_CURRENT_KERNEL_RESIDUAL,
        "minimum_fixed_to_current_active_ratio":MIN_ACTIVE_RATIO,"minimum_fine_cost_ratio":MIN_FINE_COST_RATIO,"minimum_current_cost_slope":MIN_CURRENT_COST_SLOPE,
        "maximum_fixed_cost_slope":MAX_FIXED_COST_SLOPE,"required_passing_nodes":REQUIRED_NODES,"maximum_path_response_error":MAX_PATH_RESPONSE_ERROR},
      "instance_summaries":[{"seed":x["seed"],"passing_nodes":x["passing_nodes"],"maximum_path_response_error":x["maximum_path_response_error"],"instance_supported":x["instance_supported"],
         "node_summaries":[{k:n[k] for k in ("node","principal_angle_radians","fixed_to_current_active_ratio","current_cost_slope","fixed_cost_slope","fine_fixed_to_current_cost_ratio","node_gate")} for n in x["nodes"]]} for x in rows],
      "claim_boundary":"Prospective six-seed audit of finite local current-kernel versus source-fixed LoRA replay along a constructed response-retracted GPT-2 path. A pass supports state-dependent response-fibre realizability advantage in this rank-2 LoRA-B domain only; it is not ordinary optimizer behavior, complete-kernel transport, a global action, or semantic generality."}
    (out/"report.json").write_text(json.dumps(report,indent=2)); (out/"protocol.json").write_text(json.dumps({"protocol":PROTOCOL,"seeds":SEEDS,"prospective":True,"frozen_gates":report["frozen_gates"],"anchors":ANCHORS,"task":TASK},indent=2))
    print("="*96); print(json.dumps(report,indent=2)); shutil.make_archive(str(out),"zip",out.parent,out.name); print(f"Results ZIP: {out}.zip")
if __name__=="__main__": main()
