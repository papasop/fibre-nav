#!/usr/bin/env python3
"""GPTW-v1-r3: prospective new-seed GPT-2/LoRA adaptive-value confirmation."""
from __future__ import annotations
import argparse, json, math, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL="CNER_GPT2_LORA_RESPONSE_FIBRE_ADAPTIVE_VALUE_V1_R3_NEWSEED"
MODEL_IDS=["openai-community/gpt2","gpt2"]
SEEDS=[25742,25743,25744,25745,25746,25747,25748,25749]
REQUIRED=6; RANDOM_CONTROLS=16; LORA_RANK=2; LAYERS=[10,11]
ANCHOR_RESPONSE_DIM=16; UPDATE_NORM=0.35; FD_EPSILONS=[1e-2,3e-3,1e-3]
MAX_RELATIVE_ANCHOR_LEAKAGE=0.025; MAX_RANDOM_LEAKAGE_MISMATCH=0.015
MIN_GAIN_MARGIN=1e-4; MAX_FD_JACOBIAN_ABSOLUTE_NORMALIZED_ERROR=5e-5
MAX_FD_SUCCESSIVE_NORMALIZED_CHANGE=5e-5; WALL_CLOCK_LIMIT_SECONDS=3300

ANCHORS=["The capital of France is","Water freezes at","A triangle has","The opposite of hot is"]
RESPONSE_TOKENS=[" Paris"," zero"," three"," cold"]
CONSTRUCTION=[
 ("Codeword dax means the color of snow. Snow is", " white"),
 ("Codeword wug means the color of grass. Grass is", " green"),
 ("Codeword zorp means the opposite of down. The answer is", " up"),
 ("Codeword blick means the opposite of night. The answer is", " day"),
 ("Remember: dax maps to white. Therefore dax is", " white"),
 ("Remember: wug maps to green. Therefore wug is", " green"),
 ("Remember: zorp maps to up. Therefore zorp is", " up"),
 ("Remember: blick maps to day. Therefore blick is", " day"),
]
HELDOUT=[
 ("In this task dax denotes snow's colour. Answer:", " white"),
 ("In this task wug denotes grass's colour. Answer:", " green"),
 ("Under the stated code, zorp is the reverse of down:", " up"),
 ("Under the stated code, blick is the reverse of night:", " day"),
]

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class NativeLoRAConv1D(torch.nn.Module):
    """Minimal LoRA wrapper for transformers.pytorch_utils.Conv1D."""
    def __init__(self,base,rank,alpha):
        super().__init__(); self.base=base; self.scale=alpha/rank
        for p in self.base.parameters(): p.requires_grad_(False)
        in_features=int(base.weight.shape[0]); out_features=int(base.weight.shape[1])
        self.lora_A=torch.nn.Parameter(torch.empty(in_features,rank),requires_grad=False); self.lora_B=torch.nn.Parameter(torch.zeros(rank,out_features))
        torch.nn.init.normal_(self.lora_A,std=0.02)
    def forward(self,x): return self.base(x)+(x@self.lora_A@self.lora_B)*self.scale

def load_hf():
    errors=[]
    for model_id in MODEL_IDS:
        try:
            tok=AutoTokenizer.from_pretrained(model_id); model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32)
            return model_id,tok,model
        except Exception as exc: errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError("GPT-2 download/load failed for both model IDs:\n"+"\n".join(errors))

def build(device):
    model_id,tok,model=load_hf(); tok.pad_token=tok.eos_token
    for p in model.parameters(): p.requires_grad_(False)
    for layer in LAYERS: model.transformer.h[layer].attn.c_attn=NativeLoRAConv1D(model.transformer.h[layer].attn.c_attn,LORA_RANK,4)
    model.to(device=device,dtype=torch.float64).eval(); params=[]
    for layer in LAYERS:
        module=model.transformer.h[layer].attn.c_attn; params.append(module.lora_B)
    return model_id,tok,model,params

def flat(params): return torch.cat([p.detach().reshape(-1) for p in params])

def assign(params,v):
    k=0
    with torch.no_grad():
        for p in params: n=p.numel(); p.copy_(v[k:k+n].view_as(p)); k+=n

def grad_flat(y,params,retain=False):
    gs=torch.autograd.grad(y,params,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])

def next_logits(model,tok,prompt,device):
    ids=tok(prompt,return_tensors="pt").input_ids.to(device)
    return model(input_ids=ids).logits[0,-1]

def token_id(tok,s):
    ids=tok.encode(s,add_special_tokens=False)
    if not ids: raise RuntimeError(f"target tokenization is empty: {s!r}")
    return ids[-1]

def task_loss(model,tok,examples,device,targets=None):
    losses=[]
    for i,(prompt,target) in enumerate(examples):
        tid=token_id(tok,targets[i] if targets else target)
        losses.append(torch.nn.functional.cross_entropy(next_logits(model,tok,prompt,device)[None,:],torch.tensor([tid],device=device)))
    return torch.stack(losses).mean()

def response(model,tok,device):
    vals=[]
    for prompt,target in zip(ANCHORS,RESPONSE_TOKENS):
        logits=next_logits(model,tok,prompt,device); tid=token_id(tok,target)
        vals.extend([logits[tid],logits.topk(3).values.mean(),logits.mean(),logits.std()])
    return torch.stack(vals)[:ANCHOR_RESPONSE_DIM]

def jacobian_response(model,tok,params,device):
    r=response(model,tok,device); rows=[]
    for i in range(len(r)): rows.append(grad_flat(r[i],params,retain=i+1<len(r)))
    return r.detach(),torch.stack(rows)

def project_kernel(v,J):
    gram=J@J.T; scale=float(torch.trace(gram)/max(1,len(gram)))
    coeff=torch.linalg.solve(gram+(1e-8*max(scale,1e-12))*torch.eye(len(gram),device=J.device,dtype=J.dtype),J@v.to(dtype=J.dtype))
    return v-J.T@coeff

def unit(v): return v/v.norm().clamp_min(1e-30)

def eval_direction(model,tok,params,base_vec,r0,direction,device):
    assign(params,base_vec+UPDATE_NORM*unit(direction)); loss=float(task_loss(model,tok,HELDOUT,device)); r=response(model,tok,device).detach()
    leak=float((r-r0).norm()/r0.norm().clamp_min(1e-30)); assign(params,base_vec)
    return loss,leak

def run_seed(seed,device):
    seed_all(seed); model_id,tok,model,params=build(device); base_vec=flat(params); r0,J=jacobian_response(model,tok,params,device)
    base_loss=float(task_loss(model,tok,HELDOUT,device).detach()); selected=project_kernel(-grad_flat(task_loss(model,tok,CONSTRUCTION,device),params),J)
    shuffled_targets=[x[1] for x in CONSTRUCTION]; shuffled_targets=shuffled_targets[2:]+shuffled_targets[:2]
    shuffled=project_kernel(-grad_flat(task_loss(model,tok,CONSTRUCTION,device,shuffled_targets),params),J)
    sel_loss,sel_leak=eval_direction(model,tok,params,base_vec,r0,selected,device)
    sign_loss,sign_leak=eval_direction(model,tok,params,base_vec,r0,-selected,device)
    shuf_loss,shuf_leak=eval_direction(model,tok,params,base_vec,r0,shuffled,device)
    gen=torch.Generator(device=device).manual_seed(seed+991); random_rows=[]
    for _ in range(RANDOM_CONTROLS):
        q=project_kernel(torch.randn(len(base_vec),generator=gen,device=device,dtype=J.dtype),J)
        loss,leak=eval_direction(model,tok,params,base_vec,r0,q,device); random_rows.append({"loss":loss,"gain":base_loss-loss,"leakage":leak})
    best_random=max(random_rows,key=lambda x:x["gain"]); selected_gain=base_loss-sel_loss
    # A kernel direction has Jv≈0, so dividing by ||Jv|| is ill-conditioned.
    # Certify central differences against Jv in the ambient Jacobian scale instead.
    u=unit(selected); jvp=J@u; jacobian_scale=J.norm().clamp_min(1e-30); fd_rows=[]
    for eps in FD_EPSILONS:
        assign(params,base_vec+eps*u); rp=response(model,tok,device).detach()
        assign(params,base_vec-eps*u); rm=response(model,tok,device).detach(); assign(params,base_vec)
        fd=(rp-rm)/(2*eps)
        fd_rows.append({"epsilon":eps,"absolute_normalized_error":float((fd-jvp).norm()/jacobian_scale),"vector":fd.cpu().tolist()})
    fd_abs=fd_rows[-1]["absolute_normalized_error"]
    fd_changes=[]
    for a,b in zip(fd_rows[:-1],fd_rows[1:]):
        va=torch.tensor(a["vector"],device=device,dtype=torch.float64); vb=torch.tensor(b["vector"],device=device,dtype=torch.float64)
        fd_changes.append(float((vb-va).norm()/jacobian_scale))
    fd_change=fd_changes[-1]
    kernel_rel=float((J@u).norm()/J.norm().clamp_min(1e-30))
    leak_mismatch=max(abs(sel_leak-x["leakage"]) for x in random_rows)
    gates={"beats_no_tangent":selected_gain>=MIN_GAIN_MARGIN,
           "beats_best_random":selected_gain>=best_random["gain"]+MIN_GAIN_MARGIN,
           "beats_sign_reverse":selected_gain>=base_loss-sign_loss+MIN_GAIN_MARGIN,
           "beats_shuffled_target":selected_gain>=base_loss-shuf_loss+MIN_GAIN_MARGIN,
           "anchor_leakage":sel_leak<=MAX_RELATIVE_ANCHOR_LEAKAGE,
           "random_leakage_match":leak_mismatch<=MAX_RANDOM_LEAKAGE_MISMATCH,
           "fd_absolute_consistency":fd_abs<=MAX_FD_JACOBIAN_ABSOLUTE_NORMALIZED_ERROR,
           "fd_successive_convergence":fd_change<=MAX_FD_SUCCESSIVE_NORMALIZED_CHANGE,
           "kernel_projection":kernel_rel<=1e-5}
    out={"seed":seed,"model":model_id,"lora_implementation":"native Conv1D wrapper with frozen random A and audited B coordinates","lora_rank":LORA_RANK,"lora_layers":LAYERS,"trainable_parameters":len(base_vec),
         "base_heldout_loss":base_loss,"selected":{"loss":sel_loss,"gain":selected_gain,"leakage":sel_leak},
         "sign_reversed":{"loss":sign_loss,"gain":base_loss-sign_loss,"leakage":sign_leak},
         "shuffled_target":{"loss":shuf_loss,"gain":base_loss-shuf_loss,"leakage":shuf_leak},
         "best_of_16_random":best_random,"random_controls":random_rows,"max_selected_random_leakage_mismatch":leak_mismatch,
         "finite_difference_audit":fd_rows,"successive_normalized_changes":fd_changes,
         "finest_absolute_normalized_error":fd_abs,"finest_successive_normalized_change":fd_change,
         "projected_kernel_relative_residual":kernel_rel,
         "gates":gates,"instance_supported":all(gates.values())}
    del model; torch.cuda.empty_cache(); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="cner_gpt2_lora_response_fibre_adaptive_value_v1_r3_newseed_results"); args,_=ap.parse_known_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPTW-v1-r3 requires CUDA; select an A100 runtime")
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); rows=[]
    for i,s in enumerate(SEEDS):
        if time.monotonic()-started>WALL_CLOCK_LIMIT_SECONDS: raise TimeoutError("GPTW-v1-r3 exceeded its 55-minute hard limit")
        print(f"[GPTW-v1-r3 prospective {i+1}/8] seed={s}",flush=True); row=run_seed(s,device); rows.append(row)
        (out/f"seed_{s}.json").write_text(json.dumps(row,indent=2)); print(json.dumps({"seed":s,"supported":row["instance_supported"],"gates":row["gates"]}),flush=True)
    elapsed=time.monotonic()-started; count=sum(x["instance_supported"] for x in rows); passed=count>=REQUIRED
    report={"protocol":PROTOCOL,"scientific_status":"PROSPECTIVE_NEWSEED_GPT2_LORA_ADAPTIVE_VALUE_AUDIT_COMPLETED",
      "decision":"GPT2_LORA_SELECTED_RESPONSE_FIBRE_VALUE_PROSPECTIVELY_CONFIRMED" if passed else "GPT2_LORA_SELECTED_RESPONSE_FIBRE_VALUE_PROSPECTIVE_CONFIRMATION_FAILED",
      "model_candidates":MODEL_IDS,"lora_implementation":"native Conv1D wrapper; PEFT not required","seeds":SEEDS,"supporting_instances":count,"required_instances":REQUIRED,"elapsed_seconds":elapsed,
      "wall_clock_limit_seconds":WALL_CLOCK_LIMIT_SECONDS,"frozen_gates":{"best_of_random_controls":RANDOM_CONTROLS,
      "minimum_gain_margin":MIN_GAIN_MARGIN,"maximum_anchor_leakage":MAX_RELATIVE_ANCHOR_LEAKAGE,
      "maximum_random_leakage_mismatch":MAX_RANDOM_LEAKAGE_MISMATCH,
      "maximum_fd_jacobian_absolute_normalized_error":MAX_FD_JACOBIAN_ABSOLUTE_NORMALIZED_ERROR,
      "maximum_fd_successive_normalized_change":MAX_FD_SUCCESSIVE_NORMALIZED_CHANGE},
      "instance_summaries":[{k:x[k] for k in ("seed","base_heldout_loss","selected","sign_reversed","shuffled_target","best_of_16_random","finest_absolute_normalized_error","finest_successive_normalized_change","projected_kernel_relative_residual","gates","instance_supported")} for x in rows],
      "claim_boundary":"Prospective eight-new-seed confirmation in frozen GPT-2 with native rank-2 LoRA-B coordinates only in attention c_attn modules of layers 10-11, a finite built-in prompt task and declared anchor-logit response. A pass supports selected response-fibre tangent adaptive value in this LoRA domain only; it is not moving-fibre transport, full-model or semantic generality, an action principle, or global minimality."}
    (out/"report.json").write_text(json.dumps(report,indent=2)); (out/"protocol.json").write_text(json.dumps({"protocol":PROTOCOL,"model_candidates":MODEL_IDS,"seeds":SEEDS,"required":REQUIRED,"prospective":True,"construction":CONSTRUCTION,"heldout":HELDOUT,"anchors":ANCHORS,"response_tokens":RESPONSE_TOKENS},indent=2))
    print("="*96); print(json.dumps(report,indent=2)); shutil.make_archive(str(out),"zip",out.parent,out.name); print(f"Results ZIP: {out}.zip")

if __name__=="__main__": main()
