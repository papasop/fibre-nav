#!/usr/bin/env python3
"""Fail-closed dynamic response-kernel Picard development audit.

The primary update explicitly recomputes ker DR(theta) at every step.  The
source-frozen, identity-metric, and spectrum-matched random-metric controls use
the same warm start, intrinsic LoRA chart, batches, update norm, and evaluation
schedule.  This is a development experiment, not a pre-declared confirmation.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, random, statistics, time
from pathlib import Path

PROTOCOL = "GPT2_LORA_DYNAMIC_RESPONSE_KERNEL_PICARD_V0_3_R7_DEVELOPMENT"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
SEEDS = [25211, 25217, 25229, 25237, 25247]
ARMS = ["current_fisher", "source_fisher", "current_identity", "current_random_metric"]


def seed_all(seed, torch):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sync(torch, device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def get_data(path: Path):
    import urllib.request
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != DATA_SHA256: raise RuntimeError(f"dataset SHA-256 mismatch: {got}")
    return raw


def make_blocks(raw, seq_len, torch):
    # Byte-level task avoids tokenizer/network ambiguity and binds exactly to r4/r5 data.
    x = torch.tensor(list(raw), dtype=torch.long)
    n = (len(x) - 1) // seq_len
    x = x[: n * seq_len + 1]
    xb = x[:-1].view(n, seq_len); yb = x[1:].view(n, seq_len)
    cut = int(0.9 * n)
    return (xb[:cut], yb[:cut]), (xb[cut:], yb[cut:])


def inject_chart_lora(model, chart_dim, rank, layers, torch):
    """Replace selected GPT-2 Conv1D c_attn modules with a shared chart LoRA."""
    import torch.nn as nn
    class ChartLoRAConv1D(nn.Module):
        def __init__(self, base, coord, offset, gen):
            super().__init__(); self.nf = base.nf; self.rank = rank; self.scale = 1.0 / rank
            self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            nin, nout = self.weight.shape
            count = rank * nin + nout * rank
            basis = torch.randn(count, chart_dim, generator=gen, dtype=torch.float64)
            basis, _ = torch.linalg.qr(basis, mode="reduced")
            self.register_buffer("basis", basis.to(dtype=torch.float32))
            base_vec = torch.zeros(count, dtype=torch.float32)
            base_vec[:rank*nin] = 0.02 * torch.randn(rank*nin, generator=gen)
            self.register_buffer("base_vec", base_vec)
            self.coord = coord; self.nin=nin; self.nout=nout; self.offset=offset
        def forward(self, x):
            v = self.base_vec + self.basis @ self.coord
            k = self.rank * self.nin
            A = v[:k].view(self.rank, self.nin)
            B = v[k:].view(self.nout, self.rank)
            return torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight).reshape(*x.shape[:-1], self.nf) + ((x @ A.T) @ B.T) * self.scale

    for p in model.parameters(): p.requires_grad_(False)
    coord = nn.Parameter(torch.zeros(chart_dim, dtype=torch.float32))
    model.register_parameter("intrinsic_lora_coordinate", coord)
    gen = torch.Generator(device="cpu").manual_seed(83017)
    chosen = list(range(model.config.n_layer-layers, model.config.n_layer))
    for i in chosen:
        old = model.transformer.h[i].attn.c_attn
        model.transformer.h[i].attn.c_attn = ChartLoRAConv1D(old, coord, i, gen)
    return coord, chosen


def batch_loss(model, xb, yb, idx, device, torch):
    x=xb[idx].to(device, non_blocking=True); y=yb[idx].to(device, non_blocking=True)
    logits=model(x).logits
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def grad_vec(loss, coord, torch, retain=False):
    return torch.autograd.grad(loss, coord, retain_graph=retain, create_graph=False)[0]


def response_jacobian(model, coord, anchors, device, torch):
    rows=[]
    for xb,yb,idx in anchors:
        rows.append(grad_vec(batch_loss(model,xb,yb,idx,device,torch),coord,torch).detach().double())
    return torch.stack(rows)


def null_basis(J, torch):
    # Complete float64 SVD: rows of Vh after numerical rank span ker J.
    U,S,Vh=torch.linalg.svd(J.double(), full_matrices=True)
    tol=max(J.shape)*torch.finfo(torch.float64).eps*(S.max() if S.numel() else 1.0)
    rank=int((S>tol).sum().item()); N=Vh[rank:].T.contiguous()
    if N.shape[1] < 1: raise RuntimeError("response Jacobian has empty numerical kernel")
    P=N@N.T
    leak=float((J@P).norm()/(J.norm().clamp_min(1e-30)))
    idem=float((P@P-P).norm()/P.norm().clamp_min(1e-30))
    orth=float((N.T@N-torch.eye(N.shape[1],dtype=N.dtype,device=N.device)).norm())
    return N,{"rank":rank,"min_nonzero_sv":float(S[rank-1]) if rank else 0.0,"linear_leakage":leak,"idempotence":idem,"orthogonality":orth}


def fisher_in_basis(model, coord, metric_batches, N, device, damping, torch):
    cols=[]
    for xb,yb,idx in metric_batches:
        g=grad_vec(batch_loss(model,xb,yb,idx,device,torch),coord,torch).detach().double()
        cols.append(N.T@g)
    H=torch.stack(cols)
    G=(H.T@H)/len(cols)
    scale=torch.trace(G)/G.shape[0]
    return G + (damping*(scale.clamp_min(1e-12)))*torch.eye(G.shape[0],dtype=G.dtype,device=G.device)


def direction(g, N, metric, arm, random_Q, torch):
    h=N.T@g.double()
    if arm=="current_identity": u=h
    else:
        if arm=="current_random_metric":
            eig=torch.linalg.eigvalsh(metric).clamp_min(1e-12)
            M=random_Q@torch.diag(eig)@random_Q.T
        else: M=metric
        u=torch.linalg.solve(M,h)
    d=-(N@u)
    return d/d.norm().clamp_min(1e-30)


def evaluate(model, xv, yv, device, torch, max_blocks=32, bs=8):
    model.eval(); vals=[]
    with torch.no_grad():
        for s in range(0,min(len(xv),max_blocks),bs):
            idx=torch.arange(s,min(s+bs,min(len(xv),max_blocks)))
            vals.append(float(batch_loss(model,xv,yv,idx,device,torch)))
    model.train(); return statistics.mean(vals)


def frozen_indices(n, steps, bs, seed, torch):
    gen=torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randint(0,n,(bs,),generator=gen) for _ in range(steps)]


def warm_start(model, coord, xb, yb, batches, device, steps, lr, torch):
    opt=torch.optim.AdamW([coord],lr=lr,weight_decay=0.0)
    for k in range(steps):
        opt.zero_grad(set_to_none=True); loss=batch_loss(model,xb,yb,batches[k],device,torch); loss.backward(); opt.step()
    return coord.detach().clone()


def run_arm(model, coord, warm, arm, train, val, batches, anchors, metric_batches, args, device, torch, seed, out):
    xb,yb=train; xv,yv=val; coord.data.copy_(warm); model.train()
    # Source objects are frozen once after the common warm start.
    J0=response_jacobian(model,coord,anchors,device,torch); N0,diag0=null_basis(J0,torch)
    G0=fisher_in_basis(model,coord,metric_batches,N0,device,args.damping,torch)
    qgen=torch.Generator(device="cpu").manual_seed(seed+99173)
    Z=torch.randn(N0.shape[1],N0.shape[1],generator=qgen,dtype=torch.float64).to(device)
    random_Q,_=torch.linalg.qr(Z)
    trace=[]; max_leak=max_idem=max_orth=0.; ranks=[]; t0=time.perf_counter()
    for step in range(args.steps):
        loss=batch_loss(model,xb,yb,batches[args.warm_steps+step],device,torch)
        g=grad_vec(loss,coord,torch).detach()
        if arm=="source_fisher": N=N0; metric=G0; dg=diag0
        else:
            J=response_jacobian(model,coord,anchors,device,torch); N,dg=null_basis(J,torch)
            metric=fisher_in_basis(model,coord,metric_batches,N,device,args.damping,torch)
        # random Q dimension is stable because response rank is frozen by gate.
        if random_Q.shape[0]!=N.shape[1]: raise RuntimeError("kernel rank changed inside frozen chart")
        d=direction(g,N,metric,arm,random_Q,torch)
        # Matched trust-region norm for all arms; lr is a geometric step length.
        coord.data.add_(d.to(coord.dtype),alpha=args.lr)
        max_leak=max(max_leak,dg["linear_leakage"]); max_idem=max(max_idem,dg["idempotence"]); max_orth=max(max_orth,dg["orthogonality"]); ranks.append(dg["rank"])
        if (step+1)%args.eval_interval==0 or step==args.steps-1:
            vl=evaluate(model,xv,yv,device,torch,args.val_blocks,args.batch_size)
            trace.append({"step":step+1,"train_loss":float(loss),"validation_loss":vl})
            print(f"[{arm}] step={step+1}/{args.steps} val={vl:.6f} leak={max_leak:.2e}",flush=True)
    sync(torch,device); elapsed=time.perf_counter()-t0
    rec={"seed":seed,"arm":arm,"final_validation_loss":trace[-1]["validation_loss"],"best_validation_loss":min(x["validation_loss"] for x in trace),"timed_seconds":elapsed,"max_linear_kernel_leakage":max_leak,"max_projector_idempotence":max_idem,"max_basis_orthogonality_error":max_orth,"response_ranks":sorted(set(ranks)),"trace":trace}
    (out/f"{arm}_{seed}.json").write_text(json.dumps(rec,indent=2)+"\n")
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",default="cuda"); ap.add_argument("--outdir",default="dynamic_r7_results"); ap.add_argument("--data-root",default="data")
    ap.add_argument("--steps",type=int,default=180); ap.add_argument("--warm-steps",type=int,default=50); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--seq-len",type=int,default=64)
    ap.add_argument("--chart-dim",type=int,default=24); ap.add_argument("--lora-rank",type=int,default=4); ap.add_argument("--layers",type=int,default=2); ap.add_argument("--lr",type=float,default=0.035); ap.add_argument("--warm-lr",type=float,default=0.02); ap.add_argument("--damping",type=float,default=0.08)
    ap.add_argument("--eval-interval",type=int,default=10); ap.add_argument("--val-blocks",type=int,default=32); ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)
    if args.quick: args.steps=40; args.warm_steps=10; args.val_blocks=16; seeds=SEEDS[:1]
    else: seeds=SEEDS
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    raw=get_data(Path(args.data_root)/"tinyshakespeare.txt"); train,val=make_blocks(raw,args.seq_len,torch)
    print(f"protocol={PROTOCOL} device={device} seeds={len(seeds)} arms={ARMS}",flush=True)
    records=[]; started=time.time()
    for si,seed in enumerate(seeds,1):
        seed_all(seed,torch)
        # Randomly initialized compact GPT-2 is deliberate: offline, deterministic mechanism audit.
        cfg=GPT2Config(vocab_size=256,n_positions=args.seq_len,n_ctx=args.seq_len,n_embd=256,n_layer=6,n_head=8,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)
        model=GPT2LMHeadModel(cfg).to(device); coord,layers=inject_chart_lora(model,args.chart_dim,args.lora_rank,args.layers,torch); model.to(device)
        batches=frozen_indices(len(train[0]),args.warm_steps+args.steps,args.batch_size,seed+11,torch)
        # Disjoint fixed response and metric anchors.
        response_ids=[torch.arange(0,args.batch_size),torch.arange(args.batch_size,2*args.batch_size)]
        metric_ids=[torch.arange(2*args.batch_size+i*args.batch_size,2*args.batch_size+(i+1)*args.batch_size) for i in range(4)]
        anchors=[(train[0],train[1],i) for i in response_ids]; metrics=[(train[0],train[1],i) for i in metric_ids]
        warm=warm_start(model,coord,train[0],train[1],batches,device,args.warm_steps,args.warm_lr,torch)
        print(f"[seed {si}/{len(seeds)}] {seed} warm start complete",flush=True)
        for arm in ARMS:
            records.append(run_arm(model,coord,warm,arm,train,val,batches,anchors,metrics,args,device,torch,seed,out))
    by={(r["seed"],r["arm"]):r for r in records}; pairs=[]
    for s in seeds:
        cur=by[s,"current_fisher"]
        contrasts={a:by[s,a]["final_validation_loss"]-cur["final_validation_loss"] for a in ARMS[1:]}
        pairs.append({"seed":s,"control_minus_current_final_loss":contrasts,"current_beats_all":all(x>0 for x in contrasts.values())})
    med={a:statistics.median(by[s,a]["final_validation_loss"]-by[s,"current_fisher"]["final_validation_loss"] for s in seeds) for a in ARMS[1:]}
    gates={
      "five_new_development_seeds":len(seeds)==5,
      "all_runs_finite":all(math.isfinite(r["final_validation_loss"]) for r in records),
      "float64_kernel_leakage_at_most_1e_10":max(r["max_linear_kernel_leakage"] for r in records)<=1e-10,
      "projector_idempotence_at_most_1e_10":max(r["max_projector_idempotence"] for r in records)<=1e-10,
      "response_rank_constant":all(len(r["response_ranks"])==1 for r in records),
      "current_beats_source_median":med["source_fisher"]>0,
      "current_beats_identity_median":med["current_identity"]>0,
      "current_beats_spectrum_matched_random_median":med["current_random_metric"]>0,
      "current_beats_all_controls_in_at_least_4_of_5_seeds":sum(p["current_beats_all"] for p in pairs)>=4,
    }
    passed=all(gates.values()) and not args.quick
    summary={"protocol":PROTOCOL,"mode":"quick_nonclaim" if args.quick else "new_seed_development","model":"deterministic compact GPT-2 byte LM","data_sha256":DATA_SHA256,"design":{"dynamic_current_kernel":True,"kernel_refresh_steps":1,"float64_svd":True,"same_chart_warm_start_batches_step_norm":True,"random_control":"same current kernel and Fisher eigenvalues; frozen random eigenvectors","arms":ARMS},"seeds":seeds,"pairs":pairs,"median_control_minus_current_final_loss":med,"gates":gates,"wall_seconds":time.time()-started,"scientific_status":"DYNAMIC_RESPONSE_KERNEL_PICARD_R7_DEVELOPMENT_SIGNAL_SUPPORTED" if passed else "DYNAMIC_RESPONSE_KERNEL_PICARD_R7_INCONCLUSIVE_FAIL_CLOSED","claim_boundary":"Development audit in a deterministic compact GPT-2 byte language model and a shared 24-dimensional rank-4 LoRA chart. A positive result freezes a candidate protocol for later independent confirmation; it is not pretrained GPT-2 superiority, semantic transfer, a universal optimizer ordering, or a global Picard theorem."}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return 0 if passed or args.quick else 2


if __name__=="__main__": raise SystemExit(main())
