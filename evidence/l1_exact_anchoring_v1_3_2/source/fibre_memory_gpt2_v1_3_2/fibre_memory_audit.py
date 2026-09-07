#!/usr/bin/env python3
"""GPT-2 response-fibre memory write/read/overwrite development audit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ANCHORS = [
    "The sky above the quiet city was", "A careful scientific measurement should",
    "When the rain finally stopped, the", "The musician opened the old wooden",
    "A reliable explanation begins with", "At the edge of the forest stood",
    "The small library remained open because", "To solve the problem, first identify",
    "The train crossed the bridge while", "An unexpected result may indicate",
    "In the middle of winter, the", "The engineer checked the circuit before",
    "A good map helps a traveler", "The teacher wrote the final equation",
    "The garden changed slowly throughout", "Before publishing the result, researchers"
]

SLOT_KEYS = ["amber", "cedar"]


def exact_prompt(key):
    # The unique key is immediately adjacent to the prediction site.
    return f"Private memory key {key}:"


def paraphrase_prompt(key):
    return f"Recall the saved code for {key}:"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def make_model(cfg, device):
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)
    lc = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
        target_modules=["c_attn"], bias="none", task_type="CAUSAL_LM",
        layers_to_transform=[10, 11], layers_pattern="h"
    )
    model = get_peft_model(base, lc).to(device)
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.0
    # Restrict the chart to LoRA-B; LoRA-A is a frozen deterministic chart basis.
    for n, p in model.named_parameters(): p.requires_grad_("lora_B" in n)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError("No LoRA-B parameters found")
    return tok, model, params


def single_token_pair(tok):
    candidates = [" red", " blue", " yes", " no", " one", " two", " cat", " dog"]
    ids = [(s, tok.encode(s, add_special_tokens=False)) for s in candidates]
    ids = [(s, x[0]) for s, x in ids if len(x) == 1]
    for i in range(0, len(ids)-1, 2):
        if ids[i][1] != ids[i+1][1]: return ids[i], ids[i+1]
    raise RuntimeError("Could not find two single-token code symbols")


def next_logits(model, tok, prompts, device):
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)
    out = model(**batch).logits
    last = batch["attention_mask"].sum(1) - 1
    return out[torch.arange(len(prompts), device=device), last]


def response_vector(model, tok, prompts, response_ids, device):
    logits = next_logits(model, tok, prompts, device)
    selected = logits[:, response_ids]
    return selected.mean(0)


def flat(xs): return torch.cat([x.reshape(-1) for x in xs])


def assign_flat(params, vector):
    off = 0
    with torch.no_grad():
        for p in params:
            n = p.numel(); p.copy_(vector[off:off+n].view_as(p)); off += n


def param_flat(params): return flat([p.detach() for p in params]).clone()


def jacobian_rows(response, params, create_graph=False):
    rows = []
    for i in range(response.numel()):
        gs = torch.autograd.grad(response[i], params, retain_graph=True, create_graph=create_graph,
                                 allow_unused=False)
        rows.append(flat(gs))
    return torch.stack(rows)


def kernel_project(v, J, ridge=1e-8):
    gram = J @ J.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, J @ v)
    return v - J.T @ coeff


def memory_loss(model, tok, prompts, targets, candidate_ids, device):
    """Binary candidate loss; avoids spending fibre budget on the full vocabulary."""
    logits = next_logits(model, tok, prompts, device)[:, candidate_ids]
    index = {token_id: i for i, token_id in enumerate(candidate_ids)}
    y = torch.tensor([index[t] for t in targets], device=device)
    return torch.nn.functional.cross_entropy(logits, y)


@torch.no_grad()
def read_accuracy(model, tok, prompts, targets, candidate_ids, device):
    logits = next_logits(model, tok, prompts, device)[:, candidate_ids]
    pred = logits.argmax(1)
    y = torch.tensor([candidate_ids.index(t) for t in targets], device=device)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def signed_margins(model, tok, prompts, targets, candidate_ids, device):
    logits = next_logits(model, tok, prompts, device)[:, candidate_ids]
    out = []
    for row, target in zip(logits, targets):
        ti = candidate_ids.index(target); oi = 1-ti
        out.append(float((row[ti]-row[oi]).item()))
    return out


def evaluate_content(model, tok, exact_prompts, para_prompts, targets, candidate_ids, device):
    em=signed_margins(model,tok,exact_prompts,targets,candidate_ids,device)
    pm=signed_margins(model,tok,para_prompts,targets,candidate_ids,device)
    return {"exact_accuracy":read_accuracy(model,tok,exact_prompts,targets,candidate_ids,device),
            "paraphrase_accuracy":read_accuracy(model,tok,para_prompts,targets,candidate_ids,device),
            "exact_signed_margins":em,"exact_median_signed_margin":float(np.median(em)),
            "paraphrase_signed_margins":pm,"paraphrase_median_signed_margin":float(np.median(pm))}


def anchor_metrics(model, tok, anchor_prompts, response_ids, response0, logits0, device):
    logits = next_logits(model, tok, anchor_prompts, device)
    response = logits[:, response_ids].mean(0)
    drift = float((response-response0).abs().max().item())
    p0 = torch.softmax(logits0, -1)
    kl = float((p0 * (torch.log_softmax(logits0, -1)-torch.log_softmax(logits, -1))).sum(-1).mean().item())
    return drift, max(0.0, kl)


def retract(model, tok, params, anchors, response_ids, target_response, device, steps):
    correction_norm = 0.0
    for _ in range(steps):
        r = response_vector(model, tok, anchors, response_ids, device)
        err = r - target_response
        if float(err.detach().abs().max()) < 1e-7: break
        J = jacobian_rows(r, params)
        gram = J @ J.T
        eye = torch.eye(gram.shape[0], device=device)
        delta = -J.T @ torch.linalg.solve(gram + 1e-8*eye, err.detach())
        correction_norm += float(delta.norm().item())
        assign_flat(params, param_flat(params) + delta)
    return correction_norm


def run_write(model, tok, params, cfg, anchors, response_ids, response0, logits0,
              train_prompts, write_targets, exact_eval_targets,
              read_prompts, paraphrase_eval_targets, candidate_ids,
              arm, seed, device, steps=None):
    steps = steps or cfg["write_steps"]
    start = param_flat(params)
    rng = torch.Generator(device=device).manual_seed(seed + 99173)
    hist, accepted = [], 0
    initial_drift, initial_kl = anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
    last_eligible = start.clone()
    selected_step = 0
    selected_kl = initial_kl
    selected_drift = initial_drift
    for k in range(steps):
        before = param_flat(params)
        loss = memory_loss(model, tok, train_prompts, write_targets, candidate_ids, device)
        g = flat(torch.autograd.grad(loss, params))
        r = response_vector(model, tok, anchors, response_ids, device)
        J = jacobian_rows(r, params)
        d = -kernel_project(g, J)
        if arm == "sign_reversed": d = -d
        elif arm == "random_kernel":
            z = torch.randn(d.shape, generator=rng, device=device, dtype=d.dtype)
            z = kernel_project(z, J); d = z * (d.norm() / (z.norm()+1e-12))
        elif arm == "no_move": d.zero_()
        norm = d.norm()
        if norm > 0: d = d / norm
        step = cfg["learning_rate"]
        ok = False; corr = 0.0; drift = math.inf; kl = math.inf; halves = 0
        if arm == "no_move":
            step = 0.0
        for halves in range(cfg.get("max_backtracking", 1) + 1):
            assign_flat(params, before + step * d)
            corr = retract(model, tok, params, anchors, response_ids, response0, device, cfg["retraction_steps"])
            drift, kl = anchor_metrics(model, tok, anchors, response_ids, response0, logits0, device)
            # Fibre membership is defined by the prospectively declared finite
            # response. Full-vocabulary anchor KL is recorded, but is not a
            # second hidden stepwise response constraint.
            ok = drift <= cfg["response_budget"]
            if ok: break
            assign_flat(params, before)
            step *= 0.5
            if step < cfg.get("min_step", 0.0): break
        if ok: accepted += 1
        else: assign_flat(params, before); step = 0.0
        endpoint_eligible = ok and kl <= cfg["endpoint_kl_gate"]
        if endpoint_eligible:
            last_eligible = param_flat(params)
            selected_step = k + 1
            selected_kl = kl
            selected_drift = drift
        para_acc = read_accuracy(model, tok, read_prompts, paraphrase_eval_targets, candidate_ids, device)
        exact_acc = read_accuracy(model, tok, train_prompts, exact_eval_targets, candidate_ids, device)
        objective_acc = read_accuracy(model, tok, train_prompts, write_targets, candidate_ids, device)
        hist.append({"step":k+1,"loss":float(loss.item()),"paraphrase_accuracy":para_acc,"response_drift":drift,
                     "exact_accuracy":exact_acc,"write_objective_accuracy":objective_acc,
                     "anchor_kl_diagnostic":kl,"accepted":ok,"accepted_step_size":step,
                     "endpoint_eligible_checkpoint":endpoint_eligible,
                     "backtracking_halvings":halves,"retraction_norm":corr,
                     "kernel_residual":float((J@d).norm().item())})
    raw_drift, raw_kl = anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
    raw_final_step = steps
    # Every arm is evaluated at its last prospectively eligible checkpoint.
    assign_flat(params,last_eligible)
    drift, kl = anchor_metrics(model, tok, anchors, response_ids, response0, logits0, device)
    exact_margins=signed_margins(model,tok,train_prompts,exact_eval_targets,candidate_ids,device)
    para_margins=signed_margins(model,tok,read_prompts,paraphrase_eval_targets,candidate_ids,device)
    own_margins=signed_margins(model,tok,train_prompts,write_targets,candidate_ids,device)
    return {"exact_accuracy":read_accuracy(model,tok,train_prompts,exact_eval_targets,candidate_ids,device),
            "paraphrase_accuracy":read_accuracy(model,tok,read_prompts,paraphrase_eval_targets,candidate_ids,device),
            "write_objective_accuracy":read_accuracy(model,tok,train_prompts,write_targets,candidate_ids,device),
            "write_objective_signed_margins":own_margins,
            "exact_signed_margins":exact_margins,
            "exact_median_signed_margin":float(np.median(exact_margins)),
            "paraphrase_signed_margins":para_margins,
            "paraphrase_median_signed_margin":float(np.median(para_margins)),
            "response_drift":drift,"anchor_kl_diagnostic":kl,"accepted_steps":accepted,
            "selected_checkpoint_step":selected_step,
            "selected_checkpoint_was_truncated":selected_step < raw_final_step,
            "raw_final_response_drift":raw_drift,"raw_final_anchor_kl":raw_kl,
            "selected_recorded_response_drift":selected_drift,"selected_recorded_anchor_kl":selected_kl,
            "position_distance":float((param_flat(params)-start).norm().item()),"history":hist}


def one_seed(cfg, seed, output):
    set_seed(seed); device, _ = device_dtype()
    tok, model, params = make_model(cfg, device)
    sym0, sym1 = single_token_pair(tok); candidate_ids=[sym0[1],sym1[1]]
    rng=random.Random(seed)
    anchors=rng.sample(ANCHORS, cfg["anchor_count"])
    response_ids=rng.sample(list(range(100, min(50000, tok.vocab_size))), cfg["response_dim"])
    with torch.no_grad():
        logits0=next_logits(model,tok,anchors,device).detach()
        response0=logits0[:,response_ids].mean(0).detach()
    # Balanced random codeword prevents a constant-token decoder from scoring above chance.
    bits=[0,1] * ((cfg["bits"]+1)//2); bits=bits[:cfg["bits"]]; rng.shuffle(bits)
    complement=[1-b for b in bits]
    keys=SLOT_KEYS[:cfg["bits"]]
    train=[exact_prompt(key) for key in keys]
    read=[paraphrase_prompt(key) for key in keys]
    targets=[candidate_ids[b] for b in bits]
    read_targets=targets[:]
    initial=param_flat(params)
    results={}
    for arm in cfg["arms"]:
        runs=[]
        repeats=cfg["best_of_random"] if arm=="random_kernel" else 1
        use_targets=targets
        if arm=="shuffled_label":
            # Deterministic wrong content; guaranteed to differ at every slot.
            use_targets=[candidate_ids[1-b] for b in bits]
        for j in range(repeats):
            assign_flat(params, initial)
            runs.append(run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                                  train,use_targets,targets,read,read_targets,candidate_ids,
                                  "true_current" if arm=="shuffled_label" else arm,seed+1009*j,device))
        results[arm]=max(runs,key=lambda x:x["exact_median_signed_margin"]) if arm=="random_kernel" else runs[0]
    # Causal overwrite A -> not-A from the true endpoint.
    assign_flat(params, initial)
    source_cross={
        "read_as_A":evaluate_content(model,tok,train,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,train,read,[candidate_ids[b] for b in complement],candidate_ids,device)
    }
    first=run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                    train,targets,targets,read,read_targets,candidate_ids,"true_current",seed,device)
    at_a=param_flat(params)
    comp_targets=[candidate_ids[b] for b in complement]
    a_cross={
        "read_as_A":evaluate_content(model,tok,train,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,train,read,comp_targets,candidate_ids,device)
    }
    second=run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                     train,comp_targets,comp_targets,read,comp_targets,candidate_ids,"true_current",seed+700001,device,
                     steps=cfg["rewrite_steps"])
    second["distance_from_A"]=float((param_flat(params)-at_a).norm().item())
    results["overwrite_A_to_notA"]={"A_endpoint":first,"notA_endpoint":second}
    nota_cross={
        "read_as_A":evaluate_content(model,tok,train,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,train,read,comp_targets,candidate_ids,device)
    }
    cross_read_matrix={"source":source_cross,"A_endpoint":a_cross,"notA_endpoint":nota_cross}
    tr=results["true_current"]; rr=results["random_kernel"]
    control_eligibility={name:{
        "response_eligible":results[name]["response_drift"]<=cfg["response_budget"],
        "endpoint_kl_eligible":results[name]["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "eligible":results[name]["response_drift"]<=cfg["response_budget"] and results[name]["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"]
    } for name in cfg["arms"]}
    primary_gates={
        "true_response_budget":tr["response_drift"]<=cfg["response_budget"],
        "true_endpoint_kl_gate":tr["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "true_exact_accuracy":tr["exact_accuracy"]>=cfg["train_accuracy_gate"],
        "true_beats_best_random_margin":tr["exact_median_signed_margin"]-rr["exact_median_signed_margin"]>=cfg["true_random_signed_margin_gate"],
        "all_control_arms_eligible":all(x["eligible"] for x in control_eligibility.values()),
        "overwrite_exact_accuracy":second["exact_accuracy"]>=cfg["train_accuracy_gate"],
        "overwrite_endpoint_kl_gate":second["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "overwrite_response_budget":second["response_drift"]<=cfg["response_budget"],
        "cross_read_switch":a_cross["read_as_A"]["exact_accuracy"]==1.0 and a_cross["read_as_notA"]["exact_accuracy"]==0.0 and nota_cross["read_as_notA"]["exact_accuracy"]==1.0 and nota_cross["read_as_A"]["exact_accuracy"]==0.0
    }
    secondary_gates={
        "true_paraphrase_accuracy":tr["paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"],
        "overwrite_paraphrase_accuracy":second["paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"]
    }
    gates={**primary_gates,**secondary_gates}
    record={"seed":seed,"device":str(device),"symbols":[sym0,sym1],"slot_keys":keys,"bits_A":bits,
            "bits_notA":complement,"anchors":anchors,"response_token_ids":response_ids,
            "arms":results,"control_eligibility":control_eligibility,
            "cross_read_matrix":cross_read_matrix,"primary_gates":primary_gates,
            "secondary_gates":secondary_gates,"gates":gates,
            "all_primary_gates_pass":all(primary_gates.values()),
            "all_secondary_gates_pass":all(secondary_gates.values()),
            "all_gates_pass":all(gates.values())}
    (output/f"seed_{seed}.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config_quick.json")
    ap.add_argument("--output",default="results")
    ap.add_argument("--seeds",nargs="*",type=int)
    args=ap.parse_args()
    cfg_path=Path(args.config); cfg=json.loads(cfg_path.read_text())
    if args.seeds: cfg["seeds"]=args.seeds
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    started=time.time(); records=[]
    for seed in cfg["seeds"]:
        print(f"[seed {seed}] starting",flush=True)
        records.append(one_seed(cfg,seed,out))
        print(f"[seed {seed}] primary={records[-1]['primary_gates']}",flush=True)
        print(f"[seed {seed}] secondary={records[-1]['secondary_gates']}",flush=True)
    summary={"protocol":cfg["protocol"],"scientific_status":"DEVELOPMENT_NOT_CONFIRMATION",
             "config":cfg,"config_sha256":sha256_file(cfg_path),"seeds_complete":len(records),
             "seeds_all_primary_gates_pass":sum(r["all_primary_gates_pass"] for r in records),
             "all_seeds_primary_pass":all(r["all_primary_gates_pass"] for r in records),
             "seeds_all_secondary_gates_pass":sum(r["all_secondary_gates_pass"] for r in records),
             "all_seeds_secondary_pass":all(r["all_secondary_gates_pass"] for r in records),
             "seeds_all_gates_pass":sum(r["all_gates_pass"] for r in records),
             "all_seeds_pass":all(r["all_gates_pass"] for r in records),
             "elapsed_seconds":time.time()-started,"python":sys.version,"platform":platform.platform(),
             "torch":torch.__version__,"transformers":transformers.__version__}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
