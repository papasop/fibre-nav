#!/usr/bin/env python3
"""L2 semantic-access worst-margin audit v1.5.6 (one frozen seed)."""
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

L3_NODES = [
    {"name":"paris", "description":"Paris, the capital city of France", "bits":[0,0]},
    {"name":"france", "description":"France, a country in Europe", "bits":[0,1]},
    {"name":"europe", "description":"Europe, the continent containing France", "bits":[1,0]},
    {"name":"banana", "description":"banana, a yellow fruit", "bits":[1,1]},
]
L3_EDGES = [("paris", "france"), ("france", "europe")]


def exact_prompt(key):
    # The unique key is immediately adjacent to the prediction site.
    return f"Private memory key {key}:"


def train_paraphrase_prompts(key):
    return [
        f"The code assigned to memory {key} is",
        f"For saved key {key}, the colour is",
        f"Memory {key} stores the colour",
        f"Looking up {key} returns",
    ]


def heldout_paraphrase_prompts(key):
    return [
        f"Which colour belongs to memory {key}?",
        f"Recall {key}'s stored colour:",
        f"The remembered colour for {key} is",
    ]


def concept_prompts(node):
    d=node["description"]
    exact=[f"Concept record {d}; first code:", f"Concept record {d}; second code:"]
    train=[f"For {d}, code component one is", f"For {d}, code component two is"]
    held=[f"Retrieve bit one for the idea {d}:", f"Retrieve bit two for the idea {d}:"]
    return exact, train, held


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    """Serialize NumPy/PyTorch scalar diagnostics without weakening schemas."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def semantic_memory_loss(model, tok, prompts, targets, candidate_ids, device,
                         pair_count, pull_weight, consistency_weight,
                         worst_item_weight, worst_item_temperature,
                         margin_weight, margin_target):
    """Mean fit plus smooth worst-item and signed-margin objectives."""
    batch = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)
    out = model(**batch, output_hidden_states=True)
    last = batch["attention_mask"].sum(1) - 1
    rows = torch.arange(len(prompts), device=device)
    logits = out.logits[rows, last][:, candidate_ids]
    index = {token_id: i for i, token_id in enumerate(candidate_ids)}
    y = torch.tensor([index[t] for t in targets], device=device)
    per_item_ce = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    ce = per_item_ce.mean()
    hidden = out.hidden_states[-1][rows, last]
    if len(prompts)%pair_count:
        raise ValueError("Semantic prompts must contain complete view-major blocks")
    view_count=len(prompts)//pair_count
    tau=max(float(worst_item_temperature),1e-6)
    robust_ce=tau*torch.logsumexp(per_item_ce/tau,dim=0)
    signed=(2*y.float()-1)*(logits[:,1]-logits[:,0])
    margin_penalty=torch.nn.functional.softplus(margin_target-signed)
    robust_margin=tau*torch.logsumexp(margin_penalty/tau,dim=0)
    h=torch.nn.functional.normalize(hidden,dim=-1).reshape(view_count,pair_count,-1)
    centroid=torch.nn.functional.normalize(h.mean(0),dim=-1)
    pull=(h-centroid.unsqueeze(0)).pow(2).sum(-1).mean()
    p=torch.softmax(logits,dim=-1).reshape(view_count,pair_count,-1)
    mean_p=p.mean(0).clamp_min(1e-8)
    consistency=(p*(p.clamp_min(1e-8).log()-mean_p.log().unsqueeze(0))).sum(-1).mean()
    total=(ce + worst_item_weight*robust_ce + margin_weight*robust_margin
           + pull_weight*pull + consistency_weight*consistency)
    return total, ce, pull, consistency, robust_ce, robust_margin


@torch.no_grad()
def training_selection_tuple(model,tok,prompts,targets,candidate_ids,device,pair_count):
    """Prospective lexicographic checkpoint rule using training views only."""
    logits=next_logits(model,tok,prompts,device)[:,candidate_ids]
    index={token_id:i for i,token_id in enumerate(candidate_ids)}
    y=torch.tensor([index[t] for t in targets],device=device)
    signed=(2*y.float()-1)*(logits[:,1]-logits[:,0])
    pred=logits.argmax(-1)
    correct=(pred==y).float().reshape(-1,pair_count).mean(-1)
    margins=signed.reshape(-1,pair_count).amin(-1)
    return (float(correct.min().item()),float(margins.min().item()),
            float(signed.median().item()))


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
    para_targets=targets*(len(para_prompts)//len(targets))
    pm=signed_margins(model,tok,para_prompts,para_targets,candidate_ids,device)
    return {"exact_accuracy":read_accuracy(model,tok,exact_prompts,targets,candidate_ids,device),
            "paraphrase_accuracy":read_accuracy(model,tok,para_prompts,para_targets,candidate_ids,device),
            "exact_signed_margins":em,"exact_median_signed_margin":float(np.median(em)),
            "paraphrase_signed_margins":pm,"paraphrase_median_signed_margin":float(np.median(pm))}


@torch.no_grad()
def view_accuracies(model,tok,prompts,targets,candidate_ids,device,view_width):
    return [read_accuracy(model,tok,prompts[i:i+view_width],targets,
                          candidate_ids,device)
            for i in range(0,len(prompts),view_width)]


def anchor_metrics(model, tok, anchor_prompts, response_ids, response0, logits0, device):
    logits = next_logits(model, tok, anchor_prompts, device)
    response = logits[:, response_ids].mean(0)
    drift = float((response-response0).abs().max().item())
    p0 = torch.softmax(logits0, -1)
    kl = float((p0 * (torch.log_softmax(logits0, -1)-torch.log_softmax(logits, -1))).sum(-1).mean().item())
    return drift, max(0.0, kl)


def anchor_kl_tensor(model,tok,anchor_prompts,logits0,device):
    logits=next_logits(model,tok,anchor_prompts,device)
    p0=torch.softmax(logits0,-1)
    return (p0*(torch.log_softmax(logits0,-1)-torch.log_softmax(logits,-1))).sum(-1).mean()


def kl_barrier(kl,cfg):
    onset=cfg["kl_barrier_onset"]
    gate=cfg["endpoint_kl_gate"]
    active=torch.clamp(kl-onset,min=0.0)
    gap=torch.clamp(gate-kl,min=cfg["kl_barrier_epsilon"])
    return cfg["kl_barrier_weight"]*active*active/gap


def kl_barrier_scalar(kl,cfg):
    active=max(0.0,kl-cfg["kl_barrier_onset"])
    gap=max(cfg["kl_barrier_epsilon"],cfg["endpoint_kl_gate"]-kl)
    return cfg["kl_barrier_weight"]*active*active/gap


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
              write_prompts, write_targets, exact_prompts, exact_eval_targets,
              train_para_prompts, read_prompts, paraphrase_eval_targets, candidate_ids,
              arm, seed, device, steps=None, selection_kl_gate=None):
    steps = steps or cfg["write_steps"]
    selection_kl_gate = selection_kl_gate or cfg["endpoint_kl_gate"]
    start = param_flat(params)
    rng = torch.Generator(device=device).manual_seed(seed + 99173)
    hist, accepted = [], 0
    initial_drift, initial_kl = anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
    last_eligible = start.clone()
    selected_step = 0
    selected_kl = initial_kl
    selected_drift = initial_drift
    with torch.no_grad():
        if arm == "exact_only":
            selected_train_score=float(memory_loss(
                model,tok,exact_prompts,write_targets[:len(exact_prompts)],
                candidate_ids,device).item())+kl_barrier_scalar(initial_kl,cfg)
        else:
            selected_train_score=float(semantic_memory_loss(
                model,tok,write_prompts,write_targets,candidate_ids,device,
                len(exact_prompts),cfg["semantic_pull_weight"],
                cfg["semantic_consistency_weight"],cfg["worst_item_weight"],
                cfg["worst_item_temperature"],cfg["margin_weight"],
                cfg["margin_target"])[0].item())+kl_barrier_scalar(initial_kl,cfg)
    selected_train_tuple=training_selection_tuple(
        model,tok,write_prompts,write_targets,candidate_ids,device,len(exact_prompts))
    for k in range(steps):
        before = param_flat(params)
        if arm == "exact_only":
            loss = memory_loss(model, tok, exact_prompts, write_targets[:len(exact_prompts)], candidate_ids, device)
            ce, pull, consistency = loss, loss.new_zeros(()), loss.new_zeros(())
        else:
            loss, ce, pull, consistency, robust_ce, robust_margin = semantic_memory_loss(
                model, tok, write_prompts, write_targets, candidate_ids, device,
                len(exact_prompts), cfg["semantic_pull_weight"],
                cfg["semantic_consistency_weight"],cfg["worst_item_weight"],
                cfg["worst_item_temperature"],cfg["margin_weight"],cfg["margin_target"])
        if arm == "exact_only":
            robust_ce, robust_margin = loss.new_zeros(()), loss.new_zeros(())
        current_kl=anchor_kl_tensor(model,tok,anchors,logits0,device)
        barrier=kl_barrier(current_kl,cfg)
        total_loss=loss+barrier
        g = flat(torch.autograd.grad(total_loss, params))
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
            ok = drift <= cfg["response_budget"] and kl <= cfg["endpoint_kl_gate"]
            if ok: break
            assign_flat(params, before)
            step *= 0.5
            if step < cfg.get("min_step", 0.0): break
        if ok: accepted += 1
        else:
            assign_flat(params, before); step = 0.0
            drift,kl=anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
        endpoint_eligible = ok and kl <= selection_kl_gate
        with torch.no_grad():
            if arm == "exact_only":
                post_score=float(memory_loss(model,tok,exact_prompts,
                    write_targets[:len(exact_prompts)],candidate_ids,device).item())+kl_barrier_scalar(kl,cfg)
            else:
                post_score=float(semantic_memory_loss(
                    model,tok,write_prompts,write_targets,candidate_ids,device,
                    len(exact_prompts),cfg["semantic_pull_weight"],
                    cfg["semantic_consistency_weight"],cfg["worst_item_weight"],
                    cfg["worst_item_temperature"],cfg["margin_weight"],
                    cfg["margin_target"])[0].item())+kl_barrier_scalar(kl,cfg)
        post_tuple=training_selection_tuple(
            model,tok,write_prompts,write_targets,candidate_ids,device,len(exact_prompts))
        better=(post_score < selected_train_score) if arm=="exact_only" else (post_tuple > selected_train_tuple)
        if endpoint_eligible and better:
            last_eligible = param_flat(params)
            selected_step = k + 1
            selected_kl = kl
            selected_drift = drift
            selected_train_score = post_score
            selected_train_tuple = post_tuple
        para_acc = read_accuracy(model, tok, read_prompts, paraphrase_eval_targets, candidate_ids, device)
        exact_acc = read_accuracy(model, tok, exact_prompts, exact_eval_targets, candidate_ids, device)
        objective_acc = read_accuracy(model, tok, write_prompts, write_targets, candidate_ids, device)
        train_para_targets=exact_eval_targets*(len(train_para_prompts)//len(exact_prompts))
        train_para_acc = read_accuracy(model, tok, train_para_prompts, train_para_targets, candidate_ids, device)
        hist.append({"step":k+1,"loss":float(loss.item()),"total_loss_with_kl_barrier":float(total_loss.item()),
                     "kl_barrier":float(barrier.item()),"candidate_ce":float(ce.item()),
                     "semantic_pull":float(pull.item()),"semantic_consistency":float(consistency.item()),
                     "worst_item_ce":float(robust_ce.item()),
                     "worst_item_margin_penalty":float(robust_margin.item()),
                     "training_selection_tuple":post_tuple,
                     "train_selection_score":post_score,
                     "paraphrase_accuracy":para_acc,"response_drift":drift,
                     "exact_accuracy":exact_acc,"write_objective_accuracy":objective_acc,
                     "train_paraphrase_accuracy":train_para_acc,
                     "anchor_kl_diagnostic":kl,"accepted":ok,"accepted_step_size":step,
                     "endpoint_eligible_checkpoint":endpoint_eligible,
                     "backtracking_halvings":halves,"retraction_norm":corr,
                     "kernel_residual":float((J@d).norm().item())})
    raw_drift, raw_kl = anchor_metrics(model,tok,anchors,response_ids,response0,logits0,device)
    raw_final_step = steps
    # Every arm is evaluated at its last prospectively eligible checkpoint.
    assign_flat(params,last_eligible)
    drift, kl = anchor_metrics(model, tok, anchors, response_ids, response0, logits0, device)
    exact_margins=signed_margins(model,tok,exact_prompts,exact_eval_targets,candidate_ids,device)
    para_margins=signed_margins(model,tok,read_prompts,paraphrase_eval_targets,candidate_ids,device)
    own_margins=signed_margins(model,tok,write_prompts,write_targets,candidate_ids,device)
    return {"exact_accuracy":read_accuracy(model,tok,exact_prompts,exact_eval_targets,candidate_ids,device),
            "train_paraphrase_accuracy":read_accuracy(model,tok,train_para_prompts,
                exact_eval_targets*(len(train_para_prompts)//len(exact_prompts)),candidate_ids,device),
            "train_paraphrase_view_accuracies":view_accuracies(model,tok,train_para_prompts,
                exact_eval_targets,candidate_ids,device,len(exact_prompts)),
            "paraphrase_accuracy":read_accuracy(model,tok,read_prompts,paraphrase_eval_targets,candidate_ids,device),
            "heldout_view_accuracies":view_accuracies(model,tok,read_prompts,
                exact_eval_targets,candidate_ids,device,len(exact_prompts)),
            "write_objective_accuracy":read_accuracy(model,tok,write_prompts,write_targets,candidate_ids,device),
            "write_objective_signed_margins":own_margins,
            "exact_signed_margins":exact_margins,
            "exact_median_signed_margin":float(np.median(exact_margins)),
            "paraphrase_signed_margins":para_margins,
            "paraphrase_median_signed_margin":float(np.median(para_margins)),
            "response_drift":drift,"anchor_kl_diagnostic":kl,"accepted_steps":accepted,
            "selected_checkpoint_step":selected_step,
            "selected_train_score":selected_train_score,
            "selected_training_tuple":selected_train_tuple,
            "selection_kl_gate":selection_kl_gate,
            "selected_checkpoint_was_truncated":selected_step < raw_final_step,
            "raw_final_response_drift":raw_drift,"raw_final_anchor_kl":raw_kl,
            "selected_recorded_response_drift":selected_drift,"selected_recorded_anchor_kl":selected_kl,
            "position_distance":float((param_flat(params)-start).norm().item()),"history":hist}


def run_l3_atlas(model, tok, params, cfg, anchors, response_ids, response0,
                 logits0, candidate_ids, initial, seed, device):
    """Average endpoint distances over cyclic code assignments to remove code confounding."""
    names=[x["name"] for x in L3_NODES]
    base_codes=[x["bits"] for x in L3_NODES]
    source_cpu=initial.detach().cpu()
    pair_keys=["--".join(sorted((a,b))) for i,a in enumerate(names) for b in names[i+1:]]
    codebooks=[]; distance_samples={k:[] for k in pair_keys}
    moved_distance_samples={k:[] for k in pair_keys}
    random_distance_samples={k:[] for k in pair_keys}
    transport_runs=[]
    node_runs={name:[] for name in names}
    for rotation in range(cfg["l3_code_rotations"]):
        endpoints={}; assignment={}
        codes=base_codes[rotation:]+base_codes[:rotation]
        for idx,(node,bits) in enumerate(zip(L3_NODES,codes)):
            assign_flat(params,initial)
            exact,train_para,held=concept_prompts(node)
            targets=[candidate_ids[b] for b in bits]
            result=run_write(
                model,tok,params,cfg,anchors,response_ids,response0,logits0,
                exact+train_para,targets+targets,exact,targets,train_para,held,targets,
                candidate_ids,"semantic_current",seed+900000+rotation*10007+idx*1009,
                device,steps=cfg["l3_write_steps"])
            endpoint=param_flat(params).detach().cpu(); endpoints[node["name"]]=endpoint
            assignment[node["name"]]=bits
            node_runs[node["name"]].append({
                "rotation":rotation,"bits":bits,"exact_accuracy":result["exact_accuracy"],
                "train_paraphrase_accuracy":result["train_paraphrase_accuracy"],
                "heldout_paraphrase_accuracy":result["paraphrase_accuracy"],
                "response_drift":result["response_drift"],
                "anchor_kl_diagnostic":result["anchor_kl_diagnostic"],
                "selected_checkpoint_step":result["selected_checkpoint_step"],
                "position_norm_from_source":float((endpoint-source_cpu).norm().item()),
                "position_sha256":hashlib.sha256(endpoint.numpy().tobytes()).hexdigest()})
        for i,a in enumerate(names):
            for b in names[i+1:]:
                distance_samples["--".join(sorted((a,b)))].append(float((endpoints[a]-endpoints[b]).norm().item()))
        move_sign=torch.tensor([1.0 if i%2==0 else -1.0 for i in range(response0.numel())],device=device)
        move_target=response0+cfg["l4_response_shift"]*move_sign/move_sign.norm()
        moved_endpoints={}; random_endpoints={}
        for idx,(node,bits) in enumerate(zip(L3_NODES,codes)):
            name=node["name"]; original=endpoints[name].to(device)
            exact,train_para,held=concept_prompts(node)
            targets=[candidate_ids[b] for b in bits]
            assign_flat(params,original)
            with torch.no_grad(): old_logits=next_logits(model,tok,anchors,device).detach()
            source_new_residual=float((response_vector(model,tok,anchors,response_ids,device)-move_target).abs().max().item())
            retract(model,tok,params,anchors,response_ids,move_target,device,cfg["l4_retraction_steps"])
            moved=param_flat(params); moved_endpoints[name]=moved.detach().cpu()
            true_residual,true_kl=anchor_metrics(model,tok,anchors,response_ids,move_target,old_logits,device)
            true_eval=evaluate_content(model,tok,exact,held,targets,candidate_ids,device)
            transport_norm=float((moved-original).norm().item())
            assign_flat(params,original)
            r=response_vector(model,tok,anchors,response_ids,device); J=jacobian_rows(r,params)
            gen=torch.Generator(device=device).manual_seed(seed+7000000+rotation*10007+idx)
            z=kernel_project(torch.randn(original.shape,generator=gen,device=device),J)
            z=z*(transport_norm/(z.norm()+1e-12))
            assign_flat(params,original+z)
            retract(model,tok,params,anchors,response_ids,move_target,device,cfg["l4_retraction_steps"])
            random_point=param_flat(params); random_endpoints[name]=random_point.detach().cpu()
            random_residual,random_kl=anchor_metrics(model,tok,anchors,response_ids,move_target,old_logits,device)
            random_eval=evaluate_content(model,tok,exact,held,targets,candidate_ids,device)
            transport_runs.append({"rotation":rotation,"concept":name,"bits":bits,
                "source_new_fibre_residual":source_new_residual,
                "transport_norm":transport_norm,"transport_response_residual":true_residual,
                "transport_anchor_kl_from_old_endpoint":true_kl,
                "transport_exact_accuracy":true_eval["exact_accuracy"],
                "transport_heldout_accuracy":true_eval["paraphrase_accuracy"],
                "random_response_residual":random_residual,
                "random_anchor_kl_from_old_endpoint":random_kl,
                "random_exact_accuracy":random_eval["exact_accuracy"],
                "random_heldout_accuracy":random_eval["paraphrase_accuracy"]})
        for i,a in enumerate(names):
            for b in names[i+1:]:
                key="--".join(sorted((a,b)))
                moved_distance_samples[key].append(float((moved_endpoints[a]-moved_endpoints[b]).norm().item()))
                random_distance_samples[key].append(float((random_endpoints[a]-random_endpoints[b]).norm().item()))
        codebooks.append(assignment)
    distances={k:float(np.mean(v)) for k,v in distance_samples.items()}
    train_edge="--".join(sorted(("paris","france")))
    heldout_edge="--".join(sorted(("france","europe")))
    heldout_controls=["--".join(sorted(x)) for x in [("paris","banana"),("france","banana"),("europe","banana")]]
    heldout_control_mean=float(np.mean([distances[k] for k in heldout_controls]))
    rng=random.Random(seed+424242)
    shuffled=[]
    for _ in range(cfg["l3_graph_shuffles"]):
        perm=names[:]; rng.shuffle(perm); mapping=dict(zip(names,perm))
        vals=[]
        for a,b in L3_EDGES:
            vals.append(distances["--".join(sorted((mapping[a],mapping[b])))])
        shuffled.append(float(np.mean(vals)))
    nodes={node["name"]:{"description":node["description"],"runs":node_runs[node["name"]]} for node in L3_NODES}
    all_runs=[x for runs in node_runs.values() for x in runs]
    related_mean=float(np.mean([distances["--".join(sorted(e))] for e in L3_EDGES]))
    unrelated=[v for k,v in distances.items() if k not in {"--".join(sorted(e)) for e in L3_EDGES}]
    unrelated_mean=float(np.mean(unrelated))
    gates={
        "all_concept_endpoints_eligible":all(x["response_drift"]<=cfg["response_budget"] and x["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"] for x in all_runs),
        "all_concepts_exact":all(x["exact_accuracy"]>=cfg["train_accuracy_gate"] for x in all_runs),
        "all_concepts_heldout":all(x["heldout_paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"] for x in all_runs),
        "development_edge_closer_than_unrelated":distances[train_edge] < unrelated_mean,
        "heldout_edge_closer_than_banana_controls":distances[heldout_edge] < cfg["l3_related_distance_ratio"]*heldout_control_mean,
        "related_positions_closer_than_unrelated":related_mean <= cfg["l3_related_distance_ratio"]*unrelated_mean,
        "graph_beats_median_shuffle":related_mean < float(np.median(shuffled))
    }
    moved_distances={k:float(np.mean(v)) for k,v in moved_distance_samples.items()}
    random_distances={k:float(np.mean(v)) for k,v in random_distance_samples.items()}
    true_distortion=float(np.mean([abs(moved_distances[k]/distances[k]-1.0) for k in pair_keys]))
    random_distortion=float(np.mean([abs(random_distances[k]/distances[k]-1.0) for k in pair_keys]))
    moved_heldout_control=float(np.mean([moved_distances[k] for k in heldout_controls]))
    l4_gates={
        "source_endpoints_leave_moved_fibre":all(x["source_new_fibre_residual"]>cfg["l4_response_budget"] for x in transport_runs),
        "all_transports_reach_moved_fibre":all(x["transport_response_residual"]<=cfg["l4_response_budget"] for x in transport_runs),
        "all_transports_within_collateral_kl":all(x["transport_anchor_kl_from_old_endpoint"]<=cfg["l4_collateral_kl_gate"] for x in transport_runs),
        "transport_preserves_exact_memory":all(x["transport_exact_accuracy"]>=cfg["train_accuracy_gate"] for x in transport_runs),
        "transport_preserves_heldout_access":all(x["transport_heldout_accuracy"]>=cfg["paraphrase_accuracy_gate"] for x in transport_runs),
        "transport_preserves_pair_geometry":true_distortion<=cfg["l4_geometry_distortion_gate"],
        "transport_preserves_heldout_relation":moved_distances[heldout_edge]<cfg["l3_related_distance_ratio"]*moved_heldout_control,
        "transport_beats_random_geometry":true_distortion+cfg["l4_random_distortion_margin"]<=random_distortion,
        "transport_read_not_worse_than_random":bool(np.mean([x["transport_heldout_accuracy"] for x in transport_runs])>=np.mean([x["random_heldout_accuracy"] for x in transport_runs]))
    }
    l4={"response_shift_norm":cfg["l4_response_shift"],"runs":transport_runs,
        "moved_pairwise_distances":moved_distances,"random_pairwise_distances":random_distances,
        "mean_relative_geometry_distortion":true_distortion,
        "random_mean_relative_geometry_distortion":random_distortion,
        "gates":l4_gates,"all_gates_pass":all(l4_gates.values())}
    return {"nodes":nodes,"code_assignments":codebooks,
            "distance_samples_by_rotation":distance_samples,
            "development_edge":["paris","france"],"heldout_edge":["france","europe"],
            "heldout_negative_edges":[["paris","banana"],["france","banana"],["europe","banana"]],
            "declared_edges":L3_EDGES,"pairwise_distances":distances,
            "related_mean_distance":related_mean,"unrelated_mean_distance":unrelated_mean,
            "heldout_control_mean_distance":heldout_control_mean,
            "shuffled_edge_mean_distances":shuffled,"gates":gates,
            "all_gates_pass":all(gates.values()),"l4_transport":l4}


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
    exact=[exact_prompt(key) for key in keys]
    train_sets={key:train_paraphrase_prompts(key) for key in keys}
    heldout_sets={key:heldout_paraphrase_prompts(key) for key in keys}
    train_para=[train_sets[key][view] for view in range(4) for key in keys]
    read=[heldout_sets[key][view] for view in range(3) for key in keys]
    targets=[candidate_ids[b] for b in bits]
    write_prompts=exact+train_para
    write_targets=targets*5
    read_targets=targets*3
    initial=param_flat(params)
    results={}
    for arm in cfg["arms"]:
        runs=[]
        repeats=cfg["best_of_random"] if arm=="random_kernel" else 1
        use_targets=write_targets
        if arm=="shuffled_label":
            # Deterministic wrong content; guaranteed to differ at every slot.
            wrong=[candidate_ids[1-b] for b in bits]
            use_targets=wrong*5
        for j in range(repeats):
            assign_flat(params, initial)
            runs.append(run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                                  write_prompts,use_targets,exact,targets,train_para,
                                  read,read_targets,candidate_ids,
                                  "semantic_current" if arm=="shuffled_label" else arm,seed+1009*j,device))
        results[arm]=max(runs,key=lambda x:x["paraphrase_median_signed_margin"]) if arm=="random_kernel" else runs[0]
    # Causal overwrite A -> not-A from the true endpoint.
    assign_flat(params, initial)
    source_cross={
        "read_as_A":evaluate_content(model,tok,exact,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,exact,read,[candidate_ids[b] for b in complement],candidate_ids,device)
    }
    first=run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                    write_prompts,write_targets,exact,targets,train_para,read,read_targets,candidate_ids,
                    "semantic_current",seed,device,
                    selection_kl_gate=cfg["overwrite_a_kl_reserve_gate"])
    at_a=param_flat(params)
    comp_targets=[candidate_ids[b] for b in complement]
    a_cross={
        "read_as_A":evaluate_content(model,tok,exact,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,exact,read,comp_targets,candidate_ids,device)
    }
    second=run_write(model,tok,params,cfg,anchors,response_ids,response0,logits0,
                     write_prompts,comp_targets*5,exact,comp_targets,train_para,read,comp_targets*3,candidate_ids,
                     "semantic_current",seed+700001,device,
                     steps=cfg["rewrite_steps"])
    second["distance_from_A"]=float((param_flat(params)-at_a).norm().item())
    results["overwrite_A_to_notA"]={"A_endpoint":first,"notA_endpoint":second}
    nota_cross={
        "read_as_A":evaluate_content(model,tok,exact,read,targets,candidate_ids,device),
        "read_as_notA":evaluate_content(model,tok,exact,read,comp_targets,candidate_ids,device)
    }
    cross_read_matrix={"source":source_cross,"A_endpoint":a_cross,"notA_endpoint":nota_cross}
    tr=results["semantic_current"]; rr=results["random_kernel"]; eo=results["exact_only"]
    control_eligibility={name:{
        "response_eligible":results[name]["response_drift"]<=cfg["response_budget"],
        "endpoint_kl_eligible":results[name]["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "eligible":results[name]["response_drift"]<=cfg["response_budget"] and results[name]["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"]
    } for name in cfg["arms"]}
    primary_gates={
        "semantic_response_budget":tr["response_drift"]<=cfg["response_budget"],
        "semantic_endpoint_kl_gate":tr["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "semantic_exact_accuracy":tr["exact_accuracy"]>=cfg["train_accuracy_gate"],
        "semantic_train_view_accuracy":tr["train_paraphrase_accuracy"]>=cfg["train_accuracy_gate"],
        "every_semantic_train_view_accurate":all(x>=cfg["train_accuracy_gate"] for x in tr["train_paraphrase_view_accuracies"]),
        "semantic_beats_best_random_exact_margin":tr["exact_median_signed_margin"]-rr["exact_median_signed_margin"]>=cfg["true_random_signed_margin_gate"],
        "all_control_arms_eligible":all(x["eligible"] for x in control_eligibility.values()),
        "overwrite_A_start_exact_accuracy":first["exact_accuracy"]>=cfg["train_accuracy_gate"],
        "overwrite_A_start_reserve_kl":first["anchor_kl_diagnostic"]<=cfg["overwrite_a_kl_reserve_gate"],
        "overwrite_exact_accuracy":second["exact_accuracy"]>=cfg["train_accuracy_gate"],
        "overwrite_endpoint_kl_gate":second["anchor_kl_diagnostic"]<=cfg["endpoint_kl_gate"],
        "overwrite_response_budget":second["response_drift"]<=cfg["response_budget"],
        "cross_read_switch":a_cross["read_as_A"]["exact_accuracy"]==1.0 and a_cross["read_as_notA"]["exact_accuracy"]==0.0 and nota_cross["read_as_notA"]["exact_accuracy"]==1.0 and nota_cross["read_as_A"]["exact_accuracy"]==0.0
    }
    secondary_gates={
        "semantic_heldout_paraphrase_accuracy":tr["paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"],
        "every_heldout_view_accurate":all(x>=cfg["paraphrase_accuracy_gate"] for x in tr["heldout_view_accuracies"]),
        "semantic_beats_exact_only_worst_heldout_margin":min(tr["paraphrase_signed_margins"])-min(eo["paraphrase_signed_margins"])>=cfg["semantic_exact_only_margin_gate"],
        "overwrite_A_start_heldout_accuracy":first["paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"],
        "overwrite_A_start_every_heldout_view":all(x>=cfg["paraphrase_accuracy_gate"] for x in first["heldout_view_accuracies"]),
        "overwrite_heldout_paraphrase_accuracy":second["paraphrase_accuracy"]>=cfg["paraphrase_accuracy_gate"],
        "overwrite_every_heldout_view_accurate":all(x>=cfg["paraphrase_accuracy_gate"] for x in second["heldout_view_accuracies"])
    }
    gates={**primary_gates,**secondary_gates}
    record={"seed":seed,"device":str(device),"symbols":[sym0,sym1],"slot_keys":keys,"bits_A":bits,
            "bits_notA":complement,"prompts":{"exact":exact,"semantic_train":train_para,"heldout":read},
            "anchors":anchors,"response_token_ids":response_ids,
            "arms":results,"control_eligibility":control_eligibility,
            "cross_read_matrix":cross_read_matrix,
            "primary_gates":primary_gates,"secondary_gates":secondary_gates,
            "gates":gates,
            "all_primary_gates_pass":all(primary_gates.values()),
            "all_secondary_gates_pass":all(secondary_gates.values()),
            "all_gates_pass":all(gates.values())}
    (output/f"seed_{seed}.json").write_text(json.dumps(record,indent=2,default=json_default),encoding="utf-8")
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
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=json_default),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=json_default))


if __name__=="__main__": main()
