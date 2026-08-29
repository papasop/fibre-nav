#!/usr/bin/env python3
"""GPTW-v3: held-out natural-text moving-current versus source-fixed LoRA audit."""
from __future__ import annotations

import argparse, json, random, shutil, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL = "CNER_GPT2_LORA_NATURAL_PROMPT_MOVING_FIBRE_V3"
MODEL_IDS = ["openai-community/gpt2", "gpt2"]
SEEDS = [38171, 38172, 38173, 38174, 38175, 38176, 38177, 38178]
LORA_RANK, LAYERS = 2, [10, 11]
AUDIT_RADII = [0.20, 0.10, 0.05]
PATH_STEP, INTERIOR_NODES, RETRACTION_ROUNDS = 0.20, 4, 3
REQUIRED_INSTANCES, REQUIRED_NODES = 6, 3
MIN_CURRENT_SLOPE, MAX_FIXED_SLOPE, MIN_SLOPE_GAP = 1.70, 1.40, 0.50
MIN_ACTIVE_RATIO, MIN_FINE_COST_RATIO = 100.0, 3.0
MAX_CURRENT_KERNEL_RESIDUAL, MAX_PATH_RESPONSE_ERROR = 1e-5, 1e-10
WALL_CLOCK_LIMIT_SECONDS = 3300

# Frozen natural-language development set. These prompts choose the tangent direction only.
DEVELOPMENT = [
    ("The gardener watered the dry soil before planting the", " seeds"),
    ("When the lights went out, Mira reached for a", " candle"),
    ("The hikers checked the map before crossing the", " river"),
    ("A mechanic tightened the loose bolt with a", " wrench"),
    ("The teacher wrote the final equation on the", " board"),
    ("Because the soup was hot, Daniel waited for it to", " cool"),
    ("The pianist opened the score and began to", " play"),
    ("At dawn, the fishing boats left the", " harbor"),
]

# Disjoint held-out natural continuations define the response map.
# Each target/distractor pair contributes target logit and target-minus-distractor margin.
CONFIRMATION = [
    ("After walking all day, the tired traveler wanted to", " rest", " argue"),
    ("The baker placed the warm bread on the wooden", " table", " cloud"),
    ("Rain gathered on the window during the", " storm", " concert"),
    ("The physician listened carefully to the patient's", " heart", " suitcase"),
    ("To see the stars clearly, they drove beyond the city", " lights", " kitchen"),
    ("The child zipped his coat before stepping into the", " snow", " oven"),
    ("After reading the letter twice, she placed it in the", " drawer", " ocean"),
    ("The train slowed as it approached the crowded", " station", " forest"),
]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class NativeLoRAConv1D(torch.nn.Module):
    def __init__(self, base, rank=2, alpha=4):
        super().__init__(); self.base = base; self.scale = alpha / rank
        for p in base.parameters(): p.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.empty(int(base.weight.shape[0]), rank), requires_grad=False)
        self.B = torch.nn.Parameter(torch.zeros(rank, int(base.weight.shape[1])))
        torch.nn.init.normal_(self.A, std=0.02)

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale


def build(device):
    errors = []
    for model_id in MODEL_IDS:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
            break
        except Exception as exc:
            errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("GPT-2 load failed:\n" + "\n".join(errors))
    tok.pad_token = tok.eos_token
    for p in model.parameters(): p.requires_grad_(False)
    for layer in LAYERS:
        model.transformer.h[layer].attn.c_attn = NativeLoRAConv1D(
            model.transformer.h[layer].attn.c_attn, LORA_RANK, 4
        )
    model.to(device=device, dtype=torch.float64).eval()
    params = [model.transformer.h[i].attn.c_attn.B for i in LAYERS]
    return model_id, tok, model, params


def flat(params): return torch.cat([p.detach().reshape(-1) for p in params])


def assign(params, vector):
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel(); p.copy_(vector[offset:offset+n].view_as(p)); offset += n


def grads(value, params, retain=False):
    gs = torch.autograd.grad(value, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gs, params)])


def unit(v): return v / v.norm().clamp_min(1e-30)


def token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Frozen completion must be one GPT-2 token: {text!r} -> {ids}")
    return ids[0]


def next_logits(model, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    return model(input_ids=ids).logits[0, -1]


def development_loss(model, tok, device):
    rows = []
    for prompt, target in DEVELOPMENT:
        z = next_logits(model, tok, prompt, device)
        rows.append(torch.nn.functional.cross_entropy(z[None, :], torch.tensor([token_id(tok, target)], device=device)))
    return torch.stack(rows).mean()


def response(model, tok, device):
    values = []
    for prompt, target, distractor in CONFIRMATION:
        z = next_logits(model, tok, prompt, device)
        ti, di = token_id(tok, target), token_id(tok, distractor)
        values.extend([z[ti], z[ti] - z[di]])
    return torch.stack(values)


def jacobian(model, tok, params, device):
    r = response(model, tok, device)
    J = torch.stack([grads(r[i], params, retain=i + 1 < len(r)) for i in range(len(r))])
    return r.detach(), J


def solve_normal(J, error):
    gram = J @ J.T
    scale = float(torch.trace(gram) / len(gram))
    ridge = 1e-10 * max(scale, 1e-12)
    eye = torch.eye(len(gram), device=J.device, dtype=J.dtype)
    return -J.T @ torch.linalg.solve(gram + ridge * eye, error)


def project(v, J): return v + solve_normal(J, J @ v)


def current_direction(model, tok, params, J, device):
    return unit(project(-grads(development_loss(model, tok, device), params), J))


def finite_cost(model, tok, params, theta, r_target, J, direction, radius, device):
    assign(params, theta + radius * direction)
    error = response(model, tok, device).detach() - r_target
    assign(params, theta)
    correction = solve_normal(J, error)
    return {
        "radius": radius,
        "active_response_residual": float(error.norm() / r_target.norm().clamp_min(1e-30)),
        "linear_newton_correction_norm": float(correction.norm()),
        "correction_to_step_ratio": float(correction.norm() / (radius * direction.norm()).clamp_min(1e-30)),
    }


def retract(model, tok, params, theta, source_response, device):
    q, trace = theta.clone(), []
    for _ in range(RETRACTION_ROUNDS):
        assign(params, q); r, J = jacobian(model, tok, params, device)
        error = r - source_response; correction = solve_normal(J, error); q = q + correction
        trace.append({"relative_error_before": float(error.norm()/source_response.norm().clamp_min(1e-30)), "correction_norm": float(correction.norm())})
    assign(params, q)
    final = float((response(model, tok, device).detach()-source_response).norm()/source_response.norm().clamp_min(1e-30))
    return q, trace, final


def slope(rows):
    x = np.log([row["radius"] for row in rows])
    y = np.log([max(row["linear_newton_correction_norm"], 1e-30) for row in rows])
    fit = np.polyfit(x, y, 1); pred = np.polyval(fit, x)
    sse, sst = float(np.sum((y-pred)**2)), float(np.sum((y-y.mean())**2))
    return float(fit[0]), (float(1-sse/sst) if sst > 0 else 1.0)


def run_seed(seed, device):
    seed_all(seed); model_id, tok, model, params = build(device)
    theta = flat(params); assign(params, theta); source_response, J0 = jacobian(model, tok, params, device)
    fixed_direction = current_direction(model, tok, params, J0, device)
    theta, trace, err = retract(model, tok, params, theta + PATH_STEP*fixed_direction, source_response, device)
    retractions, nodes = [{"advance": 0, "trace": trace, "final_relative_error": err}], []
    for node in range(INTERIOR_NODES):
        assign(params, theta); r_node, J = jacobian(model, tok, params, device)
        current = current_direction(model, tok, params, J, device)
        current_costs = [finite_cost(model,tok,params,theta,r_node,J,current,h,device) for h in AUDIT_RADII]
        fixed_costs = [finite_cost(model,tok,params,theta,r_node,J,fixed_direction,h,device) for h in AUDIT_RADII]
        current_slope, current_r2 = slope(current_costs); fixed_slope, fixed_r2 = slope(fixed_costs)
        current_residual = float((J@current).norm()/J.norm().clamp_min(1e-30))
        fixed_active = float((J@fixed_direction).norm()/J.norm().clamp_min(1e-30))
        active_ratio = fixed_active/max(current_residual,1e-30)
        fine_ratio = fixed_costs[-1]["linear_newton_correction_norm"]/max(current_costs[-1]["linear_newton_correction_norm"],1e-30)
        slope_gap = current_slope-fixed_slope
        gate = bool(current_residual <= MAX_CURRENT_KERNEL_RESIDUAL and active_ratio >= MIN_ACTIVE_RATIO and
                    fine_ratio >= MIN_FINE_COST_RATIO and current_slope >= MIN_CURRENT_SLOPE and
                    fixed_slope <= MAX_FIXED_SLOPE and slope_gap >= MIN_SLOPE_GAP)
        nodes.append({"node":node+1,"current_kernel_relative_residual":current_residual,"fixed_active_relative_residual":fixed_active,
            "fixed_to_current_active_ratio":active_ratio,"current_costs":current_costs,"fixed_costs":fixed_costs,
            "current_cost_slope":current_slope,"current_cost_r2":current_r2,"fixed_cost_slope":fixed_slope,
            "fixed_cost_r2":fixed_r2,"slope_gap":slope_gap,"fine_fixed_to_current_cost_ratio":fine_ratio,"node_gate":gate})
        theta, trace, err = retract(model,tok,params,theta+PATH_STEP*current,source_response,device)
        retractions.append({"advance":node+1,"trace":trace,"final_relative_error":err})
    passing = sum(row["node_gate"] for row in nodes); max_error = max(row["final_relative_error"] for row in retractions)
    supported = passing >= REQUIRED_NODES and max_error <= MAX_PATH_RESPONSE_ERROR
    result = {"seed":seed,"model":model_id,"nodes":nodes,"retractions":retractions,"passing_nodes":passing,
              "required_nodes":REQUIRED_NODES,"maximum_path_response_error":max_error,"instance_supported":supported}
    del model; torch.cuda.empty_cache(); return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="cner_gpt2_lora_natural_prompt_moving_fibre_v3_results")
    args, _ = ap.parse_known_args()
    if not torch.cuda.is_available(): raise RuntimeError("GPTW-v3 requires CUDA; select an A100 runtime")
    if "A100" not in torch.cuda.get_device_name(0).upper():
        print(f"WARNING: designed for A100; detected {torch.cuda.get_device_name(0)}", flush=True)
    device=torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); rows=[]
    for index, seed in enumerate(SEEDS):
        if time.monotonic()-started > WALL_CLOCK_LIMIT_SECONDS: raise TimeoutError("GPTW-v3 exceeded 55-minute hard limit")
        print(f"[GPTW-v3 natural confirmation {index+1}/{len(SEEDS)}] seed={seed}",flush=True)
        row=run_seed(seed,device); rows.append(row); (out/f"seed_{seed}.json").write_text(json.dumps(row,indent=2))
        print(json.dumps({"seed":seed,"passing_nodes":row["passing_nodes"],"supported":row["instance_supported"]}),flush=True)
    elapsed=time.monotonic()-started; supporting=sum(row["instance_supported"] for row in rows); passed=supporting>=REQUIRED_INSTANCES
    gates={"audit_radii":AUDIT_RADII,"path_step":PATH_STEP,"interior_nodes_per_seed":INTERIOR_NODES,
        "minimum_current_cost_slope":MIN_CURRENT_SLOPE,"maximum_fixed_cost_slope":MAX_FIXED_SLOPE,
        "minimum_slope_gap":MIN_SLOPE_GAP,"minimum_fixed_to_current_active_ratio":MIN_ACTIVE_RATIO,
        "minimum_fine_cost_ratio":MIN_FINE_COST_RATIO,"maximum_current_kernel_residual":MAX_CURRENT_KERNEL_RESIDUAL,
        "maximum_path_response_error":MAX_PATH_RESPONSE_ERROR,"required_passing_nodes_per_seed":REQUIRED_NODES,
        "required_supporting_instances":REQUIRED_INSTANCES}
    report={"protocol":PROTOCOL,"scientific_status":"PROSPECTIVE_NATURAL_TEXT_AUDIT_COMPLETED",
        "decision":"NATURAL_TEXT_CURRENT_FIBRE_ADVANTAGE_SUPPORTED" if passed else "NATURAL_TEXT_CURRENT_FIBRE_ADVANTAGE_NOT_SUPPORTED",
        "seeds":SEEDS,"supporting_instances":supporting,"required_instances":REQUIRED_INSTANCES,"total_interior_nodes":len(SEEDS)*INTERIOR_NODES,
        "elapsed_seconds":elapsed,"wall_clock_limit_seconds":WALL_CLOCK_LIMIT_SECONDS,"frozen_gates":gates,
        "instance_summaries":[{"seed":r["seed"],"passing_nodes":r["passing_nodes"],"maximum_path_response_error":r["maximum_path_response_error"],
          "instance_supported":r["instance_supported"],"node_summaries":[{k:n[k] for k in ("node","fixed_to_current_active_ratio","current_cost_slope","fixed_cost_slope","slope_gap","fine_fixed_to_current_cost_ratio","node_gate")} for n in r["nodes"]]} for r in rows],
        "claim_boundary":"Prospective eight-seed confirmation on disjoint, frozen natural-English continuations in rank-2 LoRA-B for GPT-2's final two blocks. A pass upgrades cross-modal evidence beyond synthetic codeword prompts, but does not establish semantic invariance, full-parameter transport, optimizer convergence, or a global variational principle."}
    (out/"report.json").write_text(json.dumps(report,indent=2))
    (out/"protocol.json").write_text(json.dumps({"protocol":PROTOCOL,"prospective":True,"seeds":SEEDS,"development":DEVELOPMENT,"confirmation":CONFIRMATION,"frozen_gates":gates},indent=2))
    print("="*96); print(json.dumps(report,indent=2)); shutil.make_archive(str(out),"zip",out.parent,out.name); print(f"Results ZIP: {out}.zip")


if __name__ == "__main__": main()
