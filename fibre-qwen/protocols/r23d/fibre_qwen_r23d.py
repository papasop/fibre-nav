#!/usr/bin/env python3
"""R23d: frozen five-seed confirmation of the R23c-r2 compute-core signal."""
import argparse, json, math, random, statistics, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL = "FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23D_CONFIRMATORY"
DEVELOPMENT_REFERENCE = "FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23C_R2_PRECISION_AUDIT"
MODEL = "Qwen/Qwen3-0.6B"
SEEDS = [64007, 64013, 64019, 64033, 64037]
STEPS = 24
STEP_NORM = 0.018
RANK_TOL = 1e-8
PRECISION_FLOOR = 1e-7
MAX_BACKTRACKS = 12
BUDGETS = [0.002660127151245491, 0.005320254302490982]

TRAIN = [
    ("用一句话解释什么是二分查找。", "二分查找在有序序列中反复排除一半候选区间。"),
    ("为什么相关性不等于因果性？", "相关性可能由混杂变量、反向因果或偶然性造成，不能单独识别因果方向。"),
    ("写出水的化学式。", "水的化学式是 H2O。"),
    ("Python 中字典的主要用途是什么？", "字典用键到值的映射来组织和快速查找数据。"),
    ("给出一个防止过拟合的方法。", "可使用独立验证集、正则化、数据增强或早停来降低过拟合。"),
    ("什么是可证伪性？", "可证伪性要求主张给出可能被观察或实验否定的明确条件。"),
    ("解释均值与中位数的区别。", "均值使用全部数值求平均，中位数是排序后的中间位置，对极端值通常更稳健。"),
    ("什么是软件单元测试？", "单元测试对最小可测试组件的预期行为进行自动验证。"),
    ("为什么实验需要对照组？", "对照组帮助区分干预效应与背景变化、随机波动或其他因素。"),
    ("简述梯度下降。", "梯度下降沿目标函数局部最陡下降方向的反方向迭代更新参数。"),
    ("遇到证据不足时应该怎样回答？", "应明确说明不确定性、已有证据和缺失证据，避免把猜测写成事实。"),
    ("什么是哈希值的用途？", "哈希值可用于检测内容是否改变，但它本身通常不能证明内容真实。"),
]
VALID = [
    ("为什么测试集不应参与调参？", "测试集应保留用于最终泛化评估，参与调参会造成信息泄漏和偏乐观估计。"),
    ("什么是递归？", "递归是函数或过程通过调用自身来解决规模更小的同类问题。"),
    ("解释科学结论的适用边界。", "适用边界说明结论在哪些对象、条件、数据和协议范围内成立。"),
    ("为什么要报告失败实验？", "失败实验能界定主张边界、减少选择性报告并指导后续检验。"),
]
PROBES = [
    ("判断：单个任务上的改进足以证明方法普遍优越。只回答 A(否) 或 B(是)。", "A", "B"),
    ("判断：相关性本身必然证明因果关系。只回答 A(否) 或 B(是)。", "A", "B"),
    ("判断：独立测试集可以降低选择性调参造成的偏差。只回答 A(否) 或 B(是)。", "B", "A"),
    ("判断：哈希一致就能证明科学结论正确。只回答 A(否) 或 B(是)。", "A", "B"),
    ("判断：有限步变化可由一阶导数为零自动保证。只回答 A(否) 或 B(是)。", "A", "B"),
    ("判断：明确拒绝门槛有助于可证伪性。只回答 A(否) 或 B(是)。", "B", "A"),
]

def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def trainable(model): return [p for p in model.parameters() if p.requires_grad]
def flat(xs): return torch.cat([x.reshape(-1) for x in xs])

@torch.no_grad()
def get_vec(ps): return flat([p.detach().float() for p in ps])

@torch.no_grad()
def set_vec(ps, vector):
    offset = 0
    for p in ps:
        n = p.numel(); p.copy_(vector[offset:offset+n].view_as(p).to(p.dtype)); offset += n

def prompt_text(tok, question):
    return tok.apply_chat_template([{"role":"user","content":question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def supervised_loss(model, tok, pairs, device):
    losses = []
    for question, answer in pairs:
        prefix = prompt_text(tok, question); full = prefix + answer + tok.eos_token
        enc = tok(full, return_tensors="pt", truncation=True, max_length=256).to(device)
        labels = enc.input_ids.clone()
        plen = tok(prefix, return_tensors="pt", truncation=True, max_length=256).input_ids.shape[1]
        labels[:, :plen] = -100
        losses.append(model(**enc, labels=labels).loss.float())
    return torch.stack(losses).mean()

def response_coordinate(model, tok, device, probe):
    question, good, bad = probe
    enc = tok(prompt_text(tok, question), return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    good_ids = tok.encode(good, add_special_tokens=False); bad_ids = tok.encode(bad, add_special_tokens=False)
    if len(good_ids) != 1 or len(bad_ids) != 1:
        raise RuntimeError(f"probe verbalizers must be one token: {good}/{bad}")
    return logits[good_ids[0]] - logits[bad_ids[0]]

def response_vector(model, tok, device):
    with torch.no_grad():
        return torch.stack([response_coordinate(model, tok, device, p) for p in PROBES]).double()

def jacobian(model, tok, ps, device):
    rows = []
    for probe in PROBES:
        response = response_coordinate(model, tok, device, probe)
        grads = torch.autograd.grad(response, ps, allow_unused=False)
        rows.append(flat([g.detach().float() for g in grads]))
    return torch.stack(rows).double()

def project(vector, jac, damping=1e-10):
    vd = vector.double(); gram = jac @ jac.T
    eye = torch.eye(len(jac), device=jac.device, dtype=jac.dtype)
    coeff = torch.linalg.solve(gram + damping * eye, jac @ vd)
    return (vd - jac.T @ coeff).float()

def rowspace_distance(j0, j1):
    q0 = torch.linalg.qr(j0.T, mode="reduced").Q; q1 = torch.linalg.qr(j1.T, mode="reduced").Q
    rank = min(q0.shape[1], q1.shape[1]); overlap = torch.linalg.matrix_norm(q0.T @ q1).square()
    return float(torch.sqrt(torch.clamp(2*rank - 2*overlap, min=0.0)))

def eval_loss(model, tok, device): return float(supervised_loss(model, tok, VALID, device).detach())

def run_arm(model, tok, ps, source, source_response, source_jac, arm, budget, seed, device):
    set_vec(ps, source); seed_all(seed)
    first = torch.zeros_like(source); second = torch.zeros_like(source); records = []; accepted = 0
    order = [i % len(TRAIN) for i in torch.randperm(STEPS).tolist()]
    for step in range(STEPS):
        loss = supervised_loss(model, tok, [TRAIN[order[step]]], device)
        gradient = flat([x.detach().float() for x in torch.autograd.grad(loss, ps)])
        first = .9*first + .1*gradient; second = .999*second + .001*gradient.square()
        mh = first/(1-.9**(step+1)); vh = second/(1-.999**(step+1))
        direction = -mh/(vh.sqrt()+1e-8) - .01*get_vec(ps)
        current_jac = jacobian(model, tok, ps, device) if arm == "moving_current_kernel" else source_jac
        residual = None
        if arm != "budgeted_lora_adamw":
            direction = project(direction, current_jac)
            residual = float(torch.linalg.vector_norm(current_jac @ direction.double()) /
                             direction.double().norm().clamp_min(1e-30))
        direction = direction / direction.norm().clamp_min(1e-12) * STEP_NORM
        before = get_vec(ps); scale = 1.0; ok = False
        for _ in range(MAX_BACKTRACKS + 1):
            set_vec(ps, before + scale*direction)
            drift = float(torch.linalg.vector_norm(response_vector(model, tok, device)-source_response))
            if math.isfinite(drift) and drift <= budget + 1e-7:
                ok = True; accepted += 1; break
            scale *= .5
        if not ok:
            set_vec(ps, before)
            drift = float(torch.linalg.vector_norm(response_vector(model, tok, device)-source_response))
        if step in {0, STEPS-1} or (step+1) % 6 == 0:
            print(f"[seed={seed} {arm} B={budget:.3e}] {step+1}/{STEPS} "
                  f"loss={float(loss.detach()):.5f} drift={drift:.6e}", flush=True)
        records.append({"step":step+1, "train_loss":float(loss.detach()), "drift":drift,
                        "scale":scale if ok else 0.0, "projection_residual":residual})
    endpoint_response = response_vector(model, tok, device); endpoint_jac = jacobian(model, tok, ps, device)
    return {"seed":seed, "arm":arm, "budget":budget,
            "final_validation_loss":eval_loss(model, tok, device),
            "final_response_drift":float(torch.linalg.vector_norm(endpoint_response-source_response)),
            "response_budget_utilization":float(torch.linalg.vector_norm(endpoint_response-source_response))/budget,
            "accepted_steps":accepted, "rowspace_distance_from_source":rowspace_distance(source_jac, endpoint_jac),
            "max_projection_residual":max((x["projection_residual"] or 0.0) for x in records), "trace":records}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("CUDA required")
    device = torch.device(args.device); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    seed_all(63017)
    print(f"protocol={PROTOCOL} model={MODEL} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    layers = len(model.model.layers)
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                        target_modules=["q_proj","v_proj"], layers_to_transform=[layers-2,layers-1])
    model = get_peft_model(model, config); model.train(); ps = trainable(model)
    source = get_vec(ps); source_response = response_vector(model, tok, device); source_jac = jacobian(model, tok, ps, device)
    singular = torch.linalg.svdvals(source_jac)
    response_rank = int((singular > RANK_TOL*singular.max().clamp_min(1.0)).sum())
    print(f"[freeze] seeds={SEEDS} budgets={BUDGETS} trainable={source.numel()} rank={response_rank}", flush=True)
    runs = []
    for index, seed in enumerate(SEEDS, 1):
        print(f"[eval {index}/{len(SEEDS)}] seed={seed}", flush=True)
        for budget in BUDGETS:
            for arm in ["moving_current_kernel", "source_frozen_kernel", "budgeted_lora_adamw"]:
                runs.append(run_arm(model, tok, ps, source, source_response, source_jac, arm, budget, seed, device))

    pairs = []
    seed_support = []
    for seed in SEEDS:
        supports = True; comparisons = []
        for budget in BUDGETS:
            selected = {r["arm"]:r for r in runs if r["seed"] == seed and r["budget"] == budget}
            source_delta = selected["source_frozen_kernel"]["final_validation_loss"]-selected["moving_current_kernel"]["final_validation_loss"]
            adam_delta = selected["budgeted_lora_adamw"]["final_validation_loss"]-selected["moving_current_kernel"]["final_validation_loss"]
            this_support = source_delta > 0 and adam_delta > 0
            supports = supports and this_support
            comparisons.append({"budget":budget, "source_minus_moving_validation_loss":source_delta,
                                "adamw_minus_moving_validation_loss":adam_delta, "supports_budget":this_support})
        seed_support.append(supports); pairs.append({"seed":seed, "comparisons":comparisons, "supports_both_budgets":supports})

    median_deltas = {}
    for budget in BUDGETS:
        selected = [c for p in pairs for c in p["comparisons"] if c["budget"] == budget]
        median_deltas[str(budget)] = {
            "source_minus_moving_validation_loss":statistics.median(c["source_minus_moving_validation_loss"] for c in selected),
            "adamw_minus_moving_validation_loss":statistics.median(c["adamw_minus_moving_validation_loss"] for c in selected)}
    projected = [r["max_projection_residual"] for r in runs if r["arm"] != "budgeted_lora_adamw"]
    moving = [r for r in runs if r["arm"] == "moving_current_kernel"]
    numerical = {
        "all_finite":all(math.isfinite(r["final_validation_loss"]) and math.isfinite(r["final_response_drift"]) for r in runs),
        "all_budgets_respected":all(r["final_response_drift"] <= r["budget"] + 1e-6 for r in runs),
        "response_rank_six":response_rank == 6,
        "measured_drifts_not_all_zero":any(x["drift"] > PRECISION_FLOOR for r in runs for x in r["trace"]),
        "current_rowspace_moves":min(r["rowspace_distance_from_source"] for r in moving) > PRECISION_FLOOR,
        "projected_direction_residual_at_most_1e-5":max(projected) <= 1e-5,
    }
    gates = {
        "five_untouched_confirmation_seeds":len(SEEDS) == 5,
        "frozen_two_budgets_exact":BUDGETS == [0.002660127151245491, 0.005320254302490982],
        "all_worker_numerical_gates":all(numerical.values()),
        "at_least_four_of_five_seeds_support_both_budgets":sum(seed_support) >= 4,
        "median_source_minus_moving_positive_at_both_budgets":all(v["source_minus_moving_validation_loss"] > 0 for v in median_deltas.values()),
        "median_adamw_minus_moving_positive_at_both_budgets":all(v["adamw_minus_moving_validation_loss"] > 0 for v in median_deltas.values()),
    }
    passed = all(gates.values())
    summary = {
        "protocol":PROTOCOL, "mode":"untouched_seed_confirmatory", "development_reference":DEVELOPMENT_REFERENCE,
        "model":MODEL, "seeds":SEEDS,
        "frozen_configuration":{"budgets":BUDGETS, "steps":STEPS, "step_norm":STEP_NORM,
                                "rank_tolerance":RANK_TOL, "precision_floor":PRECISION_FLOOR,
                                "max_backtracks":MAX_BACKTRACKS, "lora_rank":4,
                                "target_modules":["q_proj","v_proj"], "last_layers":2,
                                "learning_records":12, "validation_records":4, "response_coordinates":6},
        "trainable_lora_parameters":source.numel(), "response_rank":response_rank,
        "pairs":pairs, "supporting_seed_count":sum(seed_support), "median_deltas":median_deltas,
        "moving_budget_utilization":{"median":statistics.median(r["response_budget_utilization"] for r in moving),
                                     "maximum":max(r["response_budget_utilization"] for r in moving)},
        "numerical_gates":numerical, "gates":gates, "runs":runs, "wall_seconds":time.time()-t0,
        "scientific_status":"R23D_MOVING_RESPONSE_KERNEL_CONFIRMED" if passed else "R23D_MOVING_RESPONSE_KERNEL_INCONCLUSIVE_FAIL_CLOSED",
        "claim_boundary":"Frozen five-seed confirmation within Qwen3-0.6B, 12 tiny authored learning records, four validation records, six disjoint response coordinates and one restricted 40960-parameter LoRA chart. It tests the R23c-r2 moving-current-kernel ordering at two frozen global response budgets only; it is not broad capability, personalization, continual learning, safety, deployment readiness, universal optimizer superiority or a theorem about response fibres."
    }
    (out/"run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "runs"}, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2

if __name__ == "__main__": raise SystemExit(main())
