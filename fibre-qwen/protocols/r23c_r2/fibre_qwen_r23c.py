#!/usr/bin/env python3
"""R23c: actual DR/kernel/budget compute-core, one-seed general-instruction smoke test."""
import argparse, json, math, random, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL = "FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23C_R2_PRECISION_AUDIT"
MODEL = "Qwen/Qwen3-0.6B"
SEED = 63017
STEPS = 24
WARM_STEPS = 4
STEP_NORM = 0.018
RANK_TOL = 1e-8
PRECISION_FLOOR = 1e-7
MAX_BACKTRACKS = 12

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

def seed_all(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]

def flat(xs):
    return torch.cat([x.reshape(-1) for x in xs])

@torch.no_grad()
def get_vec(ps):
    return flat([p.detach().float() for p in ps])

@torch.no_grad()
def set_vec(ps, v):
    k = 0
    for p in ps:
        n = p.numel(); p.copy_(v[k:k+n].view_as(p).to(p.dtype)); k += n

def prompt_text(tok, q):
    return tok.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def supervised_loss(model, tok, pairs, device):
    losses = []
    for q, a in pairs:
        prefix = prompt_text(tok, q)
        full = prefix + a + tok.eos_token
        enc = tok(full, return_tensors="pt", truncation=True, max_length=256).to(device)
        labels = enc.input_ids.clone()
        plen = tok(prefix, return_tensors="pt", truncation=True, max_length=256).input_ids.shape[1]
        labels[:, :plen] = -100
        losses.append(model(**enc, labels=labels).loss.float())
    return torch.stack(losses).mean()

def response_coordinate(model, tok, device, probe):
    q, good, bad = probe
    enc = tok(prompt_text(tok, q), return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    gi = tok.encode(good, add_special_tokens=False)
    bi = tok.encode(bad, add_special_tokens=False)
    if len(gi) != 1 or len(bi) != 1:
        raise RuntimeError(f"probe verbalizers must be one token: {good}/{bad}")
    return logits[gi[0]] - logits[bi[0]]

def response_vector(model, tok, device, grad=False):
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        return torch.stack([response_coordinate(model, tok, device, p) for p in PROBES]).double()

def jacobian(model, tok, ps, device):
    rows = []
    for probe in PROBES:
        r = response_coordinate(model, tok, device, probe)
        gs = torch.autograd.grad(r, ps, allow_unused=False)
        rows.append(flat([g.detach().float() for g in gs]))
    return torch.stack(rows).double()

def project(v, J, damping=1e-10):
    vd = v.double(); gram = J @ J.T
    coeff = torch.linalg.solve(gram + damping * torch.eye(len(J), device=J.device, dtype=J.dtype), J @ vd)
    return (vd - J.T @ coeff).float()

def rowspace_distance(J0, J1):
    q0 = torch.linalg.qr(J0.T, mode="reduced").Q
    q1 = torch.linalg.qr(J1.T, mode="reduced").Q
    r = min(q0.shape[1], q1.shape[1])
    overlap = torch.linalg.matrix_norm(q0.T @ q1).square()
    return float(torch.sqrt(torch.clamp(2*r - 2*overlap, min=0.0)))

def eval_loss(model, tok, device):
    return float(supervised_loss(model, tok, VALID, device).detach())

def run_arm(model, tok, ps, source, source_r, source_J, arm, budget, device):
    set_vec(ps, source); seed_all(SEED)
    m = torch.zeros_like(source); v = torch.zeros_like(source); records=[]; accepted=0
    order = [i % len(TRAIN) for i in torch.randperm(STEPS).tolist()]
    for step in range(STEPS):
        batch = [TRAIN[order[step]]]
        loss = supervised_loss(model, tok, batch, device)
        gs = torch.autograd.grad(loss, ps)
        g = flat([x.detach().float() for x in gs])
        m = .9*m + .1*g; v = .999*v + .001*g.square()
        mh = m/(1-.9**(step+1)); vh=v/(1-.999**(step+1))
        d = -mh/(vh.sqrt()+1e-8) - .01*get_vec(ps)
        J = jacobian(model,tok,ps,device) if arm=="moving_current_kernel" else source_J
        projection_residual = None
        if arm != "budgeted_lora_adamw":
            d = project(d,J)
            projection_residual = float(torch.linalg.vector_norm(J @ d.double()) / d.double().norm().clamp_min(1e-30))
        d = d / d.norm().clamp_min(1e-12) * STEP_NORM
        before=get_vec(ps); scale=1.0; ok=False
        for _ in range(MAX_BACKTRACKS+1):
            cand=before+scale*d; set_vec(ps,cand)
            drift=float(torch.linalg.vector_norm(response_vector(model,tok,device)-source_r))
            if math.isfinite(drift) and drift <= budget+1e-7:
                ok=True; accepted+=1; break
            scale*=.5
        if not ok: set_vec(ps,before); drift=float(torch.linalg.vector_norm(response_vector(model,tok,device)-source_r))
        if step in {0,STEPS-1} or (step+1)%6==0:
            print(f"[{arm} B={budget:.3e}] {step+1}/{STEPS} loss={float(loss.detach()):.5f} drift={drift:.6e}",flush=True)
        records.append({"step":step+1,"train_loss":float(loss.detach()),"drift":drift,"scale":scale if ok else 0.0,
                        "projection_residual":projection_residual})
    final_r=response_vector(model,tok,device)
    end_J = jacobian(model,tok,ps,device)
    return {"arm":arm,"budget":budget,"final_validation_loss":eval_loss(model,tok,device),
            "final_response_drift":float(torch.linalg.vector_norm(final_r-source_r)),"accepted_steps":accepted,
            "rowspace_distance_from_source":rowspace_distance(source_J,end_J),
            "max_projection_residual":max((x["projection_residual"] or 0.0) for x in records),
            "trace":records}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda"); ap.add_argument("--outdir",required=True); a=ap.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise SystemExit("CUDA required")
    device=torch.device(a.device); out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True); seed_all(SEED); t0=time.time()
    print(f"protocol={PROTOCOL} model={MODEL} device={device}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    # FP32 is deliberate: r1 showed BF16 response margins quantized all finite drift to zero.
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32).to(device)
    layers=len(model.model.layers)
    cfg=LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",
                   target_modules=["q_proj","v_proj"],layers_to_transform=[layers-2,layers-1])
    model=get_peft_model(model,cfg); model.train(); ps=trainable(model)
    source=get_vec(ps); source_r=response_vector(model,tok,device); source_J=jacobian(model,tok,ps,device)
    sv=torch.linalg.svdvals(source_J); rank=int((sv > RANK_TOL*sv.max().clamp_min(1.0)).sum())
    # Calibrate budgets from one unprojected normalized gradient step.
    loss=supervised_loss(model,tok,[TRAIN[0]],device); g=flat([x.detach().float() for x in torch.autograd.grad(loss,ps)])
    proposal=-g/g.norm().clamp_min(1e-12)*STEP_NORM;set_vec(ps,source+proposal)
    one=float(torch.linalg.vector_norm(response_vector(model,tok,device)-source_r));set_vec(ps,source)
    budgets=[max(one*.75,1e-4),max(one*1.5,2e-4)]
    print(f"[preflight] dtype=float32 trainable={source.numel()} response_dim={len(PROBES)} rank={rank} one_step={one:.9e} budgets={budgets}",flush=True)
    runs=[]
    for B in budgets:
        for arm in ["moving_current_kernel","source_frozen_kernel","budgeted_lora_adamw"]:
            runs.append(run_arm(model,tok,ps,source,source_r,source_J,arm,B,device))
    pairs=[]
    for B in budgets:
        q={r["arm"]:r for r in runs if r["budget"]==B}
        pairs.append({"budget":B,
          "source_minus_moving_validation_loss":q["source_frozen_kernel"]["final_validation_loss"]-q["moving_current_kernel"]["final_validation_loss"],
          "adamw_minus_moving_validation_loss":q["budgeted_lora_adamw"]["final_validation_loss"]-q["moving_current_kernel"]["final_validation_loss"]})
    nonzero_drifts=[x["drift"] for r in runs for x in r["trace"] if x["drift"]>PRECISION_FLOOR]
    low=[r for r in runs if r["budget"]==budgets[0]]; high=[r for r in runs if r["budget"]==budgets[1]]
    distinguishable=(max(abs(a["final_response_drift"]-b["final_response_drift"]) for a,b in zip(low,high))>PRECISION_FLOOR
                     or any(a["trace"]!=b["trace"] for a,b in zip(low,high)))
    current_moves=[r["rowspace_distance_from_source"] for r in runs if r["arm"]=="moving_current_kernel"]
    projected=[r["max_projection_residual"] for r in runs if r["arm"]!="budgeted_lora_adamw"]
    gates={"all_finite":all(math.isfinite(r["final_validation_loss"]) for r in runs),
           "all_budgets_respected":all(r["final_response_drift"]<=r["budget"]+1e-6 for r in runs),
           "jacobian_nonzero":rank>0,"one_step_drift_above_precision_floor":one>PRECISION_FLOOR,
           "measured_drifts_not_all_zero":len(nonzero_drifts)>0,"two_budgets_empirically_distinguishable":distinguishable,
           "current_rowspace_moves":max(current_moves)>PRECISION_FLOOR,
           "projected_direction_residual_at_most_1e-5":max(projected)<=1e-5,
           "moving_kernel_exercised":True}
    summary={"protocol":PROTOCOL,"mode":"one_seed_general_compute_core_precision_audit","model":MODEL,"seed":SEED,
      "design":{"learning_target_L":"12 authored general instruction pairs","response_map_R":"6 disjoint frozen A/B logit-margin probes","dynamic_current_kernel":True,"three_arms":True,"global_response_backtracking":True,"model_dtype":"float32","response_aggregation":"float64"},
      "trainable_lora_parameters":source.numel(),"response_rank":rank,"budgets":budgets,"runs":runs,"pairs":pairs,
      "precision_audit":{"precision_floor":PRECISION_FLOOR,"calibration_one_step_drift":one,
                          "nonzero_drift_observation_count":len(nonzero_drifts),
                          "current_rowspace_distances":current_moves,"projected_max_residuals":projected},
      "gates":gates,
      "wall_seconds":time.time()-t0,
      "scientific_status":"R23C_R2_PRECISION_AUDIT_COMPLETE" if all(gates.values()) else "R23C_R2_PRECISION_AUDIT_FAIL_CLOSED",
      "claim_boundary":"One seed, Qwen3-0.6B, tiny authored data and a restricted LoRA chart. This audits numerical execution of DR, moving/source row spaces, projection and finite response budgets only; it does not establish general capability, personalization, continual learning, safety, or moving-fibre superiority."}
    (out/"run_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="runs"},ensure_ascii=False,indent=2),flush=True)
    return 0 if all(gates.values()) else 2

if __name__=="__main__": raise SystemExit(main())
