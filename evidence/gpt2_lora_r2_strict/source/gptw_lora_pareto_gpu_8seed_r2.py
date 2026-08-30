#!/usr/bin/env python3
"""GPTW Pareto GPU R2: strict current-kernel random-direction audit."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import gptw_lora_low_response_pareto_gpu as v

PROTOCOL="GPTW_GPT2_NATIVE_LORA_B_LOW_RESPONSE_PARETO_GPU_V1_R2_STRICT"
SEEDS=list(range(91841,91849));REQUIRED=6;RANDOM_CURRENT_DIRECTIONS=16;HARD_LIMIT=7200

def selected_curve(points,budgets):return [v.choose(points,b) for b in budgets]

def random_current_best(model,adapter,theta,row,norm,anchors,response_ids,cal,cal_y,eva,eva_y,budgets,seed,step,started):
    gen=torch.Generator(device=theta.device).manual_seed(seed*100000+step*100+73);candidates=[]
    for index in range(RANDOM_CURRENT_DIRECTIONS):
        if time.time()-started>HARD_LIMIT:raise TimeoutError("R2 frozen two-hour hard limit exceeded")
        raw=torch.randn(theta.numel(),generator=gen,device=theta.device,dtype=theta.dtype)
        direction=v.normalized(v.project_kernel(row,raw),norm)
        points=v.curve(model,adapter,theta,direction,anchors,response_ids,cal,cal_y,eva,eva_y)
        chosen=selected_curve(points,budgets)
        candidates.append({"index":index,"calibration_auc":v.auc([x["calibration_utility"] for x in chosen]),
                           "heldout_auc":v.auc([x["heldout_utility"] for x in chosen]),
                           "selected_alphas":[x["alpha"] for x in chosen]})
    return max(candidates,key=lambda x:(x["calibration_auc"],-x["index"]))

def far_snapshot(snaps,index):
    current=snaps[index];eligible=[s for j,s in enumerate(snaps) if j!=index and s["step"]!=0]
    if not eligible:eligible=[s for j,s in enumerate(snaps) if j!=index]
    return max(eligible,key=lambda s:(abs(s["step"]-current["step"]),s["step"]))

def run_seed(seed,model,data,started):
    v.seed_all(seed);anchors,response_ids,train,train_y,cal,cal_y,eva,eva_y=data
    adapter=v.NativeLoraB(model,seed);theta=torch.nn.Parameter(torch.zeros(adapter.dimension,dtype=torch.float32,device="cuda"))
    try:
        pre_eval=v.loss_float(model,adapter,theta,eva,eva_y);row0,rank0,_=v.jacobian(model,adapter,theta,anchors,response_ids)
        opt=torch.optim.AdamW([theta],lr=2e-2,weight_decay=1e-4);snaps=[]
        for step in range(v.ADAPT_STEPS):
            if time.time()-started>HARD_LIMIT:raise TimeoutError("R2 frozen two-hour hard limit exceeded")
            before=theta.detach().clone();objective=v.task_loss(model,adapter,theta,train,train_y)
            opt.zero_grad();objective.backward();opt.step();delta=theta.detach()-before
            if step%v.AUDIT_EVERY==0:
                row,rank,_=v.jacobian(model,adapter,before,anchors,response_ids)
                snaps.append({"step":step,"theta":before,"delta":delta.detach().clone(),"row":row,"rank":rank,
                              "rotation":v.rotation(row,row0),"train_loss":float(objective.detach())})
        post_eval=v.loss_float(model,adapter,theta,eva,eva_y);nodes=[]
        for i,s in enumerate(snaps):
            if time.time()-started>HARD_LIMIT:raise TimeoutError("R2 frozen two-hour hard limit exceeded")
            th,d,row=s["theta"],s["delta"],s["row"];norm=d.norm();far=far_snapshot(snaps,i);perm=v.permuted_row(row,seed*1000+s["step"])
            dirs={"actual":d,"current_kernel":v.normalized(v.project_kernel(row,d),norm),
                  "source_kernel":v.normalized(v.project_kernel(row0,d),norm),
                  "far_time_kernel":v.normalized(v.project_kernel(far["row"],d),norm),
                  "permuted_kernel":v.normalized(v.project_kernel(perm,d),norm)}
            curves={k:v.curve(model,adapter,th,q,anchors,response_ids,cal,cal_y,eva,eva_y) for k,q in dirs.items()}
            actual_full=next(x for x in curves["actual"] if x["alpha"]==1.0);budgets=[f*actual_full["response_cost"] for f in v.BUDGETS]
            selected={k:selected_curve(curves[k],budgets) for k in dirs};aucs={k:v.auc([x["heldout_utility"] for x in selected[k]]) for k in dirs}
            random_best=random_current_best(model,adapter,th,row,norm,anchors,response_ids,cal,cal_y,eva,eva_y,budgets,seed,s["step"],started)
            node={"step":s["step"],"far_step":far["step"],"row_space_rotation":s["rotation"],"response_rank":s["rank"],
                  "update_norm":float(norm),"actual_full_response_cost":actual_full["response_cost"]}
            for arm in dirs:
                node[f"{arm}_heldout_auc"]=aucs[arm];node[f"{arm}_selected_alphas"]=[x["alpha"] for x in selected[arm]]
            node["best_of_16_random_current_kernel_calibration_auc"]=random_best["calibration_auc"]
            node["best_of_16_random_current_kernel_heldout_auc"]=random_best["heldout_auc"]
            node["best_of_16_random_current_kernel_index"]=random_best["index"]
            node["best_of_16_random_current_kernel_selected_alphas"]=random_best["selected_alphas"]
            for control in ("actual","source_kernel","far_time_kernel","permuted_kernel"):
                node[f"current_minus_{control}_auc"]=aucs["current_kernel"]-aucs[control]
            node["current_minus_best_of_16_random_current_kernel_auc"]=aucs["current_kernel"]-random_best["heldout_auc"]
            nodes.append(node)
        primary=[x for x in nodes if x["step"]>0]
        summary={"pre_heldout_loss":pre_eval,"post_heldout_loss":post_eval,"heldout_loss_gain":pre_eval-post_eval,
                 "source_response_rank":rank0,"audited_nodes_total":len(nodes),"primary_noninitial_nodes":len(primary),
                 "maximum_row_space_rotation":max(x["row_space_rotation"] for x in nodes)}
        gates={"real_adaptation_improves_heldout":pre_eval-post_eval>0,"nontrivial_rotation":summary["maximum_row_space_rotation"]>=.001,
               "current_positive_auc":sum(x["current_kernel_heldout_auc"]>0 for x in primary)>=v.MIN_NODE_WINS}
        for control in ("actual","source_kernel","far_time_kernel","permuted_kernel","best_of_16_random_current_kernel"):
            dif=np.asarray([x[f"current_minus_{control}_auc"] for x in primary]);summary[f"current_beats_{control}_nodes"]=int((dif>0).sum())
            summary[f"median_current_minus_{control}_auc"]=float(np.median(dif))
            gates[f"current_beats_{control}"]=int((dif>0).sum())>=v.MIN_NODE_WINS and float(np.median(dif))>0
        return {"seed":seed,"summary":summary,"gates":gates,"supported":all(gates.values()),"nodes":nodes}
    finally:
        adapter.active=None
        for handle in adapter.handles:handle.remove()

def bootstrap_ci(values,draws=10000):
    x=np.asarray(values,float);rng=np.random.default_rng(20260830)
    means=np.asarray([rng.choice(x,len(x),replace=True).mean() for _ in range(draws)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def sign_flip_p(values):
    x=np.asarray(values,float);obs=abs(float(x.mean()));count=0
    for mask in range(1<<len(x)):
        signs=np.asarray([1.0 if mask&(1<<i) else -1.0 for i in range(len(x))])
        count+=abs(float((x*signs).mean()))>=obs-1e-15
    return count/float(1<<len(x))

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="gptw_lora_pareto_gpu_8seed_r2_results");args,unknown=p.parse_known_args()
    if unknown:print("[notice] ignored notebook arguments:",unknown,flush=True)
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required. Select Runtime > Change runtime type > GPU.")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    started=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);gpu=torch.cuda.get_device_name(0)
    frozen=json.loads(Path(__file__).with_name("protocol.json").read_text());print(f"[preflight] {PROTOCOL} gpu={gpu} seeds={SEEDS}",flush=True)
    tok=AutoTokenizer.from_pretrained(v.MODEL_ID);tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(v.MODEL_ID,torch_dtype=torch.float32).to("cuda");model.eval();model.config.use_cache=False
    for q in model.parameters():q.requires_grad_(False)
    def bc(batch):return {k:q.to("cuda") for k,q in batch.items()}
    def tc(task):return bc(task[0]),task[1].to("cuda")
    data=(bc(v.encode_prompts(tok,v.ANCHORS)),v.target_ids(tok,v.RESPONSE_WORDS).to("cuda"),
          *tc(v.build_task(tok,v.TRAIN_TEMPLATES)),*tc(v.build_task(tok,v.CAL_TEMPLATES)),*tc(v.build_task(tok,v.EVAL_TEMPLATES)))
    records=[];timeout_seed=None
    for i,seed in enumerate(SEEDS,1):
        print(f"[seed {i}/8] {seed}",flush=True)
        try:record=run_seed(seed,model,data,started)
        except TimeoutError:
            timeout_seed=seed;print(f"[timeout] seed={seed}; writing recoverable partial archive",flush=True);break
        records.append(record);(out/f"seed_{seed}.json").write_text(json.dumps(record,indent=2));print(json.dumps(record["summary"],indent=2),flush=True)
    complete=len(records)==len(SEEDS);contrasts=[r["summary"]["median_current_minus_best_of_16_random_current_kernel_auc"] for r in records]
    ci=bootstrap_ci(contrasts) if len(contrasts)>1 else ([contrasts[0]]*2 if contrasts else [None,None]);supporting=sum(r["supported"] for r in records)
    cohort_gates={"complete_8_seeds":complete,"at_least_6_supporting_seeds":supporting>=REQUIRED,
                  "random_current_seed_contrast_at_least_6_positive":sum(x>0 for x in contrasts)>=REQUIRED,
                  "random_current_bootstrap_ci_lower_positive":complete and ci[0]>0}
    decision=("GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_CONFIRMED" if all(cohort_gates.values()) else
              ("INCOMPLETE_TIMEOUT" if not complete else "GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_NOT_CONFIRMED"))
    report={"protocol":PROTOCOL,"prospective":True,"decision":decision,"completed_seeds":len(records),"attempted_timeout_seed":timeout_seed,
            "supporting_seeds":supporting,"required":REQUIRED,"primary_random_current_control":{"seed_level_contrasts":contrasts,
            "positive_seeds":sum(x>0 for x in contrasts),"bootstrap_mean_95ci":ci,
            "exact_two_sided_sign_flip_p":sign_flip_p(contrasts) if contrasts else None},"cohort_gates":cohort_gates,
            "elapsed_seconds":time.time()-started,"gpu":gpu,"records":records,"claim_boundary":frozen["claim_boundary"]}
    (out/"report.json").write_text(json.dumps(report,indent=2));(out/"protocol.json").write_text(json.dumps(frozen,indent=2))
    rows=[(r["seed"],n) for r in records for n in r["nodes"]]
    if rows:
        scalar=[k for k,val in rows[0][1].items() if not isinstance(val,list)]
        with (out/"node_metrics.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["seed"]+scalar);w.writeheader()
            for seed,node in rows:w.writerow({"seed":seed,**{k:node[k] for k in scalar}})
    archive=shutil.make_archive(str(out),"zip",out.parent,out.name);print("="*88)
    print(json.dumps({"decision":decision,"cohort_gates":cohort_gates},indent=2));print("RESULT ZIP:",archive)
    print("SHA-256:",hashlib.sha256(Path(archive).read_bytes()).hexdigest())

if __name__=="__main__":main()
