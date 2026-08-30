#!/usr/bin/env python3
"""Prospective eight-new-seed CPU confirmation of GPT-2/LoRA Pareto signal."""
from __future__ import annotations
import argparse,csv,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
import gptw_lora_low_response_pareto_cpu as v

PROTOCOL="GPTW_GPT2_NATIVE_LORA_B_LOW_RESPONSE_PARETO_CPU_V1_R1_8SEED"
SEEDS=list(range(86841,86849));REQUIRED=6;HARD_LIMIT=5400

def run_seed(seed,model,tok,data,started):
    v.seed_all(seed);anchors,response_ids,train,train_y,cal,cal_y,eva,eva_y=data
    adapter=v.NativeLoraB(model,seed);theta=torch.nn.Parameter(torch.zeros(adapter.dimension,dtype=torch.float32))
    try:
        pre_eval=v.loss_float(model,adapter,theta,eva,eva_y);row0,rank0,sigma0=v.jacobian(model,adapter,theta,anchors,response_ids)
        opt=torch.optim.AdamW([theta],lr=2e-2,weight_decay=1e-4);snaps=[]
        for step in range(v.ADAPT_STEPS):
            if time.time()-started>HARD_LIMIT:raise TimeoutError("90-minute cohort hard limit exceeded")
            before=theta.detach().clone();loss=v.task_loss(model,adapter,theta,train,train_y);opt.zero_grad();loss.backward();opt.step();delta=theta.detach()-before
            if step%v.AUDIT_EVERY==0:
                row,rank,sigma=v.jacobian(model,adapter,before,anchors,response_ids)
                snaps.append({"step":step,"theta":before,"delta":delta.detach().clone(),"row":row,"rank":rank,"rotation":v.rotation(row,row0),"train_loss":float(loss)})
        post_eval=v.loss_float(model,adapter,theta,eva,eva_y);nodes=[];n=len(snaps)
        for i,s in enumerate(snaps):
            th,d,row=s["theta"],s["delta"],s["row"];norm=d.norm();far=snaps[(i+n//2)%n]["row"];perm=v.permuted_row(row,seed*1000+s["step"])
            dirs={"actual":d,"current_kernel":v.normalized(v.project_kernel(row,d),norm),"source_kernel":v.normalized(v.project_kernel(row0,d),norm),
                  "far_time_kernel":v.normalized(v.project_kernel(far,d),norm),"permuted_kernel":v.normalized(v.project_kernel(perm,d),norm)}
            curves={k:v.curve(model,adapter,th,q,anchors,response_ids,cal,cal_y,eva,eva_y) for k,q in dirs.items()}
            actual_full=next(x for x in curves["actual"] if x["alpha"]==1.);budgets=[f*actual_full["response_cost"] for f in v.BUDGETS]
            selected={k:[v.choose(curves[k],b) for b in budgets] for k in dirs};aucs={k:v.auc([x["heldout_utility"] for x in selected[k]]) for k in dirs}
            node={"step":s["step"],"row_space_rotation":s["rotation"],"response_rank":s["rank"],"update_norm":float(norm),"actual_full_response_cost":actual_full["response_cost"]}
            for arm in dirs:node[f"{arm}_heldout_auc"]=aucs[arm];node[f"{arm}_selected_alphas"]=[x["alpha"] for x in selected[arm]]
            for control in ("actual","source_kernel","far_time_kernel","permuted_kernel"):node[f"current_minus_{control}_auc"]=aucs["current_kernel"]-aucs[control]
            nodes.append(node)
        primary=[x for x in nodes if x["step"]>0]
        summary={"pre_heldout_loss":pre_eval,"post_heldout_loss":post_eval,"heldout_loss_gain":pre_eval-post_eval,"lora_b_dimension":adapter.dimension,
                 "source_response_rank":rank0,"audited_nodes_total":len(nodes),"primary_noninitial_nodes":len(primary),"maximum_row_space_rotation":max(x["row_space_rotation"] for x in nodes),
                 "fraction_current_positive_auc":float(np.mean([x["current_kernel_heldout_auc"]>0 for x in primary]))}
        gates={"real_adaptation_improves_heldout":pre_eval-post_eval>0,"nontrivial_rotation":summary["maximum_row_space_rotation"]>=.001,
               "current_positive_auc":sum(x["current_kernel_heldout_auc"]>0 for x in primary)>=v.MIN_NODE_WINS}
        for control in ("actual","source_kernel","far_time_kernel","permuted_kernel"):
            dif=np.asarray([x[f"current_minus_{control}_auc"] for x in primary]);summary[f"current_beats_{control}_nodes"]=int((dif>0).sum());summary[f"median_current_minus_{control}_auc"]=float(np.median(dif));gates[f"current_beats_{control}"]=int((dif>0).sum())>=v.MIN_NODE_WINS and float(np.median(dif))>0
        return {"seed":seed,"summary":summary,"gates":gates,"supported":all(gates.values()),"nodes":nodes}
    finally:
        adapter.active=None
        for handle in adapter.handles:handle.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="gptw_lora_pareto_cpu_8seed_results");args,unknown=p.parse_known_args()
    if unknown:print("[notice] ignored notebook arguments:",unknown,flush=True)
    torch.set_num_threads(max(1,min(8,torch.get_num_threads())));started=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    print(f"[preflight] {PROTOCOL} model={v.MODEL_ID} device=cpu seeds={SEEDS}",flush=True)
    tok=AutoTokenizer.from_pretrained(v.MODEL_ID);tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(v.MODEL_ID);model.eval();model.config.use_cache=False
    for q in model.parameters():q.requires_grad_(False)
    data=(v.encode_prompts(tok,v.ANCHORS),v.target_ids(tok,v.RESPONSE_WORDS),*v.build_task(tok,v.TRAIN_TEMPLATES),*v.build_task(tok,v.CAL_TEMPLATES),*v.build_task(tok,v.EVAL_TEMPLATES))
    records=[]
    for i,seed in enumerate(SEEDS,1):
        print(f"[seed {i}/8] {seed}",flush=True);rec=run_seed(seed,model,tok,data,started);records.append(rec)
        (out/f"seed_{seed}.json").write_text(json.dumps(rec,indent=2));print(json.dumps(rec["summary"],indent=2),flush=True)
    supporting=sum(x["supported"] for x in records);decision="GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED" if supporting>=REQUIRED else "GPT2_LORA_LOW_RESPONSE_PARETO_NOT_CONFIRMED"
    report={"protocol":PROTOCOL,"prospective":True,"decision":decision,"supporting_seeds":supporting,"required":REQUIRED,"attempted":len(records),"elapsed_seconds":time.time()-started,"records":records,
            "claim_boundary":"Eight-new-seed CPU confirmation in standard GPT-2 with a frozen backbone and rank-2 native LoRA-B on layers 10 and 11; not full-parameter tuning, broad semantic knowledge acquisition, arbitrary language tasks, or a universal intelligence law."}
    protocol={"protocol":PROTOCOL,"prospective":True,"development_seed_excluded":86741,"seeds":SEEDS,"required":REQUIRED,"parent_protocol":v.PROTOCOL,"all_parent_prompts_budgets_alphas_layers_and_gates_unchanged":True,"model_loaded_once":True,"lora_hooks_removed_between_seeds":True,"hard_limit_seconds":HARD_LIMIT}
    (out/"report.json").write_text(json.dumps(report,indent=2));(out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    rows=[(r["seed"],n) for r in records for n in r["nodes"]];scalar=[k for k,val in rows[0][1].items() if not isinstance(val,list)]
    with (out/"node_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["seed"]+scalar);w.writeheader()
        for seed,node in rows:w.writerow({"seed":seed,**{k:node[k] for k in scalar}})
    archive=shutil.make_archive(str(out),"zip",out.parent,out.name);print("="*88);print(json.dumps({"decision":decision,"supporting_seeds":supporting,"required":REQUIRED},indent=2),flush=True);print("RESULT ZIP:",archive,flush=True)

if __name__=="__main__":main()
