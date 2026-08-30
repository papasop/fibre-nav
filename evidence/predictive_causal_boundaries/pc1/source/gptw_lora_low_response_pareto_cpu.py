#!/usr/bin/env python3
"""CPU prospective preflight: GPT-2 native LoRA-B low-response Pareto audit."""
from __future__ import annotations
import argparse,csv,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

PROTOCOL="GPTW_GPT2_NATIVE_LORA_B_LOW_RESPONSE_PARETO_CPU_V1"
MODEL_ID="openai-community/gpt2"; SEED=86741; LAYERS=[10,11]; RANK=2; SCALE=2.0
ADAPT_STEPS=24; AUDIT_EVERY=6; BUDGETS=[.02,.05,.10,.20]
ALPHAS=[0.,.015625,.03125,.0625,.125,.25,.5,.75,1.]; HARD_LIMIT=5400
MIN_NODE_WINS=2; EPS=1e-12

ANCHORS=["The history of science shows that","A careful mathematical argument must","In ordinary language, a promise means","The safest engineering decision is"]
RESPONSE_WORDS=[" the"," a"," to"," and"]
MAPPING={"dax":" blue","wug":" green","kiv":" red","zorp":" yellow"}
TRAIN_TEMPLATES=["In the private glossary, {code} denotes the color","For this experiment, the color assigned to {code} is","The laboratory code {code} corresponds to","According to the temporary key, {code} means"]
CAL_TEMPLATES=["Under the declared mapping, {code} identifies","Within this artificial vocabulary, {code} stands for"]
EVAL_TEMPLATES=["The stipulated color of {code} is","Using the unseen wording, {code} should be followed by"]

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

class NativeLoraB:
    def __init__(self,model,seed):
        self.model=model;self.active=None;self.spec=[];self.handles=[]
        gen=torch.Generator().manual_seed(seed+991)
        for layer in LAYERS:
            module=model.transformer.h[layer].attn.c_attn; in_dim=module.weight.shape[0]; out_dim=module.weight.shape[1]
            A=torch.randn(in_dim,RANK,generator=gen,dtype=torch.float32)/math.sqrt(in_dim)
            start=sum(s[3] for s in self.spec); size=RANK*out_dim
            self.spec.append((module,A,start,size,out_dim))
        for module,A,start,size,out_dim in self.spec:
            def hook(mod,inputs,output,A=A,start=start,size=size,out_dim=out_dim):
                if self.active is None:return output
                x=inputs[0];B=self.active[start:start+size].reshape(RANK,out_dim).to(x.dtype)
                return output+(SCALE/RANK)*(x@A.to(x.device,x.dtype)@B)
            self.handles.append(module.register_forward_hook(hook))
        self.dimension=sum(s[3] for s in self.spec)
    def use(self,theta):self.active=theta

def encode_prompts(tok,prompts):
    return tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=48)

def target_ids(tok,words):
    ids=[]
    for w in words:
        q=tok.encode(w,add_special_tokens=False)
        if len(q)!=1:raise RuntimeError(f"Target must be one GPT-2 token: {w!r} -> {q}")
        ids.append(q[0])
    return torch.tensor(ids,dtype=torch.long)

def logits_last(model,adapter,theta,batch):
    adapter.use(theta)
    return model(**batch,use_cache=False).logits[:,-1,:]

def task_loss(model,adapter,theta,batch,targets):
    return F.cross_entropy(logits_last(model,adapter,theta,batch),targets)

def response(model,adapter,theta,anchor_batch,response_ids):
    z=logits_last(model,adapter,theta,anchor_batch)[:,response_ids]
    z=z-z.mean(-1,keepdim=True)
    return z[:,:3].reshape(-1)

def jacobian(model,adapter,theta,anchors,response_ids):
    q=theta.detach().requires_grad_(True)
    j=torch.autograd.functional.jacobian(lambda z:response(model,adapter,z,anchors,response_ids),q,vectorize=True).detach().double()
    u,s,vh=torch.linalg.svd(j,full_matrices=False);tol=max(j.shape)*torch.finfo(torch.float64).eps*s[0].clamp_min(1e-30)
    rank=int((s>tol).sum());return vh[:rank],rank,float(s[0])

def project_kernel(row,d):
    x=d.double();return (x-row.T@(row@x)).float()

def normalized(d,norm):return d*(norm/(d.norm()+1e-30))

def permuted_row(row,seed):
    g=torch.Generator().manual_seed(seed);p=torch.randperm(row.shape[1],generator=g);sign=torch.where(torch.rand(row.shape[1],generator=g)<.5,-1.,1.).double()
    return row[:,p]*sign

def rotation(a,b):
    k=min(a.shape[0],b.shape[0]);s=torch.linalg.svdvals(a[:k]@b[:k].T).clamp(0,1);return float(torch.acos(s.min()))

def loss_float(model,adapter,theta,batch,targets):
    with torch.no_grad():return float(task_loss(model,adapter,theta,batch,targets))

def response_cost(model,adapter,t0,t1,anchors,response_ids):
    with torch.no_grad():
        d=response(model,adapter,t1,anchors,response_ids)-response(model,adapter,t0,anchors,response_ids)
        return float(d.norm()/math.sqrt(d.numel()))

def curve(model,adapter,theta,direction,anchors,response_ids,cal,cal_y,eva,eva_y):
    bc=loss_float(model,adapter,theta,cal,cal_y);be=loss_float(model,adapter,theta,eva,eva_y);out=[]
    for alpha in ALPHAS:
        q=theta+alpha*direction
        out.append({"alpha":alpha,"response_cost":response_cost(model,adapter,theta,q,anchors,response_ids),
                    "calibration_utility":bc-loss_float(model,adapter,q,cal,cal_y),
                    "heldout_utility":be-loss_float(model,adapter,q,eva,eva_y)})
    return out

def choose(points,budget):
    feasible=[p for p in points if p["response_cost"]<=budget+EPS]
    return max(feasible,key=lambda p:(p["calibration_utility"],-p["alpha"]))

def auc(values):return float(np.trapz(values,BUDGETS)/(BUDGETS[-1]-BUDGETS[0]))

def build_task(tok,templates):
    prompts=[];words=[]
    for code,target in MAPPING.items():
        for template in templates:prompts.append(template.format(code=code));words.append(target)
    return encode_prompts(tok,prompts),target_ids(tok,words)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="gptw_lora_pareto_cpu_results");args,unknown=p.parse_known_args()
    if unknown:print("[notice] ignored notebook arguments:",unknown,flush=True)
    seed_all(SEED);torch.set_num_threads(max(1,min(8,torch.get_num_threads())));started=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    print(f"[preflight] {PROTOCOL} model={MODEL_ID} device=cpu seed={SEED}",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL_ID);tok.pad_token=tok.eos_token;tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID);model.eval();model.config.use_cache=False
    for q in model.parameters():q.requires_grad_(False)
    adapter=NativeLoraB(model,SEED);theta=torch.nn.Parameter(torch.zeros(adapter.dimension,dtype=torch.float32))
    anchors=encode_prompts(tok,ANCHORS);response_ids=target_ids(tok,RESPONSE_WORDS)
    train,train_y=build_task(tok,TRAIN_TEMPLATES);cal,cal_y=build_task(tok,CAL_TEMPLATES);eva,eva_y=build_task(tok,EVAL_TEMPLATES)
    pre_eval=loss_float(model,adapter,theta,eva,eva_y);row0,rank0,sigma0=jacobian(model,adapter,theta,anchors,response_ids)
    opt=torch.optim.AdamW([theta],lr=2e-2,weight_decay=1e-4);snaps=[]
    for step in range(ADAPT_STEPS):
        if time.time()-started>HARD_LIMIT:raise TimeoutError("90-minute CPU hard limit exceeded")
        before=theta.detach().clone();loss=task_loss(model,adapter,theta,train,train_y);opt.zero_grad();loss.backward();opt.step();delta=theta.detach()-before
        if step%AUDIT_EVERY==0:
            row,rank,sigma=jacobian(model,adapter,before,anchors,response_ids)
            snaps.append({"step":step,"theta":before,"delta":delta.detach().clone(),"row":row,"rank":rank,"rotation":rotation(row,row0),"train_loss":float(loss)})
            print(f"[node] step={step} loss={float(loss):.6f} rank={rank}",flush=True)
    post_eval=loss_float(model,adapter,theta,eva,eva_y);nodes=[];n=len(snaps)
    for i,s in enumerate(snaps):
        th,d,row=s["theta"],s["delta"],s["row"];norm=d.norm();far=snaps[(i+n//2)%n]["row"];perm=permuted_row(row,SEED*1000+s["step"])
        dirs={"actual":d,"current_kernel":normalized(project_kernel(row,d),norm),"source_kernel":normalized(project_kernel(row0,d),norm),
              "far_time_kernel":normalized(project_kernel(far,d),norm),"permuted_kernel":normalized(project_kernel(perm,d),norm)}
        curves={k:curve(model,adapter,th,v,anchors,response_ids,cal,cal_y,eva,eva_y) for k,v in dirs.items()}
        actual_full=next(x for x in curves["actual"] if x["alpha"]==1.);budgets=[f*actual_full["response_cost"] for f in BUDGETS]
        selected={k:[choose(curves[k],b) for b in budgets] for k in dirs};aucs={k:auc([x["heldout_utility"] for x in selected[k]]) for k in dirs}
        node={"step":s["step"],"row_space_rotation":s["rotation"],"response_rank":s["rank"],"update_norm":float(norm),"actual_full_response_cost":actual_full["response_cost"]}
        for arm in dirs:
            node[f"{arm}_heldout_auc"]=aucs[arm];node[f"{arm}_selected_alphas"]=[x["alpha"] for x in selected[arm]]
        for control in ("actual","source_kernel","far_time_kernel","permuted_kernel"):node[f"current_minus_{control}_auc"]=aucs["current_kernel"]-aucs[control]
        nodes.append(node)
    primary=[x for x in nodes if x["step"]>0];summary={"pre_heldout_loss":pre_eval,"post_heldout_loss":post_eval,"heldout_loss_gain":pre_eval-post_eval,
        "lora_b_dimension":adapter.dimension,"source_response_rank":rank0,"audited_nodes_total":len(nodes),"primary_noninitial_nodes":len(primary),"maximum_row_space_rotation":max(x["row_space_rotation"] for x in nodes),
        "fraction_current_positive_auc":float(np.mean([x["current_kernel_heldout_auc"]>0 for x in primary]))}
    gates={"real_adaptation_improves_heldout":pre_eval-post_eval>0,"nontrivial_rotation":summary["maximum_row_space_rotation"]>=.001,"current_positive_auc":sum(x["current_kernel_heldout_auc"]>0 for x in primary)>=MIN_NODE_WINS}
    for control in ("actual","source_kernel","far_time_kernel","permuted_kernel"):
        dif=np.asarray([x[f"current_minus_{control}_auc"] for x in primary]);summary[f"current_beats_{control}_nodes"]=int((dif>0).sum());summary[f"median_current_minus_{control}_auc"]=float(np.median(dif));gates[f"current_beats_{control}"]=int((dif>0).sum())>=MIN_NODE_WINS and float(np.median(dif))>0
    supported=all(gates.values());decision="CPU_PREFLIGHT_GPT2_LORA_PARETO_SIGNAL" if supported else "CPU_PREFLIGHT_GPT2_LORA_PARETO_NOT_SUPPORTED"
    report={"protocol":PROTOCOL,"prospective":True,"mode":"CPU_SINGLE_SEED_PREFLIGHT","decision":decision,"seed":SEED,"supported":supported,"elapsed_seconds":time.time()-started,"summary":summary,"gates":gates,"nodes":nodes,
            "claim_boundary":"Single-seed CPU preflight in the standard GPT-2 frozen-backbone rank-2 native LoRA-B domain on layers 10 and 11; not multi-seed confirmation, full-parameter tuning, semantic knowledge acquisition, or a universal intelligence law."}
    protocol={"protocol":PROTOCOL,"model":MODEL_ID,"seed":SEED,"layers":LAYERS,"lora_rank":RANK,"trainable_domain":"LoRA-B only; LoRA-A frozen random","anchors":ANCHORS,"response_words":RESPONSE_WORDS,"mapping":MAPPING,"budget_fractions":BUDGETS,"alpha_grid":ALPHAS,"calibration_and_heldout_prompts_disjoint":True,"heldout_never_used_for_alpha_selection":True,"required_node_wins":MIN_NODE_WINS,"hard_limit_seconds":HARD_LIMIT}
    (out/"report.json").write_text(json.dumps(report,indent=2));(out/"protocol.json").write_text(json.dumps(protocol,indent=2));(out/f"seed_{SEED}.json").write_text(json.dumps(report,indent=2))
    scalar=[k for k,v in nodes[0].items() if not isinstance(v,list)]
    with (out/"node_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=scalar);w.writeheader();w.writerows([{k:n[k] for k in scalar} for n in nodes])
    archive=shutil.make_archive(str(out),"zip",out.parent,out.name);print("="*88);print(json.dumps({"decision":decision,"summary":summary,"gates":gates},indent=2),flush=True);print("RESULT ZIP:",archive,flush=True)

if __name__=="__main__":main()
