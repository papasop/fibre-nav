#!/usr/bin/env python3
"""MFI v1.9.6.0: prospectively frozen fresh-split semantic router."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


COHORT = {
    "city": [
        ("paris", "home to the Eiffel Tower beside the Seine"),
        ("tokyo", "a dense Japanese metropolis around Tokyo Bay"),
        ("cairo", "a major settlement on the Nile near the pyramids"),
        ("lima", "a Pacific coastal metropolis governed from Peru"),
        ("oslo", "a Norwegian urban centre at the end of a fjord"),
        ("nairobi", "an East African metropolitan centre in Kenya"),
        ("london", "an urban centre on the Thames with Westminster"),
        ("berlin", "a German metropolis marked by the Brandenburg Gate"),
        ("madrid", "a large Iberian urban centre around the Manzanares"),
        ("rome", "an Italian metropolis containing the Colosseum"),
        ("beijing", "a Chinese metropolitan seat near the Forbidden City"),
        ("seoul", "a Korean metropolis divided by the Han River"),
        ("delhi", "a vast Indian metropolitan area surrounding New Delhi"),
        ("bangkok", "a Thai metropolis crossed by the Chao Phraya"),
        ("ottawa", "a Canadian urban seat on the Ottawa River"),
        ("canberra", "an inland Australian administrative settlement"),
        ("brasilia", "a planned Brazilian metropolis with modernist buildings"),
        ("buenos_aires", "a large Argentine metropolis on the Rio de la Plata"),
        ("lagos", "a populous Nigerian coastal metropolis and commercial centre"),
        ("helsinki", "a Finnish urban centre on the Baltic coast"),
    ],
    "country": [
        ("france", "a sovereign European polity governed from Paris"),
        ("japan", "an island polity in East Asia governed from Tokyo"),
        ("egypt", "a sovereign Nile polity linking northeast Africa and Sinai"),
        ("peru", "an Andean sovereign polity on the Pacific coast"),
        ("norway", "a Nordic sovereign polity with a long Atlantic coastline"),
        ("kenya", "an East African sovereign polity governed from Nairobi"),
        ("united_kingdom", "a sovereign union of England Scotland Wales and Northern Ireland"),
        ("germany", "a federal European polity governed from Berlin"),
        ("spain", "an Iberian sovereign polity governed from Madrid"),
        ("italy", "a Mediterranean sovereign polity shaped like a boot"),
        ("china", "an East Asian sovereign polity governed from Beijing"),
        ("south_korea", "an East Asian sovereign polity governed from Seoul"),
        ("india", "a South Asian sovereign polity governed from New Delhi"),
        ("thailand", "a Southeast Asian sovereign polity governed from Bangkok"),
        ("canada", "a North American federation governed from Ottawa"),
        ("australia", "a sovereign polity occupying a large southern landmass"),
        ("brazil", "a Portuguese-speaking federation in South America"),
        ("argentina", "a Spanish-speaking sovereign polity in southern South America"),
        ("nigeria", "a populous West African federation governed from Abuja"),
        ("finland", "a Nordic sovereign polity between Sweden and Russia"),
    ],
    "fruit": [
        ("banana", "long yellow edible produce with a peel"),
        ("apple", "round crisp edible produce commonly red or green"),
        ("mango", "sweet tropical edible produce with a large stone"),
        ("pear", "sweet bell-shaped edible produce with grainy flesh"),
        ("orange", "round citrus produce with a bright peel"),
        ("grape", "small juicy produce growing in clusters on vines"),
        ("lemon", "sour yellow citrus produce rich in aromatic oil"),
        ("lime", "small green citrus produce with acidic juice"),
        ("peach", "soft stone-bearing produce with fuzzy skin"),
        ("plum", "smooth-skinned stone produce often purple"),
        ("cherry", "small red stone produce borne on trees"),
        ("strawberry", "red seeded produce borne on low plants"),
        ("blueberry", "small blue edible berry from a shrub"),
        ("raspberry", "soft aggregate berry composed of many drupelets"),
        ("pineapple", "large tropical produce with a spiny rind and crown"),
        ("watermelon", "large green-rinded produce with watery red flesh"),
        ("papaya", "tropical produce with orange flesh and black seeds"),
        ("kiwi", "small brown fuzzy produce with green flesh"),
        ("pomegranate", "red leathery produce containing many jewel-like seeds"),
        ("apricot", "small orange stone produce related to the peach"),
    ],
    "animal": [
        ("cat", "a small domesticated feline that purrs"),
        ("dog", "a domesticated canine often living with humans"),
        ("elephant", "a huge mammal with a trunk and tusks"),
        ("lion", "a large social feline whose males often have manes"),
        ("tiger", "a striped solitary feline native to Asia"),
        ("giraffe", "a tall African mammal with an exceptionally long neck"),
        ("zebra", "an African equid marked by black and white stripes"),
        ("horse", "a large domesticated equid used for riding"),
        ("cow", "a domesticated bovine raised for milk or meat"),
        ("sheep", "a wool-bearing domesticated ruminant"),
        ("goat", "an agile horned ruminant often kept on farms"),
        ("rabbit", "a small long-eared mammal that hops"),
        ("panda", "a black-and-white bear that eats bamboo"),
        ("kangaroo", "an Australian marsupial that moves by hopping"),
        ("dolphin", "an intelligent marine mammal using echolocation"),
        ("whale", "a very large marine mammal breathing through a blowhole"),
        ("eagle", "a large bird of prey with powerful talons"),
        ("penguin", "a flightless seabird adapted to swimming"),
        ("crocodile", "a large aquatic reptile with armored skin"),
        ("tortoise", "a slow land reptile protected by a shell"),
    ],
}


def sha256_file(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_rows():
    rows=[]
    for category,items in COHORT.items():
        for ordinal,(name,description) in enumerate(items):
            rows.append({"name":name,"description":description,"category":category,
                         "category_ordinal":ordinal})
    return rows


def prospective_fold_map(rows, outer_folds, split_seed):
    """Balanced category-stratified split fixed before outcome inspection."""
    assignment = {}
    for category_index, category in enumerate(sorted(COHORT)):
        indices = [i for i, row in enumerate(rows) if row["category"] == category]
        rng = random.Random(split_seed + 1009 * category_index)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignment[index] = position % outer_folds
    return assignment


@torch.no_grad()
def description_features(model,tok,rows,device,batch_size):
    chunks=[]
    for start in range(0,len(rows),batch_size):
        prompts=["Entity: "+r["name"]+". Clue: "+r["description"] for r in rows[start:start+batch_size]]
        batch=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=48).to(device)
        out=model(**batch,output_hidden_states=True)
        hidden=out.hidden_states[-1]; mask=batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean_h=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1.0)
        last_idx=batch["attention_mask"].sum(1)-1
        last_h=hidden[torch.arange(hidden.shape[0],device=device),last_idx]
        mean_h=torch.nn.functional.normalize(mean_h,dim=-1)
        last_h=torch.nn.functional.normalize(last_h,dim=-1)
        chunks.append((torch.cat([mean_h,last_h],dim=-1)/math.sqrt(2)).float().cpu().numpy())
    return np.concatenate(chunks,axis=0)


def ridge_scores(x,train_idx,train_labels,classes,ridge):
    train=np.asarray(train_idx,dtype=int); labels=np.asarray(train_labels)
    mean=x[train].mean(0); scale=x[train].std(0)+1e-6
    xs=(x-mean)/scale; xf=xs[train]
    y=np.zeros((len(train),len(classes)))
    for i,c in enumerate(labels): y[i,classes.index(c)]=1
    dual=np.linalg.solve(xf@xf.T+ridge*np.eye(len(train)),y)
    return xs@(xf.T@dual)


def prototype_scores(x,train_idx,train_labels,classes):
    train=np.asarray(train_idx,dtype=int); labels=np.asarray(train_labels)
    xn=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12); prototypes=[]
    for c in classes:
        members=xn[train[labels==c]]
        p=members.mean(0); prototypes.append(p/max(np.linalg.norm(p),1e-12))
    return xn@np.stack(prototypes).T


def row_standardize(scores):
    return (scores-scores.mean(1,keepdims=True))/(scores.std(1,keepdims=True)+1e-9)


def candidate_scores(ridge,prototype,policy):
    if policy["kind"]=="ridge": return ridge
    if policy["kind"]=="prototype": return prototype
    a=float(policy["alpha"])
    return a*row_standardize(ridge)+(1-a)*row_standardize(prototype)


def choose_policy_inner_loo(x,train_idx,labels,classes,cfg):
    candidates=[{"kind":"ridge"},{"kind":"prototype"}]
    candidates += [{"kind":"ensemble","alpha":float(a)} for a in cfg["ensemble_alphas"]]
    truth={i:labels[i] for i in train_idx}; collected={json.dumps(p,sort_keys=True):[] for p in candidates}
    for held in train_idx:
        inner=[i for i in train_idx if i!=held]; inner_labels=[labels[i] for i in inner]
        rs=ridge_scores(x,inner,inner_labels,classes,cfg["ridge"])
        ps=prototype_scores(x,inner,inner_labels,classes)
        for policy in candidates:
            sc=candidate_scores(rs,ps,policy); pred=classes[int(np.argmax(sc[held]))]
            collected[json.dumps(policy,sort_keys=True)].append(pred==truth[held])
    table=[]
    for priority,policy in enumerate(candidates):
        vals=collected[json.dumps(policy,sort_keys=True)]
        table.append({**policy,"inner_loo_accuracy":float(np.mean(vals)),"priority":priority})
    selected=max(table,key=lambda r:(r["inner_loo_accuracy"],-r["priority"]))
    return {k:v for k,v in selected.items() if k!="priority"},table


def evaluate_scores(scores,indices,labels,classes):
    rows=[]
    for i in indices:
        order=np.argsort(scores[i]); pred=classes[int(order[-1])]; true=labels[i]
        own=float(scores[i,classes.index(true)])
        other=float(max(scores[i,j] for j,c in enumerate(classes) if c!=true))
        rows.append({"index":int(i),"truth":true,"prediction":pred,"correct":pred==true,
                     "signed_score_margin":own-other})
    return rows


def run_arm(x,train_idx,test_idx,labels,classes,cfg,train_labels=None):
    supplied=[labels[i] for i in train_idx] if train_labels is None else list(train_labels)
    selected,inner=choose_policy_inner_loo(x,train_idx,
        {**{i:labels[i] for i in range(len(labels))},**{i:supplied[k] for k,i in enumerate(train_idx)}},
        classes,cfg)
    rs=ridge_scores(x,train_idx,supplied,classes,cfg["ridge"])
    ps=prototype_scores(x,train_idx,supplied,classes)
    scores=candidate_scores(rs,ps,selected)
    return evaluate_scores(scores,test_idx,labels,classes),selected,inner


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config_quick.json")
    ap.add_argument("--output",default="results"); args=ap.parse_args()
    cfg_path=Path(args.config); cfg=json.loads(cfg_path.read_text()); started=time.time()
    set_seed(cfg["seed"]); rows=make_rows(); labels=[r["category"] for r in rows]
    classes=sorted(COHORT); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg["model_name"],dtype=torch.float32).to(device).eval()
    features=description_features(model,tok,rows,device,cfg["feature_batch_size"])
    fold_map=prospective_fold_map(rows,cfg["outer_folds"],cfg["split_seed"])
    folds=[]; all_current=[]; all_pair=[]; shuffle_acc=[]
    for fold in range(cfg["outer_folds"]):
        test=[i for i in range(len(rows)) if fold_map[i]==fold]
        train=[i for i in range(len(rows)) if i not in test]
        current,policy,inner=run_arm(features,train,test,labels,classes,cfg)
        pair=[]
        for c in classes: pair += [i for i in train if labels[i]==c][:2]
        pair_rows,pair_policy,pair_inner=run_arm(features,pair,test,labels,classes,cfg)
        srng=random.Random(cfg["seed"]+10007*fold); fold_shuffle=[]
        for rep in range(cfg["shuffle_repeats"]):
            shuffled=[labels[i] for i in train]; srng.shuffle(shuffled)
            rr,_,_=run_arm(features,train,test,labels,classes,cfg,shuffled)
            fold_shuffle.append(float(np.mean([r["correct"] for r in rr])))
        folds.append({"fold":fold,"train_indices":train,"heldout_indices":test,
                      "selected_policy":policy,"inner_policy_table":inner,
                      "heldout_predictions":current,
                      "pair_only_policy":pair_policy,"pair_only_inner_table":pair_inner,
                      "pair_only_predictions":pair_rows,"shuffle_accuracies":fold_shuffle})
        all_current+=current; all_pair+=pair_rows; shuffle_acc+=fold_shuffle
    accuracy=float(np.mean([r["correct"] for r in all_current]))
    pair_accuracy=float(np.mean([r["correct"] for r in all_pair]))
    fold_accuracy=[float(np.mean([r["correct"] for r in f["heldout_predictions"]])) for f in folds]
    margins=[r["signed_score_margin"] for r in all_current]
    recalls={c:float(np.mean([r["correct"] for r in all_current if r["truth"]==c])) for c in classes}
    metrics={"concepts":len(rows),"outer_predictions":len(all_current),
             "aggregate_accuracy":accuracy,"worst_fold_accuracy":min(fold_accuracy),
             "fold_accuracies":fold_accuracy,"class_recalls":recalls,
             "minimum_class_recall":min(recalls.values()),
             "median_signed_score_margin":float(np.median(margins)),
             "tenth_percentile_signed_score_margin":float(np.quantile(margins,.10)),
             "pair_only_accuracy":pair_accuracy,"median_shuffle_accuracy":float(np.median(shuffle_acc)),
             "maximum_shuffle_accuracy":max(shuffle_acc),"chance_accuracy":1/len(classes)}
    gates={"aggregate_accuracy":accuracy>=cfg["aggregate_accuracy_gate"],
           "worst_fold_accuracy":min(fold_accuracy)>=cfg["worst_fold_accuracy_gate"],
           "minimum_class_recall":min(recalls.values())>=cfg["minimum_class_recall_gate"],
           "positive_median_margin":metrics["median_signed_score_margin"]>0,
           "positive_tenth_percentile_margin":metrics["tenth_percentile_signed_score_margin"]>0,
           "beats_pair_only":accuracy-pair_accuracy>=cfg["pair_margin_gate"],
           "beats_median_shuffle":accuracy-metrics["median_shuffle_accuracy"]>=cfg["shuffle_margin_gate"],
           "beats_maximum_shuffle":accuracy-metrics["maximum_shuffle_accuracy"]>=cfg["max_shuffle_margin_gate"]}
    record={"protocol":cfg["protocol"],"scientific_status":"ROUTING_ONLY_PREFLIGHT_NOT_L3_CONFIRMATION",
            "seed":cfg["seed"],"split_seed":cfg["split_seed"],
            "fold_assignment":[fold_map[i] for i in range(len(rows))],
            "device":str(device),"classes":classes,"cohort":rows,
            "metrics":metrics,"gates":gates,"all_gates_pass":all(gates.values()),"folds":folds,
            "config":cfg,"config_sha256":sha256_file(cfg_path),"elapsed_seconds":time.time()-started,
            "python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
            "transformers":transformers.__version__}
    (out/"routing_preflight.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    summary={k:record[k] for k in ["protocol","scientific_status","metrics","gates","all_gates_pass",
                                      "config_sha256","elapsed_seconds","python","platform","torch","transformers"]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
