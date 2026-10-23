"""Independent arithmetic for fixed C150 diagonal/cyclic controls; no model execution."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor150'))
import core150 as parent
from core150 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C151_READONLY_CYCLIC_AUDIT_CPU_V1'
INPUT_KEYS=('C150',)+parent.INPUT_KEYS
TRAIN_ARMS=parent.TRAIN_ARMS;ARMS=parent.EVAL_ARMS;N=84;TIMEOUT=1200
no_models=parent.auditcore.no_models

def registry():return parent.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source150=clean_json((ROOT/'trusted/C150_SOURCE_PINS.json').read_bytes()),binding150=parent.contract())
def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),training_states=84,training_updates=24,training_records=2016,native_preflight_records=336,functional_zero_records=84,functional_warm_records=168,frozen_points=336,native_endpoint_records=672,arms=list(ARMS),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),arithmetic='Effective-control hashes must exactly match one of two registered equivalent float64 mixing orders (explicit left sum or NumPy matmul); unknown hashes block. Independent diagonal-plus-positive-roll1 map, loss units, effective-gradient pullback, exact-max/mean aggregation and Adam updates; exact actual final tensor and frozen-point identity; reconstruct every native gate and rescue/destruction pair.',diagnostics='All84 signed residuals/slack, final diagonal/cross coefficient and unrounded parameter-shift decomposition, float32 commit residual, recorded-gradient/update dots and next recorded loss changes; response/guard aggregate alignment and projection events. Cross-block algebra is not an ablation endpoint measurement or a causal attribution.',coverage='Seven original ZIPs rehashed and manifests checked. Trusted C150 reader rebinds all84 origin rows and C148 actual points/gates. Original C149 is a sealed prior audit receipt; no cross-environment re-execution of the older recursive exact-equality audit is claimed.',worker_budget_seconds=TIMEOUT,scope='CPU archive/frozen-tensor arithmetic only. Recorded scalar gradient arrays remain unverified derivative inputs. No model replay, new observation, true-gradient certification, counterfactual endpoint, causal failure explanation, optimum, feasibility/infeasibility, architecture superiority or excluded-input generalisation. Archive consistency cannot prove chronology, non-access or measurement accuracy. Parameter count/optimizer geometry remain confounds. No arm, checkpoint, new schedule or coupling pattern is selected.',new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False)

def effective(p,w,arm):
    need(arm in TRAIN_ARMS,'Registered arm');rank=p.shape[-1];farray(p,(3,rank) if arm==TRAIN_ARMS[0] else (3,2,rank));farray(w,(3,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14 and np.max(abs(p),initial=0)<=.125+1e-15,'Convex weights/bounded controls')
    return (w[0]*p[0]+w[1]*p[1])+w[2]*p[2]

def archived_effective(p,w,arm,digest):
    # Exactly two declared float64 evaluation orders of the same three-term sum.
    # No coefficient/route/threshold is changed; unknown digests are rejected.
    left=effective(p,w,arm);native=w@p if arm==TRAIN_ARMS[0] else np.stack([w@p[:,0,:],w@p[:,1,:]])
    close(left,native,'Equivalent registered effective-mix evaluation orders',1e-15)
    variants={'explicit_left_sum':left,'numpy_matmul':native}
    matches=[name for name,e in variants.items() if row_digest(e)==digest]
    need(bool(matches),'Archived effective-control digest matches no registered arithmetic order')
    return variants[matches[0]],matches[0]

def pullback(g,w,arm):
    rank=g.shape[-1];farray(g,(rank,) if arm==TRAIN_ARMS[0] else (2,rank));farray(w,(3,));return np.multiply(w.reshape((3,)+(1,)*g.ndim),g[None,...])

def proposal(prepared,raw,V,e):
    farray(prepared,(DIM,));farray(raw,(DIM,));need(prepared.dtype==raw.dtype==np.float32,'Physical float32 domain');rank=V.shape[0];farray(V,(rank,DIM));need(e.shape in ((rank,),(2,rank)),'Effective coefficient blocks');farray(e,e.shape);need(np.max(abs(e),initial=0)<=.125+1e-15,'Effective coefficient bound');u=V@(raw.astype(np.float64)-prepared.astype(np.float64));s=e*u if e.ndim==1 else e[0]*u+e[1]*np.concatenate([u[-1:],u[:-1]]);return (raw.astype(np.float64)+V.T@s).astype(np.float32)

def loss(o,kl,point,prepared,e,case):
    o=farray(np.asarray(o,float),(22,));rank=e.shape[-1];need(e.shape in ((rank,),(2,rank)),'Effective loss shape');farray(e,e.shape);need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden');need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Finite nonnegative KL');x,y=int(case[-2]),int(case[-1]);sign=2*np.tile([x,y,1,x^y],4)-1
    return dict(response_loss=float(max(0.,float(np.max(abs(o[:6])))/.0018-1)**2),read_loss=float(np.mean(np.maximum(0.,1-sign*o[6:]/.075)**2)),KL_loss=float(max(0.,kl/.0045-1)**2),radius_loss=float(max(0.,float(np.linalg.norm(point.astype(float)-prepared.astype(float)))/20-1)**2),regularizer=float(.001*np.sum(e*e)/rank))

def aggregate(rows,shape):
    need(len(rows)==84 and [r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All84 ordered state gradients');R=farray(np.array([r['objective_parts']['response_loss'] for r in rows],float),(84,));G=farray(np.array([sum(v for k,v in r['objective_parts'].items() if k!='response_loss') for r in rows],float),(84,));need(np.all(R>=0) and np.all(G>=0),'Nonnegative objective');rg=np.array([farray(np.asarray(r['response_gradient'],float),shape) for r in rows]).reshape(84,-1);gg=np.array([farray(np.asarray(r['guard_gradient'],float),shape) for r in rows]).reshape(84,-1);top=R==R.max();weights=np.full(84,.1/84)+.9*top/int(top.sum());contrib=weights[:,None]*rg+gg/84
    return contrib.sum(0).reshape(shape),dict(objective=float(.9*R.max()+.1*R.mean()+G.mean()),response_weights=weights.tolist(),guard_weights=[1/84]*84,maximum_response_loss=float(R.max()),mean_response_loss=float(R.mean()),mean_guard_loss=float(G.mean()),maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(top)],exact_max_tie_count=int(top.sum()),state_gradient_contributions=contrib.tolist())

def update(p,g,m,v,step):
    need(type(step) is int and 1<=step<=12,'Fixed12 schedule')
    for x in (p,g,m,v):farray(x,p.shape)
    need(np.all(v>=0),'Nonnegative second moment');norm=float(np.linalg.norm(g));factor=1. if norm<=1. else 1./norm;clipped=factor*g;mm=.9*m+.1*clipped;vv=.999*v+.001*clipped*clipped;unrestricted=p-.0005*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.clip(unrestricted,-.125,.125)
    return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.count_nonzero(new!=unrestricted)),update_Linf=float(np.max(abs(new-p),initial=0)))

def cosine(x,y):
    a=float(np.linalg.norm(x));b=float(np.linalg.norm(y));return None if a==0 or b==0 else float(np.clip(np.sum(x*y)/(a*b),-1,1))

def norm(x):return float(np.linalg.norm(x))
def linf(x):return float(np.max(abs(x),initial=0))

def final_tables(records):
    sums={};pairs={}
    for group,n in parent.GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];by={(r['state_id'],r['arm']):r for r in rr};need(len(rr)==len(by)==4*n,'All native records retained');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in ARMS:
                ar=[r for r in rr if r['arm']==arm];bad=Counter(g for r in ar for g in r['gates']['failed_gates']);sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(bad));pairs[group][str(rnd)][arm]={}
                for ref in ARMS:pairs[group][str(rnd)][arm][ref]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not by[(r['state_id'],ref)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and by[(r['state_id'],ref)]['gates']['complete_pass']])
    return sums,pairs
