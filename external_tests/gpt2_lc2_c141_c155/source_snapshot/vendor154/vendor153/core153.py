"""Independent arithmetic for fixed C152 two-rate cyclic controls; no model execution."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor152'))
import core152 as parent
from core152 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C153_READONLY_RATE_AUDIT_CPU_V1'
INPUT_KEYS=('C152',)+parent.INPUT_KEYS
TRAIN_ARMS=parent.TRAIN_ARMS;ARMS=parent.EVAL_ARMS;RATES=dict(parent.RATES);N=84;TIMEOUT=1200
no_models=parent.auditcore.no_models

def registry():return parent.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source152=clean_json((ROOT/'trusted/C152_SOURCE_PINS.json').read_bytes()),binding152=parent.contract())
def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),training_states=84,training_updates=24,training_records=2016,native_preflight_records=336,functional_zero_records=84,functional_warm_records=168,frozen_points=336,native_endpoint_records=672,arms=list(ARMS),learning_rates=RATES,acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),arithmetic='Independent explicit-left float64 mixing, positive cyclic shift1 proposal, loss, pullback, exact-max aggregation and two-rate Adam; exact actual initial/final tensors and frozen points; all84 native gates and rescue/destruction pairs reconstructed.',coverage='Eight original ZIPs rehashed, C152 trusted source matched, trusted C152 reader reruns C151 audit of C150 and rebinds all84 inputs. Original C149 remains a sealed audit receipt; older recursive audits are not rerun.',diagnostics='All84 response-coordinate slack and all2016 recorded update/transition diagnostics; both arms diagonal/cross decomposition, response/guard alignment, box projection, baseline-to-final failures and within-run response threshold crossings. Every exposed state remains training. No checkpoint/arm or schedule selected.',worker_budget_seconds=TIMEOUT,scope='Read-only CPU archive and frozen-tensor arithmetic. Archived gradients are unverified derivative inputs. No independent model replay, new endpoint, true-gradient/curvature certification, causal optimizer diagnosis, convergence, feasibility/infeasibility or excluded-input generalisation. Last recorded transition bridges a functional observation and a native endpoint, and includes any forward-path difference. Archive consistency cannot prove measurement accuracy, chronology or non-access. No candidate, checkpoint, schedule or deployment selected.',new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False)

def effective(p,w,arm):
    need(arm in TRAIN_ARMS,'Registered rate arm');farray(p,(3,2,p.shape[-1]));farray(w,(3,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14 and np.max(abs(p),initial=0)<=.125+1e-15,'Convex weights/bounded coupled controls');return (w[0]*p[0]+w[1]*p[1])+w[2]*p[2]

def archived_effective(p,w,arm,digest):
    e=effective(p,w,arm);compare(row_digest(e),digest,'Exact registered explicit-left control digest');return e,'explicit_left_sum'

def pullback(g,w,arm):
    need(arm in TRAIN_ARMS,'Registered rate arm');farray(g,(2,g.shape[-1]));farray(w,(3,));return np.multiply(w.reshape(3,1,1),g[None,...])

def proposal(prepared,raw,V,e):
    farray(prepared,(DIM,));farray(raw,(DIM,));need(prepared.dtype==raw.dtype==np.float32,'Physical float32 domain');rank=V.shape[0];farray(V,(rank,DIM));need(e.shape in ((rank,),(2,rank)),'Effective coefficient blocks');farray(e,e.shape);need(np.max(abs(e),initial=0)<=.125+1e-15,'Effective coefficient bound');u=V@(raw.astype(np.float64)-prepared.astype(np.float64));s=e*u if e.ndim==1 else e[0]*u+e[1]*np.concatenate([u[-1:],u[:-1]]);return (raw.astype(np.float64)+V.T@s).astype(np.float32)

def loss(o,kl,point,prepared,e,case):
    o=farray(np.asarray(o,float),(22,));rank=e.shape[-1];need(e.shape in ((rank,),(2,rank)),'Effective loss shape');farray(e,e.shape);need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden');need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Finite nonnegative KL');x,y=int(case[-2]),int(case[-1]);sign=2*np.tile([x,y,1,x^y],4)-1
    return dict(response_loss=float(max(0.,float(np.max(abs(o[:6])))/.0018-1)**2),read_loss=float(np.mean(np.maximum(0.,1-sign*o[6:]/.075)**2)),KL_loss=float(max(0.,kl/.0045-1)**2),radius_loss=float(max(0.,float(np.linalg.norm(point.astype(float)-prepared.astype(float)))/20-1)**2),regularizer=float(.001*np.sum(e*e)/rank))

def aggregate(rows,shape):
    need(len(rows)==84 and [r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All84 ordered state gradients');R=farray(np.array([r['objective_parts']['response_loss'] for r in rows],float),(84,));G=farray(np.array([sum(v for k,v in r['objective_parts'].items() if k!='response_loss') for r in rows],float),(84,));need(np.all(R>=0) and np.all(G>=0),'Nonnegative objective');rg=np.array([farray(np.asarray(r['response_gradient'],float),shape) for r in rows]).reshape(84,-1);gg=np.array([farray(np.asarray(r['guard_gradient'],float),shape) for r in rows]).reshape(84,-1);top=R==R.max();weights=np.full(84,.1/84)+.9*top/int(top.sum());contrib=weights[:,None]*rg+gg/84
    return contrib.sum(0).reshape(shape),dict(objective=float(.9*R.max()+.1*R.mean()+G.mean()),response_weights=weights.tolist(),guard_weights=[1/84]*84,maximum_response_loss=float(R.max()),mean_response_loss=float(R.mean()),mean_guard_loss=float(G.mean()),maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(top)],exact_max_tie_count=int(top.sum()),state_gradient_contributions=contrib.tolist())

def update(p,g,m,v,step,arm):
    need(arm in TRAIN_ARMS and type(step) is int and 1<=step<=12,'Fixed registered rate and12-step schedule');need(p.shape[:2]==(3,2),'Coupled head shape')
    for x in (p,g,m,v):farray(x,p.shape)
    need(np.all(v>=0),'Nonnegative second moment');norm=float(np.linalg.norm(g));factor=1. if norm<=1. else 1./norm;clipped=factor*g;mm=.9*m+.1*clipped;vv=.999*v+.001*clipped*clipped;unrestricted=p-RATES[arm]*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.clip(unrestricted,-.125,.125)
    return new,mm,vv,dict(step=step,learning_rate=RATES[arm],gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.count_nonzero(new!=unrestricted)),update_Linf=float(np.max(abs(new-p),initial=0)))

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
