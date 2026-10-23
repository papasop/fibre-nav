"""Independent arithmetic and descriptive diagnostics for frozen C148 archives."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor148'))
import core148 as parent
from core148 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C149_READONLY_ROUTING_AUDIT_CPU_V1'
INPUT_KEYS=('C148','C147','C146','C145','C143');N=84;K=3;TIMEOUT=1200
TRAIN_ARMS=parent.TRAIN_ARMS;ARMS=parent.EVAL_ARMS
no_models=parent.previous.no_models

def registry():return parent.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source148=clean_json((ROOT/'trusted/C148_SOURCE_PINS.json').read_bytes()),binding148=parent.contract())
def design():return dict(protocol=PROTOCOL,original_archives=list(INPUT_KEYS),training_states=84,training_updates=24,training_records=2016,archived_scalar_backward_calls=4032,native_preflight_records=336,functional_zero_records=84,functional_warm_records=168,sealed_final_proposals=336,native_endpoint_records=672,arms=list(ARMS),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),arithmetic='Independent fixed router-construction replay, prewrite weights, effective coefficients, scalar-loss parts, linear routing gradient pullbacks, max/mean aggregation, all24 manual Adam updates, exact final tensors, all336 float32 proposals and672 native gate records.',registered_router_iteration_replay=8,router_replay_scope='Recompute the original declared mean/std, fixed seed selection and eight Lloyd arithmetic iterations only to compare against frozen tensors; no alternative router is fitted, selected, saved as a candidate or deployed.',diagnostics='All84 route weights and entropy; fixed-weight variation; per-step expert coefficient separation and recorded gradient alignment; effective-control and actual float32 proposal differences; all84 final signed response/slack and rescue/destruction. Exact equality and magnitudes are descriptive, not a sufficiency or causality threshold.',coverage='Original C148/C147/C146/C145/C143 bytes reopened and rehashed. Rerun trusted C147 archive arithmetic, then bind exact C146 final initialization. Older linked C144/C140/C141/C142 originals not reopened.',worker_budget_seconds=TIMEOUT,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_router_candidates=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,gradients_independently_recomputed=False,archived_source_executed=False,thresholds_changed=False,candidate_selected=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False,scope='Pure NumPy archival arithmetic and frozen-tensor diagnostics only. Recorded effective-control gradients are unverified derivative inputs. No model replay, new measurement, new fit/candidate, true-gradient certification, physical failure causality, routing superiority, convergence, feasibility/infeasibility, geometric atlas or excluded-input generalisation established. Archive consistency cannot prove chronology, non-access or measurement accuracy. Parameter count and optimizer geometry remain confounds.')

def projected_context(prepared,raw,V):
    farray(prepared,(DIM,));farray(raw,(DIM,));need(prepared.dtype==raw.dtype==np.float32,'Original float32 coordinates');farray(V,V.shape);need(V.ndim==2 and V.shape[1]==DIM,'Basis dimensions');return V@(raw.astype(float)-prepared.astype(float))

def route(u,r):
    need(set(r)=={'mean','scale','centers'},'Router tensor inventory');farray(u,r['mean'].shape);farray(r['mean'],u.shape);farray(r['scale'],u.shape);farray(r['centers'],(3,len(u)));need(np.all(r['scale']>=1e-8),'Normalizer floor');z=(u-r['mean'])/r['scale'];l=-.5*np.mean((z[None,:]-r['centers'])**2,axis=1);need(np.isfinite(l).all(),'Finite router logits');e=np.exp(l-np.max(l));w=e/np.sum(e);need(np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-1)<1e-14,'Convex weights');return w

def registered_router_arithmetic(U):
    need(U.ndim==2 and U.shape[0]==84 and U.shape[1]>0,'All84 original contexts');farray(U,U.shape);mean=np.mean(U,axis=0);scale=np.maximum(np.std(U,axis=0),1e-8);Z=(U-mean)/scale;seed=[0];centers=[Z[0].copy()]
    for _ in range(2):
        distances=np.min(np.stack([np.mean((Z-c)**2,axis=1) for c in centers]),axis=0);distances[seed]=-1.;i=int(np.argmax(distances));seed.append(i);centers.append(Z[i].copy())
    centers=np.array(centers);empty=[]
    for iteration in range(8):
        labels=np.argmin(np.mean((Z[:,None,:]-centers[None,:,:])**2,axis=2),axis=1);new=centers.copy()
        for j in range(3):
            members=labels==j
            if np.any(members):new[j]=Z[members].mean(0)
            else:empty.append(dict(iteration=iteration+1,cluster=j))
        centers=new
    router=dict(mean=mean,scale=scale,centers=centers);W=np.array([route(u,router) for u in U]);meta=dict(seed_registry_indices=seed,lloyd_iterations=8,empty_cluster_events=empty,training_hard_cluster_counts=np.bincount(np.argmax(W,axis=1),minlength=3).tolist(),soft_effective_counts=W.sum(0).tolist(),router_inputs_have_no_endpoint_labels=True);return router,W,meta

def effective(p,w,arm):
    need(arm in TRAIN_ARMS and p.ndim==(1 if arm==TRAIN_ARMS[0] else 2),'Registered control shape');farray(p,p.shape);need(np.max(abs(p),initial=0)<=.125+1e-15,'Original coefficient box');farray(w,(3,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14,'Convex weights')
    if arm==TRAIN_ARMS[0]:return p.copy()
    need(p.shape[0]==3,'Three experts');return w@p

def pullback(g,w,arm):
    farray(g,(len(g),));return g.copy() if arm==TRAIN_ARMS[0] else w[:,None]*g[None,:]

def proposal(prepared,raw,V,a):
    farray(prepared,(DIM,));farray(raw,(DIM,));farray(a,(len(a),));farray(V,(len(a),DIM));need(prepared.dtype==raw.dtype==np.float32 and np.max(abs(a),initial=0)<=.125+1e-15,'Physical dtype/coefficient bound');u=V@(raw.astype(float)-prepared.astype(float));return (raw.astype(float)+V.T@(a*u)).astype(np.float32)

def loss(o,kl,p,prepared,a,case):
    o=farray(np.asarray(o,float),(22,));farray(a,(len(a),));farray(p,(DIM,));farray(prepared,(DIM,));need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden');need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Finite nonnegative KL');x,y=int(case[-2]),int(case[-1]);sign=2*np.tile([x,y,1,x^y],4)-1
    return dict(response_loss=float(max(np.max(abs(o[:6]))/.0018-1,0)**2),read_loss=float(np.mean(np.maximum(1-sign*o[6:]/.075,0)**2)),KL_loss=float(max(kl/.0045-1,0)**2),radius_loss=float(max(np.linalg.norm(p.astype(float)-prepared.astype(float))/20-1,0)**2),regularizer=float(.001*np.mean(a*a)))

def aggregate(rows,shape):
    need([r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All84 ordered gradient rows');R=farray(np.array([r['objective_parts']['response_loss'] for r in rows],float),(84,));G=farray(np.array([sum(v for k,v in r['objective_parts'].items() if k!='response_loss') for r in rows],float),(84,));need(np.all(R>=0) and np.all(G>=0),'Nonnegative losses');rg=np.stack([farray(np.asarray(r['response_gradient'],float),shape) for r in rows]).reshape(84,-1);gg=np.stack([farray(np.asarray(r['guard_gradient'],float),shape) for r in rows]).reshape(84,-1);ties=R==np.max(R);weights=.1/84+.9*ties/int(ties.sum());contrib=weights[:,None]*rg+gg/84
    return contrib.sum(0).reshape(shape),dict(objective=float(.9*R.max()+.1*R.mean()+G.mean()),response_weights=weights.tolist(),guard_weights=[1/84]*84,maximum_response_loss=float(R.max()),mean_response_loss=float(R.mean()),mean_guard_loss=float(G.mean()),maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(ties)],exact_max_tie_count=int(ties.sum()),state_gradient_contributions=contrib.tolist())

def update(p,g,m,v,step):
    need(type(step) is int and 1<=step<=12,'Fixed12 per arm')
    for x in (p,g,m,v):farray(x,p.shape)
    need(np.all(v>=0),'Nonnegative Adam v');norm=float(np.linalg.norm(g));factor=min(1.,1/max(norm,1e-30));gclip=g*factor;mm=.9*m+.1*gclip;vv=.999*v+.001*gclip*gclip;unrestricted=p-.0005*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.minimum(.125,np.maximum(-.125,unrestricted));return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.sum(new!=unrestricted)),update_Linf=float(np.max(abs(new-p),initial=0)))

def entropy(w):return float(-np.sum(w[w>0]*np.log(w[w>0])))
def cosine(a,b):
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));return float(np.clip(np.sum(a*b)/(na*nb),-1,1)) if na>0 and nb>0 else None

def pairwise_rows(A):return [dict(i=i,j=j,Linf=float(np.max(abs(A[i]-A[j]),initial=0)),L2=float(np.linalg.norm(A[i]-A[j])),cosine=cosine(A[i],A[j])) for i in range(len(A)) for j in range(i+1,len(A))]

def routing_diagnostics(W):
    farray(W,(84,3));need(np.all(W>=0) and np.max(abs(W.sum(1)-1))<1e-14,'Convex all84 weights');ranges=np.ptp(W,axis=0);ent=[entropy(w) for w in W]
    return dict(maximum_weight_range_across_states=float(ranges.max()),per_expert_weight_ranges=ranges.tolist(),all_training_weights_exactly_identical=bool(np.array_equal(W,np.tile(W[0],(84,1)))),minimum_entropy=float(min(ent)),maximum_entropy=float(max(ent)),mean_entropy=float(np.mean(ent)),uniform_entropy=float(np.log(3)),hard_assignment_counts=np.bincount(np.argmax(W,axis=1),minlength=3).tolist(),soft_effective_counts=W.sum(0).tolist(),maximum_pairwise_weight_L1=float(max(np.sum(abs(w-W),axis=1).max() for w in W)),sufficiency_threshold_defined=False)

def control_diagnostics(global_p,local_p,W):
    farray(local_p,(3,len(global_p)));eff=W@local_p;pair=pairwise_rows(local_p);delta=eff-global_p[None,:]
    return dict(expert_pairs=pair,maximum_expert_coefficient_pairwise_Linf=max(x['Linf'] for x in pair),all_expert_controls_exactly_identical=bool(np.array_equal(local_p,np.tile(local_p[0],(3,1)))),maximum_effective_control_vs_global_Linf=float(np.max(abs(delta))),mean_effective_control_vs_global_Linf=float(np.mean(np.max(abs(delta),axis=1))),maximum_effective_control_range_across_states=float(np.ptp(eff,axis=0).max()),states_with_exactly_identical_effective_control_to_global=int(np.sum(np.all(eff==global_p,axis=1))))

def gate_signature(r):
    # Compare pass/fail decisions, not numerical margins or drift stored beside them.
    return dict(native_gates=r['metrics']['gates'],proposal_radius_passed=r['gates']['proposal_radius_passed'],preparation_passed='preparation' not in r['gates']['failed_gates'],complete_pass=r['gates']['complete_pass'])

def final_tables(records):
    sums={};pairs={}
    for group,n in parent.GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];lookup={(r['state_id'],r['arm']):r for r in rr};need(len(rr)==len(lookup)==4*n,'Complete native denominator');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in ARMS:
                ar=[r for r in rr if r['arm']==arm];bad=Counter(k for r in ar for k in r['gates']['failed_gates']);sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(bad));pairs[group][str(rnd)][arm]={}
                for ref in ARMS:pairs[group][str(rnd)][arm][ref]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not lookup[(r['state_id'],ref)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and lookup[(r['state_id'],ref)]['gates']['complete_pass']])
    return sums,pairs
