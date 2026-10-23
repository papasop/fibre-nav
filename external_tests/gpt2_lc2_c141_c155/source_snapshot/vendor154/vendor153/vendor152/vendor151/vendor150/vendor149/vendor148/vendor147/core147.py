"""Independent arithmetic for recorded C146 training; never an independent derivative check."""
from pathlib import Path
from collections import Counter
import builtins,contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor146'))
import core146 as parent
from core146 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C147_READONLY_WORST_RESPONSE_AUDIT_CPU_V1'
INPUT_KEYS=('C146','C145','C143');N=84;TIMEOUT=1200
ARMS=('raw_control','C143_frozen_step12','C146_fixed_step12')

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source146=clean_json((ROOT/'trusted/C146_SOURCE_PINS.json').read_bytes()),binding146=parent.contract(),expected_parent_decision='C146_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
def registry():return parent.registry()
def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),training_states=84,training_steps=12,training_records=1008,native_preflight_records=336,zero_parity_records=84,warm_parity_records=84,sealed_final_proposals=252,final_endpoint_records=504,arms=list(ARMS),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),training_safety=dict(response=.0018,read_margin=.075,KL=.0045),arithmetic='Independently express max-hinge and guard losses, exact-max tie weights, per-state gradient contributions, clipped/projected Adam, finite proposals, native gates and paired rescue/destruction. Use recorded response/guard gradient vectors as unverified inputs.',diagnostics='All84 state histories, exact active response coordinates, maximizer switches, response/guard contribution norms and alignments, box projection, actual update displacements, archived-gradient linear predictions versus next recorded observations; final comparison uses native endpoints and is labelled separately.',coverage='Rehash original C146/C145/C143 bytes. Recompute C145 archival arithmetic with delivered pinned reader. Other linked original C144/C140/C141/C142 ZIP bytes are not reopened or rehashed.',parent_decision_reclassified=False,candidate_selected=False,LC2_closed=False,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,archived_source_executed=False,worker_budget_seconds=TIMEOUT,scope='CPU frozen-tensor and archive arithmetic only; no model replay or true-gradient recomputation. Cannot prove measurement accuracy, chronology, non-access, stationary/converged behaviour, step-size causality, feature insufficiency, feasibility/infeasibility or generalisation. Predictions across max switches, float32 commits and nonlinear endpoints are descriptive, not derivative-error or curvature bounds. No new checkpoint or schedule is selected.')

@contextlib.contextmanager
def no_models():
    original=builtins.__import__
    def checked(name,*args,**kwargs):
        need(name.split('.')[0] not in {'torch','transformers','peft','tokenizers','huggingface_hub','accelerate','safetensors'},'C147 forbids model-library imports')
        return original(name,*args,**kwargs)
    builtins.__import__=checked
    try:yield
    finally:builtins.__import__=original

def proposal(prepared,raw,V,a):
    farray(prepared,(DIM,));farray(raw,(DIM,));farray(V,(len(a),DIM));farray(a,(len(a),));need(prepared.dtype==raw.dtype==np.float32 and np.max(abs(a),initial=0)<=.125+1e-15,'Physical dtype/gain bound')
    d=raw.astype(np.float64)-prepared.astype(np.float64);shift=V.T@(a*(V@d));point=(raw.astype(np.float64)+shift).astype(np.float32)
    return point,dict(correction_L2=float(np.linalg.norm(shift)),actual_delta_norm=float(np.linalg.norm(point.astype(float)-prepared.astype(float))),float32_commit_error_Linf=float(np.max(abs(point.astype(float)-raw.astype(float)-shift))),radial_clipping_performed=False)

def loss(o,kl,point,prepared,a,case):
    o=farray(np.asarray(o,float),(22,));farray(a,(len(a),));need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden');need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Invalid KL')
    x,y=int(case[-2]),int(case[-1]);signs=2*np.tile([x,y,1,x^y],4)-1
    response=max(0.,float(np.max(abs(o[:6])))/.0018-1.)**2
    read=float(np.mean(np.maximum(0.,1.-signs*o[6:]/.075)**2));klpart=max(0.,kl/.0045-1.)**2;radius=max(0.,float(np.linalg.norm(point.astype(float)-prepared.astype(float)))/20.-1.)**2
    return dict(response_loss=response,read_loss=read,KL_loss=klpart,radius_loss=radius,regularizer=float(.001*np.mean(a*a)))

def aggregate(rows,rank):
    need(len(rows)==84 and [r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All84 ordered gradient records')
    response=farray(np.array([r['objective_parts']['response_loss'] for r in rows],float),(84,));guard=farray(np.array([sum(v for k,v in r['objective_parts'].items() if k!='response_loss') for r in rows],float),(84,));need(np.all(response>=0) and np.all(guard>=0),'Nonnegative loss required')
    rg=farray(np.array([r['response_gradient'] for r in rows],float),(84,rank));gg=farray(np.array([r['guard_gradient'] for r in rows],float),(84,rank));top=response==np.max(response);weights=.1/84+.9*top/int(np.count_nonzero(top));contributions=weights[:,None]*rg+gg/84
    detail=dict(objective=float(.9*np.max(response)+.1*np.mean(response)+np.mean(guard)),response_weights=weights.tolist(),guard_weights=[1/84]*84,maximum_response_loss=float(np.max(response)),mean_response_loss=float(np.mean(response)),mean_guard_loss=float(np.mean(guard)),maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(top)],exact_max_tie_count=int(np.count_nonzero(top)),state_gradient_contributions=contributions.tolist())
    return np.sum(contributions,axis=0),detail

def update(a,g,m,v,step):
    need(type(step) is int and 1<=step<=12,'Fixed12 update schedule')
    for x in (a,g,m,v):farray(x,a.shape)
    need(np.all(v>=0),'Negative Adam second moment');norm=float(np.linalg.norm(g));factor=1. if norm<=1 else 1./norm;clipped=g*factor
    mm=.9*m+.1*clipped;vv=.999*v+.001*clipped**2
    unrestricted=a-.0005*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.minimum(.125,np.maximum(-.125,unrestricted))
    return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.count_nonzero(new!=unrestricted)),update_Linf=float(np.max(abs(new-a),initial=0)))

def dot_diagnostic(g,delta):
    g=farray(np.asarray(g,float),delta.shape);delta=farray(np.asarray(delta,float),delta.shape)
    return dict(dot=float(g@delta),gradient_L2=float(np.linalg.norm(g)),coefficient_delta_L2=float(np.linalg.norm(delta)))

def final_tables(records):
    sums={};pairs={}
    groups=dict(all84=84,historical12=12,C139_exposed_kernel24=24,C141_exposed_confirmation48=48,xor_00=28,xor_01=28,xor_11=28)
    for group,n in groups.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];by={(r['state_id'],r['arm']):r for r in rr};need(len(by)==n*3,'All native records retained');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in ARMS:
                ar=[r for r in rr if r['arm']==arm];need(len(ar)==n,'Complete native denominator');bad=Counter(g for r in ar for g in r['gates']['failed_gates'])
                sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(sorted(bad.items())),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar));pairs[group][str(rnd)][arm]={}
                for control in ARMS[:2]:pairs[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not by[(r['state_id'],control)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and by[(r['state_id'],control)]['gates']['complete_pass']])
    return sums,pairs
