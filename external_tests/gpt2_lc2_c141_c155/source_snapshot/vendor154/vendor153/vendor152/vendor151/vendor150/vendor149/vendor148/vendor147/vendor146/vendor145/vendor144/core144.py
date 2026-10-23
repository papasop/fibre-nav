"""Independent NumPy arithmetic for the fixed C143 archive; no model imports."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor143'))
import core143 as parent
from core143 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM
from core142 import native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C144_READONLY_JOINT84_AUDIT_CPU_V1'
INPUT_KEYS=('C143','C140','C141','C142');TIMEOUT=1200
ARMS=('raw_control','C140_frozen_step12','C143_shared_step12')
GROUPS={'all84':84,'historical12':12,'C139_exposed_kernel24':24,'C141_exposed_confirmation48':48,'xor_00':28,'xor_01':28,'xor_11':28}

def contract():
    return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source143=clean_json((ROOT/'trusted/C143_SOURCE_PINS.json').read_bytes()),binding143=parent.contract())

def design():
    return dict(protocol=PROTOCOL,parent_protocol=parent.PROTOCOL,input_archives=list(INPUT_KEYS),training_states=84,steps=12,training_observations=1008,preflight_records=252,zero_parity_records=84,warm_parity_records=84,frozen_proposals=252,final_endpoint_records=504,rounds=2,arms=list(ARMS),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),training_safety=dict(response=.00125,read_margin=.075,KL=.0045),audit_scope='Rehash all four original archives; rebind C140/C141/C142 inputs, fixed source and full3211 features; independently express objectives, aggregation, clipped Adam and finite proposal maps; recompute all native gates and repeat/control comparisons. Archived gradients are inputs to arithmetic, not independently verified derivatives.',failure_focus='Retain every state; report all failed states, each coordinate budget slack, rescued/destroyed pairs and per-step losses/weights/gradient contributions. No best checkpoint or new candidate.',worker_budget_seconds=TIMEOUT,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,excluded_input_states_evaluated=0,LC2_closed=False)

def loss(observed,kl,point,prepared,a,case):
    o=farray(np.asarray(observed,float),(22,));need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden')
    need(type(kl) in (float,int) and np.isfinite(kl) and kl>=0,'Invalid KL')
    x,y=int(case[-2]),int(case[-1]);truth=np.tile([x,y,1,x^y],4)
    response=np.square(np.clip(np.abs(o[:6])/.00125-1,0,None))
    reads=np.square(np.clip(1-(2*truth-1)*o[6:]/.075,0,None))
    return dict(response_loss=float(np.sum(response)/6+np.max(response)),read_loss=float(np.sum(reads)/16),KL_loss=float(np.square(max(0.,kl/.0045-1))),radius_loss=float(np.square(max(0.,np.linalg.norm(point.astype(float)-prepared.astype(float))/20-1))),regularizer=float(.001*np.mean(np.square(a))))

def aggregate(losses,gradients):
    values=farray(np.asarray(losses,float),(84,));grads=np.asarray(gradients,float)
    need(grads.ndim==2 and grads.shape[0]==84 and grads.shape[1]>0 and np.isfinite(grads).all() and np.all(values>=0),'All84 finite losses and gradient rows required')
    maximum=float(max(values));exponents=np.exp(8*(values-maximum));probabilities=exponents/np.sum(exponents)
    weights=np.full(84,.1/84)+.9*probabilities
    value=float(.1*np.mean(values)+.9*(maximum+np.log(np.sum(exponents)/84)/8))
    gradient=np.sum(weights[:,None]*grads,axis=0)
    return gradient,dict(objective=value,weights=weights.tolist(),state_losses=values.tolist(),temperature=8.,uniform_fraction=.1,maximum_state_loss=maximum,mean_state_loss=float(np.mean(values)))

def update(a,g,m,v,step):
    need(type(step) is int and 1<=step<=12,'Fixed12 update index required')
    for x in (a,g,m,v):farray(x,a.shape)
    need(np.all(v>=0),'Negative second moment')
    norm=float(np.linalg.norm(g));factor=1. if norm<=1. else 1./norm;clipped=g*factor
    first=.9*m+(1-.9)*clipped;second=.999*v+(1-.999)*np.square(clipped)
    unrestricted=a-.0005*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
    result=np.minimum(.125,np.maximum(-.125,unrestricted))
    return result,first,second,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.count_nonzero(result!=unrestricted)),update_Linf=float(np.max(np.abs(result-a),initial=0)))

def proposal(prepared,raw,V,a):
    farray(prepared,(DIM,));farray(raw,(DIM,));farray(V,(len(a),DIM));farray(a,(len(a),))
    need(prepared.dtype==raw.dtype==np.float32 and np.max(abs(a),initial=0)<=.125+1e-15,'Float32 points and bounded shared gain required')
    displacement=raw.astype(float)-prepared.astype(float);correction=V.T@(a*(V@displacement))
    point=np.asarray(raw.astype(float)+correction,dtype=np.float32)
    diagnostics=dict(correction_L2=float(np.linalg.norm(correction)),actual_delta_norm=float(np.linalg.norm(point.astype(float)-prepared.astype(float))),float32_commit_error_Linf=float(np.max(abs(point.astype(float)-raw.astype(float)-correction))),radial_clipping_performed=False)
    return point,diagnostics

def tables(records):
    summaries={};paired={}
    for group,n in GROUPS.items():
        summaries[group]={};paired[group]={}
        for rnd in (0,1):
            selected=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)]
            lookup={(r['state_id'],r['arm']):r for r in selected};need(len(lookup)==n*3,'Missing/duplicate state denominator')
            summaries[group][str(rnd)]={};paired[group][str(rnd)]={}
            for arm in ARMS:
                rr=[r for r in selected if r['arm']==arm];need(len(rr)==n,'Incomplete group denominator')
                gates=Counter(k for r in rr for k in r['recomputed_gates']['failed_gates'])
                item=dict(n_requested=n,n_evaluated=n,passed=sum(r['recomputed_gates']['complete_pass'] for r in rr),failed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass']],failed_gates=dict(sorted(gates.items())),maximum_response_drift=max(r['metrics']['response_drift'] for r in rr),within_training_response_safety=sum(r['metrics']['response_drift']<=.00125 for r in rr),mean_training_objective=float(np.mean([r['training_objective'] for r in rr])))
                if group=='all84':item['aggregate_training_objective']=aggregate([r['training_objective'] for r in rr],np.zeros((84,1)))[1]['objective']
                summaries[group][str(rnd)][arm]=item;paired[group][str(rnd)][arm]={}
                for control in ARMS[:2]:
                    paired[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in rr if r['recomputed_gates']['complete_pass'] and not lookup[(r['state_id'],control)]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass'] and lookup[(r['state_id'],control)]['recomputed_gates']['complete_pass']])
    return summaries,paired

def response_diagnostic(old,new):
    old=farray(np.asarray(old,float),(6,));new=farray(np.asarray(new,float),(6,));effect=new-old
    return dict(C140_signed_response=old.tolist(),C143_signed_response=new.tolist(),measured_response_change=effect.tolist(),C140_coordinate_budget_slack=(.002-abs(old)).tolist(),C143_coordinate_budget_slack=(.002-abs(new)).tolist(),C143_coordinate_excess=np.maximum(abs(new)-.002,0).tolist(),C140_Linf=float(np.max(abs(old))),C143_Linf=float(np.max(abs(new))),maximum_response_change_Linf=float(np.max(abs(effect))),identity_closure_Linf=float(np.max(abs(old+effect-new))),finite_effect_is_not_derivative_or_curvature_proof=True)
