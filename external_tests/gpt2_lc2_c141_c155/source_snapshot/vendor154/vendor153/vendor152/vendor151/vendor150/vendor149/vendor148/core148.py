"""Fixed training-only soft routing; no claim of an atlas or excluded-input acceptance."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor147'))
import core147 as previous
from core147 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
parent=previous.parent
PROTOCOL='GPT2_LC2_C148_CONTEXT_ROUTED_TRAINING_CPU_V1'
INPUT_KEYS=('C147','C146','C145','C143');N=84;K=3;STEPS=12;TIMEOUT=14400
TRAIN_ARMS=('global_continuation','context_routed')
EVAL_ARMS=('raw_control','C146_frozen_step12')+TRAIN_ARMS
RATE=.0005;BOUND=.125;LLOYD_STEPS=8
GROUPS=parent.GROUPS

def registry():return parent.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source147=clean_json((ROOT/'trusted/C147_SOURCE_PINS.json').read_bytes()),binding147=previous.contract())
def design():
    return dict(protocol=PROTOCOL,training_registry=registry(),training_states=84,all84_are_exposed_training=True,initialization='Both arms start at original C146 fixed step12; routed experts are three identical copies. Both reset moments.',train_arms=list(TRAIN_ARMS),evaluation_arms=list(EVAL_ARMS),trainable_parameters=dict(global_continuation=36,context_routed=108),parameter_count_confound_explicit=True,router='u=V@(raw-prepared) from prewrite coordinates only; training84 population mean/std with floor1e-8; deterministic three prototypes: first registered row then farthest-first, exact ties use first registry row; eight fixed Lloyd iterations, empty clusters retain old prototype; softmax(-mean((z-center)^2)/2). No case, state ID, endpoint label or final failure membership enters routing.',router_role='Fixed training-side prewrite routing, not learned online addressing, an established geometric atlas or a class lookup. All valid new contexts can be routed; no excluded-input result is measured.',routing_parameters_trainable=False,coefficient_map='global a; routed a(z)=weights(z)@A with three bounded local vectors. No post-mixture clipping; convex weights preserve coefficient bound.',proposal='float32(raw+V.T@(a(z)*(V@(raw-prepared))))',offline_training_steps_per_arm=STEPS,training_objective=parent.design()['training_objective'],regularization='Same .001*mean(effective_a(z)^2) per state in both arms, not mean of all expert coefficients.',optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[.9,.999],epsilon=1e-8,full_parameter_gradient_L2_clip=1.,parameter_box=[-BOUND,BOUND]),objective_ties='Average exact ties within response coordinates and across84 states; finite weighted objective, no monotonicity or feasibility guarantee.',acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),training_safety=dict(response=.0018,read_margin=.075,KL=.0045),fixed_requests=dict(native_preflight=336,functional_zero=84,functional_warm=168,training_graph_forwards=2016,scalar_backwards=4032,native_final=672,total_model_forwards=6553),freeze='Both fixed step12 heads and all336 proposals sealed before two-round672 native endpoints. No arm/step/cluster/seed selection or deployment.',originals='Rehash original C147,C146,C145,C143 and rerun delivered C147 arithmetic audit before model load; other linked original ZIPs not reopened.',pass_rule='Report separate fixed-final qualification for each arm: all84 original native gates in both rounds and replay/repeat consistency. Any incomplete branch invalidates complete comparison. No preferred arm is selected.',worker_budget_seconds=TIMEOUT,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,derivative_path='Scalar AD to effective36 coefficient leaf; exact fixed linear routing pullback to108 parameters, not a model-response Jacobian. Effective coefficients committed with the same NumPy map in training and frozen inference.',scope='Offline two-arm architecture-and-capacity comparison on exposed84 training states only. A routed advantage cannot isolate routing from increased parameter count or optimizer geometry. No unseen/excluded generalisation, complete geometric atlas, true-gradient certification, convergence, eventual closure or model-internal control claim. Actual GPT2 CPU endpoints are required for production qualification.')

def coordinates(prepared,raw,V):
    farray(prepared,(DIM,));farray(raw,(DIM,));need(prepared.dtype==raw.dtype==np.float32,'Physical coordinates must be float32');need(V.ndim==2 and V.shape[1]==DIM,'Basis shape');farray(V,V.shape)
    return V@(raw.astype(float)-prepared.astype(float))

def fit_router(U):
    need(U.ndim==2 and U.shape[0]==N and U.shape[1]>0,'Exactly84 train-only projected contexts');farray(U,U.shape)
    mean=U.mean(0);scale=np.maximum(U.std(0),1e-8);Z=(U-mean)/scale;indices=[0];centers=[Z[0].copy()]
    for _ in range(1,K):
        dist=np.min(np.stack([np.mean((Z-c)**2,axis=1) for c in centers]),axis=0);dist[indices]=-1.;i=int(np.argmax(dist));indices.append(i);centers.append(Z[i].copy())
    centers=np.array(centers);empty=[]
    for iteration in range(LLOYD_STEPS):
        labels=np.argmin(np.mean((Z[:,None,:]-centers[None,:,:])**2,axis=2),axis=1);new=centers.copy()
        for k in range(K):
            if np.any(labels==k):new[k]=Z[labels==k].mean(0)
            else:empty.append(dict(iteration=iteration+1,cluster=k))
        centers=new
    router=dict(mean=mean,scale=scale,centers=centers);weights=np.array([route(u,router) for u in U]);meta=dict(seed_registry_indices=indices,lloyd_iterations=LLOYD_STEPS,empty_cluster_events=empty,training_hard_cluster_counts=np.bincount(np.argmax(weights,axis=1),minlength=K).tolist(),soft_effective_counts=weights.sum(0).tolist(),router_inputs_have_no_endpoint_labels=True)
    return router,weights,meta

def route(u,router):
    need(set(router)=={'mean','scale','centers'},'Exact router tensor inventory');farray(u,router['mean'].shape);farray(router['scale'],u.shape);farray(router['centers'],(K,len(u)));need(np.all(router['scale']>=1e-8),'Positive normalizer floor');farray(router['mean'],u.shape)
    z=(u-router['mean'])/router['scale'];logits=-.5*np.mean((z[None,:]-router['centers'])**2,axis=1);need(np.isfinite(logits).all(),'Nonfinite router distances');ex=np.exp(logits-logits.max());w=ex/ex.sum();need(np.isfinite(w).all() and np.all(w>=0) and abs(w.sum()-1)<1e-14,'Convex finite routing weights');return w

def effective(params,w,arm):
    need(arm in TRAIN_ARMS,'Unknown control arm')
    need(params.ndim==(1 if arm==TRAIN_ARMS[0] else 2),'Control dimensionality');farray(params,params.shape);need(np.max(abs(params),initial=0)<=BOUND+1e-15,'Bounded parameters required')
    if arm==TRAIN_ARMS[0]:return params.copy()
    need(params.shape[0]==K,'Three local controls');farray(w,(K,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14,'Convex weights');return w@params

def lift_gradient(g,w,arm):
    farray(g,(len(g),));return g.copy() if arm==TRAIN_ARMS[0] else w[:,None]*g[None,:]

def proposal(prepared,raw,V,a):return previous.proposal(prepared,raw,V,a)
def context_proposal(prepared,raw,V,params,router,arm):
    # Reusable prewrite map for arbitrary valid contexts; no state ID or class lookup.
    w=route(coordinates(prepared,raw,V),router);a=effective(params,w,arm);point,diag=proposal(prepared,raw,V,a)
    return point,dict(diag,routing_weights=w.tolist(),effective_control=a.tolist())

def loss(o,kl,point,prepared,a,case):return previous.loss(o,kl,point,prepared,a,case)
def aggregate(rows,shape):
    need([r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'Ordered full84 batch required')
    flat=[dict(r,response_gradient=np.asarray(r['response_gradient']).reshape(-1).tolist(),guard_gradient=np.asarray(r['guard_gradient']).reshape(-1).tolist()) for r in rows]
    g,agg=previous.aggregate(flat,int(np.prod(shape)));return g.reshape(shape),agg

def update(a,g,m,v,step):
    need(type(step) is int and 1<=step<=STEPS,'Fixed12 only')
    for x in (a,g,m,v):farray(x,a.shape)
    need(np.all(v>=0),'Nonnegative Adam second moment');norm=float(np.linalg.norm(g));factor=min(1.,1./max(norm,1e-30));gg=g*factor;mm=.9*m+.1*gg;vv=.999*v+.001*gg*gg;unrestricted=a-RATE*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.clip(unrestricted,-BOUND,BOUND)
    return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.count_nonzero(new!=unrestricted)),update_Linf=float(np.max(abs(new-a),initial=0)))

def fixed_digest(d):
    tensors={sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()};tensors.update(V=d['V'],initial_C146=d['initial_a'],initial_C143=d['a']);tensors.update({'router_'+k:v for k,v in d['router'].items()});tensors.update({'weights_'+sid:w for sid,w in d['weights'].items()})
    return dict(inputs_and_router=safe.head_digest(tensors),writer=safe.head_digest(d['writer']),control=safe.head_digest(d['control']))

def final_tables(records):
    sums={};pairs={}
    for group,n in GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];lookup={(r['state_id'],r['arm']):r for r in rr};need(len(rr)==len(lookup)==n*4,'All four arms and all states retained');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in EVAL_ARMS:
                ar=[r for r in rr if r['arm']==arm];failed=Counter(k for r in ar for k in r['gates']['failed_gates']);sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(failed));pairs[group][str(rnd)][arm]={}
                for ref in EVAL_ARMS:
                    pairs[group][str(rnd)][arm][ref]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not lookup[(r['state_id'],ref)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and lookup[(r['state_id'],ref)]['gates']['complete_pass']])
    return sums,pairs

class Phase:
    def __init__(self,d):
        self.d=d;self.phase='preflight';self.active=None;self.used=set();self.zero=set();self.warm={arm:set() for arm in TRAIN_ARMS};self.batch=None;self.seen=set();self.completed={arm:0 for arm in TRAIN_ARMS};self.preflight={};self.paths={}
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind in ('prepared','raw','C143','C146'):
                point=p[kind] if kind in ('prepared','raw') else proposal(p['prepared'],p['raw'],d['V'],d['a'] if kind=='C143' else d['initial_a'])[0];self.preflight[(sid,kind)]=(safe.tensor_hash(point),sp['case'],kind!='prepared')
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.paths;tag=(self.phase,key);want=(safe.tensor_hash(point),case,write);need(self.phase in ('preflight','paths') and self.active is None and table.get(key)==want and tag not in self.used,'Unregistered/duplicate native measurement');self.used.add(tag);self.active=want
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native point');self.active=None
    def begin_zero(self):need(self.phase=='preflight' and self.active is None and len(self.used)==336,'All preflights before graphs');self.phase='functional_zero'
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==N,'All84 zero parities');self.phase='functional_warm'
    def functional(self,sid,params,arm,point):
        need(sid in self.d['points'] and arm in TRAIN_ARMS,'Registered training state and arm');p=self.d['points'][sid];a=effective(params,self.d['weights'][sid],arm);expected=proposal(p['prepared'],p['raw'],self.d['V'],a)[0]
        if self.phase=='functional_zero':need(arm==TRAIN_ARMS[0] and np.all(params==0) and sid not in self.zero and safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Exact zero raw parity');self.zero.add(sid)
        elif self.phase=='functional_warm':
            initial=self.d['initial_a'] if arm==TRAIN_ARMS[0] else np.tile(self.d['initial_a'],(K,1));need(row_digest(params)==row_digest(initial) and sid not in self.warm[arm],'Exact initial head warm parity');self.warm[arm].add(sid)
        else:need(self.phase=='training' and self.batch==(arm,row_digest(params)) and sid not in self.seen,'Wrong training head or duplicate state');self.seen.add(sid)
        return close(point,expected,'NumPy/Torch point parity',2e-6)
    def begin_training(self):need(self.phase=='functional_warm' and all(len(s)==N for s in self.warm.values()),'Both all84 warm parities');self.phase='training'
    def begin_batch(self,step,params,arm):need(self.phase=='training' and self.batch is None and type(step) is int and step==self.completed[arm] and step<STEPS,'Fixed full84 steps only');self.batch=(arm,row_digest(params));self.seen=set()
    def end_batch(self):need(self.batch is not None and len(self.seen)==N,'All84 gradients retained');self.completed[self.batch[0]]+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and all(n==STEPS for n in self.completed.values()),'Both fixed final steps before qualification');need(set(points)=={(sp['state_id'],arm) for sp in registry() for arm in EVAL_ARMS},'All336 proposals');self.paths={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.used)==336+672,'Complete native denominator')
