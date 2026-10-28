"""Fixed routed diagonal versus routed diagonal-plus-cyclic coupling; training only."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor149'))
import core149 as auditcore
from core149 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
old=auditcore.parent
PROTOCOL='GPT2_LC2_C150_FIXED_CYCLIC_COUPLING_CPU_V1'
INPUT_KEYS=('C149','C148','C147','C146','C145','C143');N=84;K=3;STEPS=12;TIMEOUT=14400
TRAIN_ARMS=('routed_diagonal_continuation','routed_cyclic_coupling')
EVAL_ARMS=('raw_control','C148_frozen_routed_step12')+TRAIN_ARMS
GROUPS=old.GROUPS
RATE=.0005;BOUND=.125

def registry():return old.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source149=clean_json((ROOT/'trusted/C149_SOURCE_PINS.json').read_bytes()),binding149=auditcore.contract())
def design():return dict(protocol=PROTOCOL,training_registry=registry(),training_states=84,all84_are_exposed_training=True,initialization='Both arms use exact original C148 routed final step12; cyclic off-diagonal coefficients start at zero; reset Adam moments.',train_arms=list(TRAIN_ARMS),evaluation_arms=list(EVAL_ARMS),trainable_parameters=dict(routed_diagonal_continuation=108,routed_cyclic_coupling=216),fixed_change='Add exactly one cyclic coupling pattern roll(u,1); do not search permutations, temperatures, charts, seeds or checkpoints. Same frozen V, original writer, router, all84 data, objective and schedule.',router='Original C148 fixed three-prototype router and normalizer reused exactly; only prewrite coordinates/projected raw proposal enter routing.',effective_control='Diagonal e=w@A, shape36. Coupled e=stack(w@A_diag,w@A_cross), shape2x36.',proposal='u=V@(raw-prepared); s=e*u (diagonal) or e[0]*u+e[1]*roll(u,1) (coupled); point=float32(raw+V.T@s). Cyclic wrapping is fixed in the archived V row order, not a certified geometry or optimal coupling.',regularizer='.001*sum(e**2)/36 in either arm; identical to original diagonal regularizer when cross block is zero. Regularization normalization changes with rank only in synthetic tests.',objective='R_i=relu(max(abs(response_i))/.0018-1)^2; G_i=read_hinge(.075)+KL_hinge(.0045)+radius_hinge(20)+regularizer. J=.9*max_i R_i+.1*mean_i R_i+mean_i G_i; exact maxima average ties.',optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[.9,.999],epsilon=1e-8,full_parameter_gradient_L2_clip=1.,parameter_box=[-BOUND,BOUND]),offline_training_steps_per_arm=12,final_checkpoint='Only fixed step12 in each arm; no arm selection or deployment.',acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),training_safety=dict(response=.0018,read_margin=.075,KL=.0045),fixed_requests=dict(native_preflight=336,functional_zero=84,functional_warm=168,training_graph_forwards=2016,scalar_backwards=4032,native_final=672,total_model_forwards=6553),preflight='Reopen six pinned originals, reuse the sealed C149 prior audit receipt, independently check all84 origin rows, actual tensors and336 archived points/672 native gates, then replay all84 prepared/raw/C146/C148 endpoints; all zero and both warm functional comparisons before any scalar backward.',freeze='Both fixed step12 heads and all336 points sealed before two-round672 native endpoints.',pass_rule='Each arm independently requires84/84 original native gates in both rounds, plus complete denominators and replay/repeat consistency. Partial comparison qualifies neither arm.',hypothesis_scope='Cross-coordinate coupling adds expressivity inside the same V span; it does not establish that diagonal control is infeasible, a causal failure explanation, an atlas, optimal pairing or excluded-input generalisation. Parameters and optimizer geometry differ between arms.',derivative_path='Scalar AD to state-effective control leaf (36 or2x36), then fixed linear router pullback; no full response Jacobian.',worker_budget_seconds=TIMEOUT,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False,scope='Offline two-arm exposed84 training qualification only. C149 diagnostics motivate a declared structural test, not a proven failure cause. No independent derivative certification, convergence or closure guarantee. Even all84 training passes require new prospective confirmation and strictly isolated excluded-input evaluation.')

def initial_parameters(d,arm):
    need(arm in TRAIN_ARMS,'Registered arm');A=d['initial_A'];farray(A,(3,A.shape[1]));return A.copy() if arm==TRAIN_ARMS[0] else np.stack([A,np.zeros_like(A)],axis=1)

def effective(p,w,arm):
    need(arm in TRAIN_ARMS and p.ndim==(2 if arm==TRAIN_ARMS[0] else 3),'Control dimensions');need(p.shape[0]==3,'Exactly three experts');farray(p,p.shape);farray(w,(3,));need(np.max(abs(p),initial=0)<=BOUND+1e-15 and np.all(w>=0) and abs(w.sum()-1)<1e-14,'Coefficient box and convex routing')
    if arm==TRAIN_ARMS[0]:return w@p
    need(p.shape[1]==2,'Diagonal and one fixed cross block');return np.stack([w@p[:,j,:] for j in range(2)])

def pullback(g,w,arm):
    need(arm in TRAIN_ARMS,'Registered arm');farray(g,g.shape);return w[:,None]*g[None,:] if arm==TRAIN_ARMS[0] else w[:,None,None]*g[None,:,:]

def context(prepared,raw,V):return auditcore.projected_context(prepared,raw,V)
def proposal(prepared,raw,V,e):
    u=context(prepared,raw,V);need(e.shape in ((len(u),),(2,len(u))),'Effective control shape');farray(e,e.shape);need(np.max(abs(e),initial=0)<=BOUND+1e-15,'Effective coefficient bound');s=e*u if e.ndim==1 else e[0]*u+e[1]*np.roll(u,1);point=(raw.astype(float)+V.T@s).astype(np.float32);return point,dict(correction_L2=float(np.linalg.norm(V.T@s)),actual_delta_norm=float(np.linalg.norm(point.astype(float)-prepared.astype(float))),radial_clipping_performed=False)

def context_proposal(prepared,raw,V,p,router,arm):
    w=auditcore.route(context(prepared,raw,V),router);e=effective(p,w,arm);point,diag=proposal(prepared,raw,V,e);return point,dict(diag,routing_weights=w.tolist(),effective_control=e.tolist())

def loss(o,kl,point,prepared,e,case):
    rank=e.shape[-1];need(e.shape in ((rank,),(2,rank)),'State-effective control shape');r=auditcore.loss(o,kl,point,prepared,e if e.ndim==1 else e[0],case);r['regularizer']=float(.001*np.sum(e*e)/rank);return r

def aggregate(rows,shape):return auditcore.aggregate(rows,shape)
def update(p,g,m,v,step):return auditcore.update(p,g,m,v,step)

def fixed_digest(d):
    tensors={sid+'__'+k:v for sid,p in d['points'].items() for k,v in p.items()};tensors.update(V=d['V'],initial_C146=d['initial_a'],initial_C148_routed=d['initial_A']);tensors.update({'router_'+k:v for k,v in d['router'].items()});tensors.update({'weights_'+sid:w for sid,w in d['weights'].items()});return dict(inputs_and_router=safe.head_digest(tensors),writer=safe.head_digest(d['writer']),control=safe.head_digest(d['control']))

def final_tables(records):
    sums={};pairs={}
    for group,n in GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];lookup={(r['state_id'],r['arm']):r for r in rr};need(len(rr)==len(lookup)==4*n,'All native states and arms retained');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in EVAL_ARMS:
                ar=[r for r in rr if r['arm']==arm];bad=Counter(k for r in ar for k in r['gates']['failed_gates']);sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(bad));pairs[group][str(rnd)][arm]={}
                for ref in EVAL_ARMS:pairs[group][str(rnd)][arm][ref]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not lookup[(r['state_id'],ref)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and lookup[(r['state_id'],ref)]['gates']['complete_pass']])
    return sums,pairs

def preflight_point(d,sid,kind):
    p=d['points'][sid]
    if kind in ('prepared','raw'):return p[kind]
    if kind=='C146':return old.proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])[0]
    need(kind=='C148','Registered preflight baseline');return proposal(p['prepared'],p['raw'],d['V'],effective(d['initial_A'],d['weights'][sid],TRAIN_ARMS[0]))[0]

class Phase:
    def __init__(self,d):
        self.d=d;self.phase='preflight';self.active=None;self.used=set();self.zero=set();self.warm={a:set() for a in TRAIN_ARMS};self.batch=None;self.seen=set();self.completed={a:0 for a in TRAIN_ARMS};self.preflight={};self.paths={}
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind in ('prepared','raw','C146','C148'):
                point=preflight_point(d,sid,kind);self.preflight[(sid,kind)]=(safe.tensor_hash(point),sp['case'],kind!='prepared')
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.paths;tag=(self.phase,key);want=(safe.tensor_hash(point),case,write);need(self.phase in ('preflight','paths') and self.active is None and table.get(key)==want and tag not in self.used,'Unregistered/duplicate native point');self.active=want;self.used.add(tag)
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native point');self.active=None
    def begin_zero(self):need(self.phase=='preflight' and self.active is None and len(self.used)==336,'All336 preflights first');self.phase='functional_zero'
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==84,'All84 zero parities first');self.phase='functional_warm'
    def functional(self,sid,p,arm,point):
        need(sid in self.d['points'] and arm in TRAIN_ARMS,'Registered state and arm');q=self.d['points'][sid];expected=proposal(q['prepared'],q['raw'],self.d['V'],effective(p,self.d['weights'][sid],arm))[0]
        if self.phase=='functional_zero':need(arm==TRAIN_ARMS[0] and np.all(p==0) and sid not in self.zero and safe.tensor_hash(point)==safe.tensor_hash(q['raw']),'Exact raw zero parity');self.zero.add(sid)
        elif self.phase=='functional_warm':need(row_digest(p)==row_digest(initial_parameters(self.d,arm)) and sid not in self.warm[arm],'Exact common initialization');self.warm[arm].add(sid)
        else:need(self.phase=='training' and self.batch==(arm,row_digest(p)) and sid not in self.seen,'Registered batch/state');self.seen.add(sid)
        return close(point,expected,'NumPy/Torch point parity',2e-6)
    def begin_training(self):need(self.phase=='functional_warm' and all(len(v)==84 for v in self.warm.values()),'Both84 warm parities');self.phase='training'
    def begin_batch(self,step,p,arm):need(self.phase=='training' and self.batch is None and type(step) is int and step==self.completed[arm] and step<12,'Fixed12 per arm');self.batch=(arm,row_digest(p));self.seen=set()
    def end_batch(self):need(self.batch is not None and len(self.seen)==84,'All84 gradients');self.completed[self.batch[0]]+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and all(v==12 for v in self.completed.values()),'Both fixed final steps first');need(set(points)=={(sp['state_id'],arm) for sp in registry() for arm in EVAL_ARMS},'All336 sealed points');self.paths={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.used)==336+672,'Complete final denominator')
