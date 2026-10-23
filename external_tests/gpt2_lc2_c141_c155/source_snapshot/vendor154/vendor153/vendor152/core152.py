"""Fixed-rate comparison: same routed cyclic head, same84 states, two registered rates."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor151'))
import core151 as auditcore
previous=auditcore.parent
from core151 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C152_FIXED_SMALLSTEP_COUPLING_CPU_V1'
INPUT_KEYS=auditcore.INPUT_KEYS;N=84;K=3;STEPS=12;TIMEOUT=14400
TRAIN_ARMS=('cyclic_rate_5e4','cyclic_rate_1e4')
RATES={TRAIN_ARMS[0]:.0005,TRAIN_ARMS[1]:.0001}
EVAL_ARMS=('raw_control','C150_frozen_cyclic_step12')+TRAIN_ARMS
GROUPS=previous.GROUPS;BOUND=.125

def registry():return previous.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),audit151=auditcore.contract(),source151=clean_json((ROOT/'trusted/C151_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,training_registry=registry(),training_states=84,all84_are_exposed_training=True,initialization='Both arms exactly copy original C150 cyclic final step12 including nonzero cross block; all Adam moments reset to zero in both arms.',train_arms=list(TRAIN_ARMS),evaluation_arms=list(EVAL_ARMS),trainable_parameters={a:216 for a in TRAIN_ARMS},fixed_difference='Learning rate only: .0005 versus .0001; common head, data, loss, batch order, router, V, cyclic offset, gradient clip and coefficient box.',learning_rates=RATES,rate_choice='Registered research choice motivated only by exposed C150/C151 training diagnostics, not an established failure cause or guaranteed improvement. No rate grid, online line search, adaptive schedule or checkpoint selection.',head='Each arm has3 experts x2 coefficient blocks x36 coordinates; both diagonal and cross blocks remain trainable.',router='Original C148 frozen three-prototype prewrite router and normalizer reused; no new route fitting or training.',effective_control='Explicit left-associated float64 sum (w0*P0+w1*P1)+w2*P2, shape2x36; same computation in both arms.',proposal='u=V@(raw-prepared); s=e[0]*u+e[1]*roll(u,1); point=float32(raw+V.T@s). Fixed positive cyclic offset1, no pairing or chart search.',regularizer='.001*sum(e**2)/36; same as C150 cyclic head.',objective='R_i=relu(max(abs(response_i))/.0018-1)^2; G_i=read_hinge(.075)+KL_hinge(.0045)+radius_hinge(20)+regularizer. J=.9*max_i R_i+.1*mean_i R_i+mean_i G_i; average exact maximizing ties.',optimiser=dict(kind='manual float64 projected Adam',learning_rates=RATES,betas=[.9,.999],epsilon=1e-8,full_parameter_gradient_L2_clip=1.,parameter_box=[-.125,.125],moments_reset_for_both=True),offline_training_steps_per_arm=12,final_checkpoint='Only fixed step12 for each arm; both retained, no arm/deployment selection.',acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),training_safety=dict(response=.0018,read_margin=.075,KL=.0045),fixed_requests=dict(native_preflight=336,functional_zero=84,functional_warm=168,training_graph_forwards=2016,scalar_backwards=4032,native_final=672,total_model_forwards=6553),preflight='Rehash seven original archives and independently rerun trusted C151 C150 arithmetic. Bind exact final coupled head, then replay all84 prepared/raw/C148/C150 native points. All84 zero and both84 common-warm functional parities before scalar backward.',freeze='Both fixed step12 heads and all336 points sealed before two-round672 native endpoints.',pass_rule='Each arm requires84/84 original gates in both complete rounds plus replay consistency; incomplete comparison qualifies neither arm.',scope='Offline training-domain rate comparison only; all84 states remain exposed training. No model-independent true-gradient certification, failure-cause proof, convergence, architecture/rate superiority or excluded-input generalisation. Improvement versus frozen C150 also includes twelve additional training steps; the matched continuation arms isolate the registered rate difference within this run. New prospective confirmation and strict excluded-input evaluation remain required after training qualification.',C151_original_result_zip_required=False,prior_C149_recursive_audit_reexecuted=False,worker_budget_seconds=TIMEOUT,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,LC2_single_fold_passed=False)

def initial_parameters(d,arm):
    need(arm in TRAIN_ARMS,'Registered rate arm');p=d['initial_P'];farray(p,(3,2,p.shape[-1]));need(np.max(abs(p),initial=0)<=BOUND+1e-15,'Common initial head box');return p.copy()

def effective(p,w,arm):
    need(arm in TRAIN_ARMS,'Registered rate arm');farray(p,(3,2,p.shape[-1]));farray(w,(3,));need(np.max(abs(p),initial=0)<=BOUND+1e-15 and np.all(w>=0) and abs(w.sum()-1)<1e-14,'Bounded head/convex fixed weights');return (w[0]*p[0]+w[1]*p[1])+w[2]*p[2]

def pullback(g,w,arm):
    need(arm in TRAIN_ARMS,'Registered rate arm');farray(g,(2,g.shape[-1]));farray(w,(3,));return w[:,None,None]*g[None,:,:]

def context(prepared,raw,V):return previous.context(prepared,raw,V)
def proposal(prepared,raw,V,e):
    farray(e,(2,V.shape[0]));farray(prepared,(DIM,));farray(raw,(DIM,));farray(V,(V.shape[0],DIM));need(prepared.dtype==raw.dtype==np.float32 and np.max(abs(e),initial=0)<=BOUND+1e-15,'Physical dtype/effective coefficient bound');u=context(prepared,raw,V);shift=V.T@(e[0]*u+e[1]*np.roll(u,1));p=(raw.astype(float)+shift).astype(np.float32);return p,dict(correction_L2=float(np.linalg.norm(shift)),actual_delta_norm=float(np.linalg.norm(p.astype(float)-prepared.astype(float))),radial_clipping_performed=False)

def context_proposal(prepared,raw,V,p,router,arm):
    w=previous.auditcore.route(context(prepared,raw,V),router);e=effective(p,w,arm);point,diag=proposal(prepared,raw,V,e);return point,dict(diag,routing_weights=w.tolist(),effective_control=e.tolist())

def loss(o,kl,point,prepared,e,case):return auditcore.loss(o,kl,point,prepared,e,case)
def aggregate(rows,shape):return auditcore.aggregate(rows,shape)
def update(p,g,m,v,step,arm):
    need(arm in TRAIN_ARMS and type(step) is int and 1<=step<=STEPS,'Fixed arm and12-step schedule')
    for x in (p,g,m,v):farray(x,p.shape)
    need(p.shape[0:2]==(3,2) and np.all(v>=0),'Coupled shape/nonnegative Adam second moment');norm=float(np.linalg.norm(g));factor=min(1.,1/max(norm,1e-30));gg=g*factor;mm=.9*m+.1*gg;vv=.999*v+.001*gg*gg;unrestricted=p-RATES[arm]*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8);new=np.minimum(BOUND,np.maximum(-BOUND,unrestricted));return new,mm,vv,dict(step=step,learning_rate=RATES[arm],gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.sum(new!=unrestricted)),update_Linf=float(np.max(abs(new-p),initial=0)))

def fixed_digest(d):
    tensors={sid+'__'+k:v for sid,q in d['points'].items() for k,v in q.items()};tensors.update(V=d['V'],initial_C148=d['initial_A'],initial_C150_cyclic=d['initial_P']);tensors.update({'router_'+k:v for k,v in d['router'].items()});tensors.update({'weights_'+sid:w for sid,w in d['weights'].items()});return dict(inputs_and_router=safe.head_digest(tensors),writer=safe.head_digest(d['writer']),control=safe.head_digest(d['control']))

def final_tables(records):
    sums={};pairs={}
    for group,n in GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];by={(r['state_id'],r['arm']):r for r in rr};need(len(rr)==len(by)==4*n,'Complete native denominator');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in EVAL_ARMS:
                ar=[r for r in rr if r['arm']==arm];bad=Counter(k for r in ar for k in r['gates']['failed_gates']);sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=.0018 for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(bad));pairs[group][str(rnd)][arm]={}
                for ref in EVAL_ARMS:pairs[group][str(rnd)][arm][ref]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not by[(r['state_id'],ref)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and by[(r['state_id'],ref)]['gates']['complete_pass']])
    return sums,pairs

def preflight_point(d,sid,kind):
    q=d['points'][sid]
    if kind in ('prepared','raw'):return q[kind]
    if kind=='C148':return previous.proposal(q['prepared'],q['raw'],d['V'],previous.effective(d['initial_A'],d['weights'][sid],previous.TRAIN_ARMS[0]))[0]
    need(kind=='C150','Registered baseline');return proposal(q['prepared'],q['raw'],d['V'],effective(d['initial_P'],d['weights'][sid],TRAIN_ARMS[0]))[0]

class Phase:
    def __init__(self,d):
        self.d=d;self.phase='preflight';self.active=None;self.used=set();self.zero=set();self.warm={a:set() for a in TRAIN_ARMS};self.batch=None;self.seen=set();self.completed={a:0 for a in TRAIN_ARMS};self.preflight={};self.paths={}
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind in ('prepared','raw','C148','C150'):
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
