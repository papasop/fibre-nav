"""Shared bounded diagonal control, trained on fixed finite endpoints."""
from pathlib import Path
import hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor136'));sys.path.insert(0,str(ROOT/'vendor136/vendor135'))
from core135 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,ARMS,DIM,RADIUS,BUDGET,registry,gate_record

PROTOCOL='GPT2_LC2_C137_WORST_STATE_TRAINING_CPU_V1'
INPUT_KEYS=('C132','C133','C134','C135','C136');STEPS=24;TIMEOUT=3000
RATE=.001;ADAM_B1=.9;ADAM_B2=.999;ADAM_EPS=1e-8;GRAD_CLIP=1.;GAIN_BOUND=.125
RESPONSE_SAFETY=.0015;MARGIN_SAFETY=.075;KL_SAFETY=.0045;L2=.001
EVAL_ARMS=('raw_control','shared_vector_step24')

TEMPERATURE=8.; UNIFORM_FRACTION=.1

def contract():
    from core136 import contract as prior_contract
    c=prior_contract();c.update(pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source136=clean_json((ROOT/'trusted/C136_SOURCE_PINS.json').read_bytes()));return c

def design():
    return dict(protocol=PROTOCOL,training_registry=registry(),training_states=12,independent_confirmation_states=0,
        all12_are_training='All twelve historical states are training; original roles are provenance only.',
        formula='z=V@(raw-prepared); point=float32(raw + V.T@(a*z)); one shared36 vector',
        initialization='Exact fixed C136 step16 vector; reset both Adam moments to zero. A new registered experiment, not extension or reclassification of C136.',
        parent_decision_retained='C136_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE',
        production_trainable_dimension=36,elementwise_a_bound=GAIN_BOUND,
        state_loss='Same per-state response mean+max hinge-square, read-margin hinge-square, KL, radius and .001*mean(a^2) as C136',
        aggregate_objective='0.1*mean(L_i) + 0.9/8*log(mean(exp(8*L_i))); stable log-sum-exp; all12 evaluated at same a before any update',
        aggregate_gradient='sum_i w_i*dL_i/da; w_i=.1/12+.9*softmax(8*L)_i. Each unweighted per-state scalar gradient retained; weights derived from all12 current training losses, no state-specific parameters.',
        temperature=TEMPERATURE,uniform_fraction=UNIFORM_FRACTION,
        offline_training_steps=STEPS,states_per_full_gradient=12,scalar_backward_calls_requested=STEPS*12,
        optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[ADAM_B1,ADAM_B2],epsilon=ADAM_EPS,full_gradient_L2_clip=GRAD_CLIP,elementwise_projection=[-GAIN_BOUND,GAIN_BOUND],initial_moments='zero'),
        training_response_safety=RESPONSE_SAFETY,training_margin_safety=MARGIN_SAFETY,training_KL_safety=KL_SAFETY,
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        radius_policy='No radial clipping; actual displacement gate remains unchanged and any excess is penalised and retained.',
        preflight='48 native points: twelve preparations, raw, old projected and C136 step16 endpoints; archived point and gate identity required, observations2e-4 KL5e-6',
        functional_parity='All12 zero controls versus native raw, then all12 C136 initial controls versus fresh native warm endpoints; observations2e-6 KL2e-7, before any backward',
        training_mechanism='Temporary graph-connected LoRA-B replacements; only a receives gradients. Frozen backbone/A/base writer. AD on float64 construction to float32 commit is not independently certified finite-precision differentiation.',
        final_checkpoint='Only new C137 update24; no early stopping, best/recovery choice, per-state heads or input-class selection',
        final_evaluation='Seal all24 raw/final points and vector before two-round48 native training endpoints; paired reporting against raw and archived C136 step16',
        input_audit='Rehash five original ZIPs; delivered trusted source only; C136 archived updates/objectives/points/gates independently recomputed arithmetically, no archived code execution or independent derivative claim',
        requested_model_forwards=817,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher=False,backtracking=False,online_response_repair=False,
        candidate_selected=False,writer_deployed=False,excluded_input_states_evaluated=0,LC2_closed=False,worker_budget_seconds=TIMEOUT)

def aggregate(losses,gradients):
    losses=farray(np.asarray(losses,float),(12,));gradients=np.asarray(gradients,float)
    need(gradients.ndim==2 and gradients.shape[0]==12 and gradients.shape[1]>0 and np.isfinite(gradients).all(),'All12 finite unweighted gradient rows required')
    need(np.all(losses>=0),'Negative state loss');shift=losses.max();e=np.exp(TEMPERATURE*(losses-shift));soft=e/e.sum();weights=UNIFORM_FRACTION/12+(1-UNIFORM_FRACTION)*soft
    value=UNIFORM_FRACTION*losses.mean()+(1-UNIFORM_FRACTION)*(shift+np.log(e.mean())/TEMPERATURE)
    g=weights@gradients
    need(np.isfinite(value) and np.isfinite(g).all(),'Nonfinite aggregate')
    return g,dict(objective=float(value),weights=weights.tolist(),state_losses=losses.tolist(),temperature=TEMPERATURE,uniform_fraction=UNIFORM_FRACTION,maximum_state_loss=float(shift),mean_state_loss=float(losses.mean()))


def proposal(prepared,raw,v,a):
    farray(prepared,(DIM,));farray(raw,(DIM,));farray(a,(len(v),));need(prepared.dtype==raw.dtype==np.float32,'Float32 archived points required')
    need(np.isfinite(v).all() and v.ndim==2 and v.shape[1]==DIM and np.max(abs(a),initial=0)<=GAIN_BOUND+1e-15,'Invalid shared vector or row basis')
    d=raw.astype(float)-prepared.astype(float);correction=v.T@(a*(v@d));point=(raw.astype(float)+correction).astype(np.float32)
    return point,dict(correction_L2=float(np.linalg.norm(correction)),actual_delta_norm=float(np.linalg.norm(point.astype(float)-prepared.astype(float))),float32_commit_error_Linf=float(np.max(abs(point.astype(float)-raw.astype(float)-correction))),radial_clipping_performed=False)

def objective(observed,kl,point,prepared,a,case):
    o=farray(np.asarray(observed,float),(22,));need(np.isfinite(kl) and kl>=0,'Invalid KL')
    rv=np.maximum(abs(o[:6])/RESPONSE_SAFETY-1,0)**2
    signs=2*np.array(safe.target_bits(case,True)*4)-1;read=np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)**2
    return dict(response_loss=float(rv.mean()+rv.max()),read_loss=float(read.mean()),KL_loss=float(max(kl/KL_SAFETY-1,0)**2),radius_loss=float(max(np.linalg.norm(point.astype(float)-prepared.astype(float))/RADIUS-1,0)**2),regularizer=float(L2*np.mean(a*a)))

def adam_step(a,gradient,m,v,step):
    need(type(step) is int and 1<=step<=STEPS,'Unregistered optimizer step')
    for x in (a,gradient,m,v):farray(x,a.shape)
    need(np.all(v>=0),'Negative Adam second moment');norm=float(np.linalg.norm(gradient));factor=min(1.,GRAD_CLIP/max(norm,1e-30));g=gradient*factor
    mm=ADAM_B1*m+(1-ADAM_B1)*g;vv=ADAM_B2*v+(1-ADAM_B2)*g*g
    change=RATE*(mm/(1-ADAM_B1**step))/(np.sqrt(vv/(1-ADAM_B2**step))+ADAM_EPS);unprojected=a-change;new=np.clip(unprojected,-GAIN_BOUND,GAIN_BOUND)
    need(np.isfinite(new).all(),'Nonfinite Adam update')
    return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.sum(new!=unprojected)),update_Linf=float(np.max(abs(new-a),initial=0)))

class Phase:
    def __init__(self,data):
        self.data=data;self.phase='preflight';self.preflight={};self.active=None;self.taken=set();self.zero=set();self.warm=set();self.batch=None;self.completed_batches=0;self.training_seen=set();self.final_points={}
        for spec in registry():
            sid=spec['state_id'];case=spec['case'];self.preflight[('prepared',sid)]=(safe.tensor_hash(data['points'][sid]['prepared']),case,False)
            for arm in ARMS[:2]:self.preflight[('endpoint',sid,arm)]=(safe.tensor_hash(data['maps'][(sid,arm)]['point']),case,True)
        for spec in registry():
            sid=spec['state_id'];point=proposal(data['points'][sid]['prepared'],data['points'][sid]['raw'],data['V'],data['initial_a'])[0];self.preflight[('warm',sid)]=(safe.tensor_hash(point),spec['case'],True)
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.final_points
        need(self.phase in ('preflight','paths') and self.active is None,'Native metric phase forbidden');tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(key in table and table[key]==expected and tag not in self.taken,'Unsealed or repeated native metric request');self.taken.add(tag);self.active=expected
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native model point');self.active=None
    def begin_zero(self):
        need(self.phase=='preflight' and self.active is None and len(self.taken)==48,'All48 preflight calls required');self.phase='functional_zero'
    def functional(self,sid,a,point):
        need(sid in self.data['points'] and sid in {r['state_id'] for r in registry()},'Functional access outside fixed training registry')
        p=self.data['points'][sid];expected,_=proposal(p['prepared'],p['raw'],self.data['V'],a)
        if self.phase=='functional_zero':
            need(np.all(a==0) and sid not in self.zero,'Only one zero-control forward per training state');need(safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Zero control must retain exact raw bytes');self.zero.add(sid)
        elif self.phase=='functional_warm':
            need(row_digest(a)==row_digest(self.data['initial_a']) and sid not in self.warm,'Only one pinned warm functional point per state');self.warm.add(sid)
        else:
            need(self.phase=='training' and self.batch is not None and row_digest(a)==self.batch['digest'] and sid not in self.training_seen,'Wrong/duplicate training vector or state');self.training_seen.add(sid)
        return close(point,expected,'NumPy/functional proposal parity',2e-6)
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==12,'Complete functional zero parity required');self.phase='functional_warm'
    def begin_training(self):need(self.phase=='functional_warm' and len(self.warm)==12,'Complete warm functional parity required');self.phase='training'
    def begin_batch(self,step,a):
        need(self.phase=='training' and self.batch is None and step==self.completed_batches and step<STEPS,'Fixed fullbatch schedule required');self.batch=dict(step=step,digest=row_digest(a));self.training_seen=set()
    def end_batch(self):
        need(self.batch is not None and len(self.training_seen)==12,'All12 gradient contributions required');self.completed_batches+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and self.completed_batches==STEPS,'Only fixed step24 may freeze')
        need(set(points)=={(r['state_id'],arm) for r in registry() for arm in EVAL_ARMS},'All24 final outcomes required')
        self.final_points={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.taken)==48+48,'All48 final endpoint records required')
