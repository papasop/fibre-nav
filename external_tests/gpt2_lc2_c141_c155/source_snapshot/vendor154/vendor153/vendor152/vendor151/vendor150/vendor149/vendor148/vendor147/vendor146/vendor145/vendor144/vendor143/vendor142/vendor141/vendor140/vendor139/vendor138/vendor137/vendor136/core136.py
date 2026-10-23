"""Shared bounded diagonal control, trained on fixed finite endpoints."""
from pathlib import Path
import hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor135'))
from core135 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,ARMS,DIM,RADIUS,BUDGET,registry,gate_record

PROTOCOL='GPT2_LC2_C136_SHARED_VECTOR_TRAINING_CPU_V1'
INPUT_KEYS=('C132','C133','C134','C135');STEPS=16;TIMEOUT=3000
RATE=.003;ADAM_B1=.9;ADAM_B2=.999;ADAM_EPS=1e-8;GRAD_CLIP=1.;GAIN_BOUND=.125
RESPONSE_SAFETY=.0015;MARGIN_SAFETY=.075;KL_SAFETY=.0045;L2=.001
EVAL_ARMS=('raw_control','shared_vector_step16')

def contract():
    return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),
        source132=clean_json((ROOT/'vendor135/vendor134/trusted/C132_SOURCE_PINS.json').read_bytes()),
        source133=clean_json((ROOT/'vendor135/vendor134/trusted/C133_SOURCE_PINS.json').read_bytes()),
        source134=clean_json((ROOT/'vendor135/trusted/C134_SOURCE_PINS.json').read_bytes()),
        source135=clean_json((ROOT/'trusted/C135_SOURCE_PINS.json').read_bytes()))

def design():
    return dict(protocol=PROTOCOL,training_registry=registry(),training_states=12,independent_confirmation_states=0,
        all12_are_training='Both original geometry6 and historical-probe6 are C136 training data. Original roles are provenance only.',
        formula='z=V@(raw-prepared); point=float32(raw + V.T@(a*z)); one shared a across all12 states, initialized exactly zero',
        gain_interpretation='Equivalent in exact arithmetic to P*d + V.T@diag(1+a)@V*d. Fixed C132 numeric row basis; no new SVD/Jacobian.',
        production_trainable_dimension=36,elementwise_a_bound=GAIN_BOUND,
        radius_policy='No radial proposal clipping: preserve exact zero-control identity and graph connection. Original actual displacement gate<=20+1e-5 remains unchanged; radius excess is penalised during training and retained as failure in final evaluation.',
        offline_training_steps=STEPS,states_per_full_gradient=12,scalar_backward_calls_requested=192,
        optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[ADAM_B1,ADAM_B2],epsilon=ADAM_EPS,full_gradient_L2_clip=GRAD_CLIP,elementwise_projection=[-GAIN_BOUND,GAIN_BOUND],weight_decay=0),
        loss='Mean over all12 of response hinge-square mean+max, read-margin hinge-square mean, KL hinge-square, displacement-radius hinge-square and .001*mean(a^2)',
        training_response_safety=RESPONSE_SAFETY,training_margin_safety=MARGIN_SAFETY,training_KL_safety=KL_SAFETY,
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        zero_functional_parity='All12 zero-control functional forwards must reproduce exact raw coordinates and native observations within2e-6, KL2e-7 before any gradient or update',
        preflight='12 preparations plus original raw/projected endpoints,36 known points; point/gate identity and observations2e-4, KL5e-6; drift stops before functional gradients',
        training_mechanism='Temporary graph-connected LoRA-B replacements via torch.func.functional_call. Backprop only to shared a. Backbone, LoRA-A and base writer not optimised.',
        functional_precision='Float64 control construction then float32 cast; AD follows the executed graph, not a certificate of exact finite-precision derivatives',
        final_checkpoint='Only update16 qualifies. No early stopping, best checkpoint, recovery choice, per-state head or input-class selector.',
        final_evaluation='Freeze step16 vector and all24 raw/trained proposal points before two-round48 native endpoint measurements. All12 remain training-qualification states.',
        new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher=False,backtracking=False,online_response_repair=False,
        candidate_selected=False,writer_deployed=False,excluded_input_states_evaluated=0,LC2_closed=False,worker_budget_seconds=TIMEOUT)

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
        self.data=data;self.phase='preflight';self.preflight={};self.active=None;self.taken=set();self.zero=set();self.batch=None;self.completed_batches=0;self.training_seen=set();self.final_points={}
        for spec in registry():
            sid=spec['state_id'];case=spec['case'];self.preflight[('prepared',sid)]=(safe.tensor_hash(data['points'][sid]['prepared']),case,False)
            for arm in ARMS[:2]:self.preflight[('endpoint',sid,arm)]=(safe.tensor_hash(data['maps'][(sid,arm)]['point']),case,True)
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.final_points
        need(self.phase in ('preflight','paths') and self.active is None,'Native metric phase forbidden');tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(key in table and table[key]==expected and tag not in self.taken,'Unsealed or repeated native metric request');self.taken.add(tag);self.active=expected
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native model point');self.active=None
    def begin_zero(self):
        need(self.phase=='preflight' and self.active is None and len(self.taken)==36,'All36 preflight calls required');self.phase='functional_zero'
    def functional(self,sid,a,point):
        need(sid in self.data['points'] and sid in {r['state_id'] for r in registry()},'Functional access outside fixed training registry')
        p=self.data['points'][sid];expected,_=proposal(p['prepared'],p['raw'],self.data['V'],a)
        if self.phase=='functional_zero':
            need(np.all(a==0) and sid not in self.zero,'Only one zero-control forward per training state');need(safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Zero control must retain exact raw bytes');self.zero.add(sid)
        else:
            need(self.phase=='training' and self.batch is not None and row_digest(a)==self.batch['digest'] and sid not in self.training_seen,'Wrong/duplicate training vector or state');self.training_seen.add(sid)
        return close(point,expected,'NumPy/functional proposal parity',2e-6)
    def begin_training(self):need(self.phase=='functional_zero' and len(self.zero)==12,'Complete functional zero parity required');self.phase='training'
    def begin_batch(self,step,a):
        need(self.phase=='training' and self.batch is None and step==self.completed_batches and step<STEPS,'Fixed fullbatch schedule required');self.batch=dict(step=step,digest=row_digest(a));self.training_seen=set()
    def end_batch(self):
        need(self.batch is not None and len(self.training_seen)==12,'All12 gradient contributions required');self.completed_batches+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and self.completed_batches==STEPS,'Only fixed step16 may freeze')
        need(set(points)=={(r['state_id'],arm) for r in registry() for arm in EVAL_ARMS},'All24 final outcomes required')
        self.final_points={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.taken)==36+48,'All48 final endpoint records required')
