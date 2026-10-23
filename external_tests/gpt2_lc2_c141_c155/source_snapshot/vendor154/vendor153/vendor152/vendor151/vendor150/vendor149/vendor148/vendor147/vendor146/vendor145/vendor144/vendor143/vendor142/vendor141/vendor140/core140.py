"""One shared control, fixed36 historical training states and fixed12 updates."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor139'))
import core139 as previous
from core139 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,RADIUS,BUDGET,head_digests
parent=previous.parent
PROTOCOL='GPT2_LC2_C140_EXPANDED_MARGIN_TRAINING_CPU_V1'
INPUT_KEYS=previous.INPUT_KEYS+('C139',);TIMEOUT=3000;STEPS=12;N=36
RATE=.0005;ADAM_B1=.9;ADAM_B2=.999;ADAM_EPS=1e-8;GRAD_CLIP=1.;GAIN_BOUND=.125
RESPONSE_SAFETY=.00125;MARGIN_SAFETY=.075;KL_SAFETY=.0045;L2=.001;TEMPERATURE=8.;UNIFORM_FRACTION=.1
EVAL_ARMS=('raw_control','C137_frozen_step24','C140_shared_step12')
proposal=parent.proposal

def contract():return dict(previous.contract(),pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source139=clean_json((ROOT/'trusted/C139_SOURCE_PINS.json').read_bytes()))

def registry():
    out=[]
    for r in parent.registry():out.append(dict(**r,training_origin='historical12',dataset_role='training',C140_membership='training',independent_validation_eligible=False))
    for r in previous.registry():out.append(dict(r,dataset_role='training',source_dataset_role='development',training_origin='C139_exposed_kernel24',C140_membership='training',independent_validation_eligible=False))
    need(len(out)==len({r['state_id'] for r in out})==N,'All36 training IDs required');return out

def design():
    return dict(protocol=PROTOCOL,training_registry=registry(),training_states=N,independent_confirmation_states=0,
        training_membership='All original12 and all C139 exposed kernel24 are training. Do not fit only the failed state. Historical C137/C138/C139 decisions remain unchanged; historical development roles are provenance, not C140 validation.',
        initialization='Exact frozen C137 step24 a; reset both Adam moments to zero. No use of C139 endpoints as proposal inputs; archived raw coordinates and prewrite features are reused unchanged.',
        formula='point=float32(raw+V.T@(a*(V@(raw-prepared)))); one shared36 vector; original raw writer, control, normalization, features and C132 V fixed',
        preparation='No new preparation or feature acquisition. Independently reconstruct C139 projection coordinates and writer maps, bind all36 stored prewrite records before model. Replay every preparation and both raw/C137 endpoints before any backward.',
        offline_training_steps=STEPS,states_per_full_gradient=N,scalar_backward_calls_requested=STEPS*N,
        optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[ADAM_B1,ADAM_B2],epsilon=ADAM_EPS,full_gradient_L2_clip=GRAD_CLIP,elementwise_projection=[-GAIN_BOUND,GAIN_BOUND],initial_moments='zero'),
        state_loss='Response hinge-square mean+max at0.00125, read-margin hinge-square at0.075, KL hinge-square at0.0045, actual displacement hinge-square at20, plus0.001*mean(a^2)',
        aggregate_objective='0.1*mean(L_i)+0.9/8*log(mean(exp(8*L_i))); stable log-sum-exp across all36 losses at the same shared vector',
        aggregate_gradient='sum_i(.1/36+.9*softmax(8L)_i)*dL_i/da; retain every unweighted state gradient and its computed weight',
        training_response_safety=RESPONSE_SAFETY,training_margin_safety=MARGIN_SAFETY,training_KL_safety=KL_SAFETY,
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        thresholds='Training safety tightened; original native acceptance thresholds unchanged. No threshold relaxation or rounding-away of failures.',
        preflight='All36 preparation/raw/C137 native points,108 calls; compare point and gate identity, observations2e-4, KL5e-6 with original references. Any drift or invalid preparation stops before gradients; no state deletion.',
        functional_parity='All36 zero-control versus native raw, then all36 initial C137 versus native warm; observations2e-6 and KL2e-7 before any backward; NumPy/Torch proposal parity2e-6',
        training_mechanism='torch.func.functional_call with temporary LoRA-B chart replacements; only shared a receives gradient. Frozen GPT2 backbone/A/base writer. AD follows executed float64-to-float32 graph; not an independent derivative accuracy certificate.',
        final_checkpoint='Only fixed new C140 update12; no early stopping, best checkpoint, recovery choice, per-state head, input-class selector or postwrite repair. No radial proposal clipping.',
        final_evaluation='Seal a and all108 raw/C137/C140 proposals before two-round216 native endpoints. Compare final against both raw and C137, retaining all36 including every failure.',
        fixed_requests=dict(native_preflight=108,functional_zero=36,functional_warm=36,training_scalar_backwards=432,final_native_endpoints=216,model_forwards_including_initialisation=1657),
        pass_rule='Both rounds fixed step12 pass all36 complete native gates, repeat/control consistency and complete registered training required. Training safety-band counts are reported separately and never substitute for acceptance.',
        next_rule='Only an all36 training pass permits designing a separately preregistered unseen local confirmation; any failure stops this fixed schedule without checkpoint substitution. No excluded-input evaluation in this run.',
        new_states_prepared=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,
        excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT,
        scope='Expanded historical training qualification only. All36 states are training; no unseen, excluded-input, cross-model or LC2 acceptance. Chronology/non-access and finite-precision AD accuracy not independently certified.')

def aggregate(losses,gradients):
    losses=farray(np.asarray(losses,float),(N,));gradients=np.asarray(gradients,float)
    need(gradients.ndim==2 and gradients.shape[0]==N and gradients.shape[1]>0 and np.isfinite(gradients).all(),'All36 finite unweighted gradients required');need(np.all(losses>=0),'Negative loss')
    shift=losses.max();e=np.exp(TEMPERATURE*(losses-shift));weights=UNIFORM_FRACTION/N+(1-UNIFORM_FRACTION)*e/e.sum();value=UNIFORM_FRACTION*losses.mean()+(1-UNIFORM_FRACTION)*(shift+np.log(e.mean())/TEMPERATURE);g=weights@gradients
    need(np.isfinite(value) and np.isfinite(g).all(),'Nonfinite aggregate')
    return g,dict(objective=float(value),weights=weights.tolist(),state_losses=losses.tolist(),temperature=TEMPERATURE,uniform_fraction=UNIFORM_FRACTION,maximum_state_loss=float(shift),mean_state_loss=float(losses.mean()))

def objective(observed,kl,point,prepared,a,case):
    o=farray(np.asarray(observed,float),(22,));need(np.isfinite(kl) and kl>=0,'Invalid KL');rv=np.maximum(abs(o[:6])/RESPONSE_SAFETY-1,0)**2;signs=2*np.array(safe.target_bits(case,True)*4)-1;read=np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)**2
    return dict(response_loss=float(rv.mean()+rv.max()),read_loss=float(read.mean()),KL_loss=float(max(kl/KL_SAFETY-1,0)**2),radius_loss=float(max(np.linalg.norm(point.astype(float)-prepared.astype(float))/RADIUS-1,0)**2),regularizer=float(L2*np.mean(a*a)))

def adam_step(a,g,m,v,step):
    need(type(step) is int and 1<=step<=STEPS,'Unregistered fixed update')
    for x in (a,g,m,v):farray(x,a.shape)
    need(np.all(v>=0),'Negative second moment');norm=float(np.linalg.norm(g));factor=min(1.,GRAD_CLIP/max(norm,1e-30));gg=g*factor;mm=ADAM_B1*m+(1-ADAM_B1)*gg;vv=ADAM_B2*v+(1-ADAM_B2)*gg*gg
    unprojected=a-RATE*(mm/(1-ADAM_B1**step))/(np.sqrt(vv/(1-ADAM_B2**step))+ADAM_EPS);new=np.clip(unprojected,-GAIN_BOUND,GAIN_BOUND);need(np.isfinite(new).all(),'Nonfinite Adam update')
    return new,mm,vv,dict(step=step,gradient_L2=norm,gradient_clip_factor=factor,projected_coordinates=int(np.sum(new!=unprojected)),update_Linf=float(np.max(abs(new-a),initial=0)))

def frozen_digest(d):
    tensors={sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')};tensors.update(V=d['V'],initial_a=d['initial_a'])
    return dict(training_inputs=safe.head_digest(tensors),original_heads=head_digests(d))

class Phase:
    def __init__(self,data):
        self.data=data;self.phase='preflight';self.active=None;self.taken=set();self.zero=set();self.warm=set();self.batch=None;self.completed_batches=0;self.training_seen=set();self.preflight={};self.final_points={}
        for sp in registry():
            sid=sp['state_id'];p=data['points'][sid]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',proposal(p['prepared'],p['raw'],data['V'],data['initial_a'])[0],True)):self.preflight[(kind,sid)]=(safe.tensor_hash(point),sp['case'],write)
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.final_points;tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(self.phase in ('preflight','paths') and self.active is None and table.get(key)==expected and tag not in self.taken,'Unsealed/duplicate native point or phase');self.active=expected;self.taken.add(tag)
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native access');self.active=None
    def begin_zero(self):need(self.phase=='preflight' and self.active is None and len(self.taken)==108,'All108 preflight calls required');self.phase='functional_zero'
    def functional(self,sid,a,point):
        need(sid in {r['state_id'] for r in registry()},'Functional state outside36 training IDs');p=self.data['points'][sid];expected=proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
        if self.phase=='functional_zero':need(np.all(a==0) and sid not in self.zero and safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Only one exact zero forward per state');self.zero.add(sid)
        elif self.phase=='functional_warm':need(row_digest(a)==row_digest(self.data['initial_a']) and sid not in self.warm,'Only one fixed C137 warm forward per state');self.warm.add(sid)
        else:need(self.phase=='training' and self.batch is not None and row_digest(a)==self.batch['digest'] and sid not in self.training_seen,'Wrong training vector, phase or duplicate state');self.training_seen.add(sid)
        return close(point,expected,'NumPy/functional point parity',2e-6)
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==N,'All36 zero parity required');self.phase='functional_warm'
    def begin_training(self):need(self.phase=='functional_warm' and len(self.warm)==N,'All36 warm parity required');self.phase='training'
    def begin_batch(self,step,a):need(self.phase=='training' and self.batch is None and step==self.completed_batches and step<STEPS,'Fixed12 full36 update schedule');self.batch=dict(step=step,digest=row_digest(a));self.training_seen=set()
    def end_batch(self):need(self.batch is not None and len(self.training_seen)==N,'All36 gradients required');self.completed_batches+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and self.completed_batches==STEPS,'Only fixed step12 may qualify');need(set(points)=={(r['state_id'],arm) for r in registry() for arm in EVAL_ARMS},'All108 proposals required')
        self.final_points={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.taken)==108+216,'All216 endpoints required')
