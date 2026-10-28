"""Fixed84 exposed training states; one shared finite-write control."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor142'))
import core142 as auditcore
prior=auditcore.prior
base=prior.parent
parent=base.parent
from core142 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM
PROTOCOL='GPT2_LC2_C143_JOINT84_TRAINING_CPU_V1'
INPUT_KEYS=('C140','C141','C142');STEPS=12;N=84;TIMEOUT=7200
RATE=.0005;ADAM_B1=.9;ADAM_B2=.999;ADAM_EPS=1e-8;GRAD_CLIP=1.;GAIN_BOUND=.125
RESPONSE_SAFETY=.00125;MARGIN_SAFETY=.075;KL_SAFETY=.0045;L2=.001;TEMPERATURE=8.;UNIFORM_FRACTION=.1
RADIUS=20.;BUDGET=.002
EVAL_ARMS=('raw_control','C140_frozen_step12','C143_shared_step12')
proposal=base.proposal

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source142=clean_json((ROOT/'trusted/C142_SOURCE_PINS.json').read_bytes()))

def registry():
    old=[dict(r,C143_membership='training',C143_training_origin='C140_training36',used_for_training=True,independent_validation_eligible=False) for r in base.registry()]
    new=[dict(r,source_dataset_role='confirmation',dataset_role='training',used_for_training=True,C143_membership='training',training_origin='C141_exposed_confirmation48',C143_training_origin='C141_exposed_confirmation48',independent_validation_eligible=False) for r in prior.registry()]
    rows=old+new;need(len(rows)==len({r['state_id'] for r in rows})==84,'Exact84 training registry');need(Counter(r['case'] for r in rows)==dict(xor_00=28,xor_01=28,xor_11=28),'Three trained classes only');return rows

def head_digests(d):return dict(writer=safe.head_digest(d['writer']),control=safe.head_digest(d['control']),V=row_digest(d['V']),initial_C140_a=row_digest(d['initial_a']))

def design():return dict(protocol=PROTOCOL,training_registry=registry(),training_states=84,all84_are_training=True,independent_confirmation_states=0,excluded_input_states_evaluated=0,input_archives=list(INPUT_KEYS),initialization='Exact C140 fixed step12; reset both Adam moments to zero.',formula='float32(raw+V.T@(a*(V@(raw-prepared)))); one shared36 vector, no radial clipping.',offline_training_steps=12,states_per_full_gradient=84,scalar_backward_calls_requested=1008,final_checkpoint='Only fixed C143 step12; no best/early/recovery/checkpoint substitution.',optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[ADAM_B1,ADAM_B2],epsilon=ADAM_EPS,full_gradient_L2_clip=GRAD_CLIP,elementwise_projection=[-.125,.125],initial_moments='zero'),training_safety=dict(response=.00125,read_margin=.075,KL=.0045),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),objective='Same C140 mean+max squared response hinges, read/KL/radius hinges, .001*mean(a^2); aggregate .1*mean(L)+.9/8*log(mean(exp(8L))) over all84 at one shared vector.',membership='Historical C14036 and all C141 exposed48 are training. C141 confirmation failure retained; no state is reused as unseen validation, and no six-failure-only fitting.',preflight='All84 prepared/raw/C140 native endpoints replayed against originals; float32 point/gate identity and observations2e-4, KL5e-6; invalid preparation or replay drift stops before any gradient.',functional_parity='All84 zero-control and all84 fixed warm-control functional/native comparisons before any backward; point/observations2e-6, KL2e-7.',frozen_components='GPT2 backbone, LoRA-A, raw writer, original control, V, feature formulas and normalisation remain fixed. Only shared a trained offline.',fixed_requests=dict(native_preflight=252,functional_zero=84,functional_warm=84,training_scalar_backwards=1008,final_native_endpoints=504,model_forwards_including_initialisation=3865),freeze='Final a and all252 raw/C140/C143 proposals sealed before two-round504 native measurements.',pass_rule='Fixed final step12 complete native gates84/84 in both rounds, full training coverage and replay/repeat consistency; training safety counts do not replace original acceptance.',next_rule='If all84 training qualifies, freeze candidate and preregister separate fresh confirmation. Otherwise retain failure and stop this fixed schedule. No tuning within confirmation, excluded-input evaluation or LC2 claim.',new_states_prepared=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,LC2_closed=False,LC2_single_fold_passed=False,scope='Exposed84-state training qualification only; no unseen/excluded-input generalisation. Offline backprop is not online response repair. No independent derivative accuracy, convergence or eventual LC2 guarantee.',worker_budget_seconds=TIMEOUT)

def aggregate(losses,gradients):
    losses=farray(np.asarray(losses,float),(N,));gradients=np.asarray(gradients,float)
    need(gradients.ndim==2 and gradients.shape[0]==N and gradients.shape[1]>0 and np.isfinite(gradients).all(),'All84 finite unweighted gradients required');need(np.all(losses>=0),'Negative loss')
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
    def begin_zero(self):need(self.phase=='preflight' and self.active is None and len(self.taken)==252,'All252 preflight calls required');self.phase='functional_zero'
    def functional(self,sid,a,point):
        need(sid in {r['state_id'] for r in registry()},'Functional state outside84 training IDs');p=self.data['points'][sid];expected=proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
        if self.phase=='functional_zero':need(np.all(a==0) and sid not in self.zero and safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Only one exact zero forward per state');self.zero.add(sid)
        elif self.phase=='functional_warm':need(row_digest(a)==row_digest(self.data['initial_a']) and sid not in self.warm,'Only one fixed C140 warm forward per state');self.warm.add(sid)
        else:need(self.phase=='training' and self.batch is not None and row_digest(a)==self.batch['digest'] and sid not in self.training_seen,'Wrong training vector, phase or duplicate state');self.training_seen.add(sid)
        return close(point,expected,'NumPy/functional point parity',2e-6)
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==N,'All84 zero parity required');self.phase='functional_warm'
    def begin_training(self):need(self.phase=='functional_warm' and len(self.warm)==N,'All84 warm parity required');self.phase='training'
    def begin_batch(self,step,a):need(self.phase=='training' and self.batch is None and step==self.completed_batches and step<STEPS,'Fixed12 full84 update schedule');self.batch=dict(step=step,digest=row_digest(a));self.training_seen=set()
    def end_batch(self):need(self.batch is not None and len(self.training_seen)==N,'All84 gradients required');self.completed_batches+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and self.completed_batches==STEPS,'Only fixed step12 may qualify');need(set(points)=={(r['state_id'],arm) for r in registry() for arm in EVAL_ARMS},'All252 proposals required')
        self.final_points={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.taken)==252+504,'All504 endpoints required')
