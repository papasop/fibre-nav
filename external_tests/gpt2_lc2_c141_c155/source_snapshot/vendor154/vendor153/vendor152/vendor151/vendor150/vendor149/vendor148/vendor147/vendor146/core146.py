"""Declared all84 nonsmooth worst-response objective; original native gates unchanged."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor145'))
import core145 as prior
base=prior.parent
from core145 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C146_WORST_RESPONSE_TRAINING_CPU_V1'
INPUT_KEYS=('C145','C144','C143','C140','C141','C142')
N=84;STEPS=12;TIMEOUT=10800
RESPONSE_SAFETY=.0018;MARGIN_SAFETY=.075;KL_SAFETY=.0045;RADIUS=20.;L2=.001
MAX_FRACTION=.9;RATE=.0005;GAIN_BOUND=.125
EVAL_ARMS=('raw_control','C143_frozen_step12','C146_fixed_step12')
GROUPS=dict(all84=84,historical12=12,C139_exposed_kernel24=24,C141_exposed_confirmation48=48,xor_00=28,xor_01=28,xor_11=28)

def registry():return [dict(r,C146_membership='training') for r in prior.registry()]
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source145=clean_json((ROOT/'trusted/C145_SOURCE_PINS.json').read_bytes()),binding145=prior.contract())
def design():return dict(protocol=PROTOCOL,training_registry=registry(),training_states=84,input_archives=list(INPUT_KEYS),initialization='Exact original C143 final step12; zero Adam moments. No C145 probe chosen.',shared_dimension=36,offline_training_steps=12,final_checkpoint='Only C146 step12; no best, recovery, early-stop or probe substitution.',proposal_formula='float32(raw+V.T@(a*(V@(raw-prepared))))',training_objective='R_i=relu(max(abs(response_i))/0.0018-1)^2; G_i=read_hinge(.075)+KL_hinge(.0045)+radius_hinge(20)+.001*mean(a^2). J=.9*max_i(R_i)+.1*mean_i(R_i)+mean_i(G_i).',nonsmooth_rule='Exact maxima use the equally averaged subgradient across exact ties, both within six response coordinates and across all84 states. At zero hinge the derivative is zero.',priority_scope='Finite weighted worst-response objective with guard penalties, not lexicographic optimisation or a guarantee that every constraint or maximum decreases per step.',optimiser=dict(kind='manual float64 projected Adam',learning_rate=RATE,betas=[.9,.999],epsilon=1e-8,full_gradient_L2_clip=1.,gain_projection=[-.125,.125],initial_moments='zero'),training_safety=dict(response=RESPONSE_SAFETY,read_margin=MARGIN_SAFETY,KL=KL_SAFETY),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),preflight='All84 prepared/raw/C140/C143 native replays; point and gate identities, observations2e-4 and KL5e-6. Then all84 zero and all84 C143 functional/native comparisons before any backward, point/observations2e-6 and KL2e-7.',fixed_requests=dict(native_preflight=336,functional_zero=84,functional_warm=84,training_graph_forwards=1008,training_scalar_backwards=2016,final_native_endpoints=504,total_model_forwards=4033),freeze='Final step12 control and all252 raw/C143/C146 proposals sealed before two-round504 endpoint measurements.',qualification='Complete original gates84/84 in both rounds plus replay/repeat consistency; safety lines and objective reduction cannot replace native acceptance.',all84_are_exposed_training=True,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,full_response_Jacobian_constructed=False,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT,scope='Offline all84 exposed-training qualification only. No new candidate deployment, independent confirmation, excluded-input generalisation, convergence, feasibility or derivative-accuracy proof. Two rounds repeat the same84 states. C145 archival derivative vectors are checked algebraically, not independently validated as derivatives.')

def objective(o,kl,point,prepared,a,case):
    o=farray(np.asarray(o,float),(22,));farray(point,(DIM,));farray(prepared,(DIM,));farray(a,(len(a),))
    need(case in ('xor_00','xor_01','xor_11'),'Excluded class forbidden')
    need(type(kl) in (float,int) and np.isfinite(kl) and kl>=0,'Invalid KL')
    q=float(np.max(abs(o[:6]))/RESPONSE_SAFETY);signs=2*np.array(safe.target_bits(case,True)*4)-1
    return dict(response_loss=float(max(q-1,0)**2),read_loss=float(np.mean(np.maximum(1-signs*o[6:]/MARGIN_SAFETY,0)**2)),KL_loss=float(max(kl/KL_SAFETY-1,0)**2),radius_loss=float(max(np.linalg.norm(point.astype(float)-prepared.astype(float))/RADIUS-1,0)**2),regularizer=float(L2*np.mean(a*a)))

def aggregate(rows,rank):
    need(len(rows)==N and [r['state_id'] for r in rows]==[r['state_id'] for r in registry()],'All84 ordered training rows required')
    r=farray(np.asarray([x['objective_parts']['response_loss'] for x in rows],float),(N,));g=farray(np.asarray([sum(v for k,v in x['objective_parts'].items() if k!='response_loss') for x in rows],float),(N,))
    gr=farray(np.asarray([x['response_gradient'] for x in rows],float),(N,rank));gg=farray(np.asarray([x['guard_gradient'] for x in rows],float),(N,rank));need(np.all(r>=0) and np.all(g>=0),'Nonnegative objective parts required')
    ties=r==r.max();w=np.full(N,(1-MAX_FRACTION)/N)+MAX_FRACTION*ties/ties.sum()
    contributions=w[:,None]*gr+gg/N;gradient=contributions.sum(0)
    return gradient,dict(objective=float(MAX_FRACTION*r.max()+(1-MAX_FRACTION)*r.mean()+g.mean()),response_weights=w.tolist(),guard_weights=[1/N]*N,maximum_response_loss=float(r.max()),mean_response_loss=float(r.mean()),mean_guard_loss=float(g.mean()),maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(ties)],exact_max_tie_count=int(ties.sum()),state_gradient_contributions=contributions.tolist())

def update(a,g,m,v,step):return base.adam_step(a,g,m,v,step)
def proposal(prepared,raw,V,a):return base.proposal(prepared,raw,V,a)
def frozen_digest(d):return prior.frozen_digest(d)

class Phase:
    def __init__(self,data):
        self.data=data;self.phase='preflight';self.active=None;self.used=set();self.zero=set();self.warm=set();self.batch=None;self.completed_batches=0;self.training_seen=set();self.preflight={};self.paths={}
        for sp in registry():
            sid=sp['state_id'];p=data['points'][sid]
            for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('C140',proposal(p['prepared'],p['raw'],data['V'],data['C140_a'])[0],True),('C143',proposal(p['prepared'],p['raw'],data['V'],data['a'])[0],True)]:self.preflight[(sid,kind)]=(safe.tensor_hash(point),sp['case'],write)
    def authorize(self,key,point,case,write):
        table=self.preflight if self.phase=='preflight' else self.paths;tag=(self.phase,key);want=(safe.tensor_hash(point),case,write)
        need(self.phase in ('preflight','paths') and self.active is None and table.get(key)==want and tag not in self.used,'Unregistered/duplicate native point');self.active=want;self.used.add(tag)
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native call');self.active=None
    def begin_zero(self):need(self.phase=='preflight' and self.active is None and len(self.used)==336,'All336 preflights first');self.phase='functional_zero'
    def functional(self,sid,a,point):
        need(sid in self.data['points'],'State outside84');p=self.data['points'][sid];expected=proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
        if self.phase=='functional_zero':need(np.all(a==0) and sid not in self.zero and safe.tensor_hash(point)==safe.tensor_hash(p['raw']),'Exact zero point required');self.zero.add(sid)
        elif self.phase=='functional_warm':need(row_digest(a)==row_digest(self.data['a']) and sid not in self.warm,'Exact original C143 control required');self.warm.add(sid)
        else:need(self.phase=='training' and self.batch is not None and row_digest(a)==self.batch['digest'] and sid not in self.training_seen,'Invalid training vector/state/phase');self.training_seen.add(sid)
        return close(point,expected,'NumPy/Torch point parity',2e-6)
    def begin_warm(self):need(self.phase=='functional_zero' and len(self.zero)==N,'All84 zero parities first');self.phase='functional_warm'
    def begin_training(self):need(self.phase=='functional_warm' and len(self.warm)==N,'All84 warm parities first');self.phase='training'
    def begin_batch(self,step,a):need(self.phase=='training' and self.batch is None and type(step) is int and step==self.completed_batches and step<STEPS,'Fixed12 complete batches only');self.batch=dict(step=step,digest=row_digest(a));self.training_seen=set()
    def end_batch(self):need(self.batch is not None and len(self.training_seen)==N,'All84 gradients required');self.completed_batches+=1;self.batch=None
    def freeze(self,points):
        need(self.phase=='training' and self.batch is None and self.completed_batches==STEPS,'Only fixed step12 may qualify');need(set(points)=={(r['state_id'],arm) for r in registry() for arm in EVAL_ARMS},'All252 proposals required')
        self.paths={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.used)==336+504,'Complete native denominator required')

def final_summaries(records):
    sums={};pairs={}
    for group,n in GROUPS.items():
        sums[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];lookup={(r['state_id'],r['arm']):r for r in rr};need(len(lookup)==n*3,'Complete final denominator required');sums[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in EVAL_ARMS:
                ar=[r for r in rr if r['arm']==arm];need(len(ar)==n,'Every state retained');failed=Counter(k for r in ar for k in r['gates']['failed_gates'])
                sums[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in ar),failed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass']],failed_gates=dict(sorted(failed.items())),maximum_response_drift=max(r['metrics']['response_drift'] for r in ar),within_training_response_safety=sum(r['metrics']['response_drift']<=RESPONSE_SAFETY for r in ar))
                pairs[group][str(rnd)][arm]={}
                for control in EVAL_ARMS[:2]:pairs[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in ar if r['gates']['complete_pass'] and not lookup[(r['state_id'],control)]['gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in ar if not r['gates']['complete_pass'] and lookup[(r['state_id'],control)]['gates']['complete_pass']])
    return sums,pairs
