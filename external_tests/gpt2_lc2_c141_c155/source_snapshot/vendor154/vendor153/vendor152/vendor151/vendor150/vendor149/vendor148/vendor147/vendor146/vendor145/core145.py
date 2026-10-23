"""Fixed all84 local direction diagnostic, not a writer selection procedure."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor144'))
import core144 as auditcore
parent=auditcore.parent
from core144 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C145_FIXED_DIRECTION_DIAGNOSTIC_CPU_V1'
INPUT_KEYS=('C144','C143','C140','C141','C142');N=84;TIMEOUT=7200
DIRECTIONS=('original_objective','response_softmax32');STEPS=(.0005,.0015);SIGNS=(1,-1)

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source144=clean_json((ROOT/'trusted/C144_SOURCE_PINS.json').read_bytes()),audit144=auditcore.contract())

def registry():return parent.registry()

def configurations():
    return [dict(config_id='C143_baseline',direction=None,step=0.,sign=0)]+[dict(config_id=f'{direction}__h{j:02d}__'+('plus' if sign==1 else 'minus'),direction=direction,step=step,sign=sign) for direction in DIRECTIONS for j,step in enumerate(STEPS) for sign in SIGNS]

def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),training_states=84,all84_remain_exposed_training=True,initial_control='Exact C143 final step12; never updated in place.',shared_dimension=36,direction_rules=['Negative aggregate gradient of the original C143 training objective over all84 at C143 final a.','Negative softmax32-weighted gradient of q_i=max(abs(response_i))/.002 over all84 at the same frozen a.'],direction_normalization='Negative gradient divided by its Linf norm; norm<=1e-14 gives an explicit zero direction, never an arbitrary fallback.',state_gradient_vectors=168,scalar_backward_calls=168,functional_endpoint_calls=168,functional_parity='All84 native/function comparisons before any backward; point/observations2e-6 and KL2e-7. Gradient-forward observations must agree with the fixed functional baseline.',preflight='All84 prepared/raw/C140/C143 native points; original point/gate identity and observations2e-4, KL5e-6; drift blocks before any backward.',preflight_metric_requests=336,configurations=configurations(),probe_control_rule='a_probe=clip(a_C143+sign*h*direction,-.125,.125); every clipping event and actual coefficient displacement recorded. All8 vectors and all756 points sealed before probe measurements.',response_softmax_temperature=32.,baseline_and_probe_configurations=9,frozen_points=756,rounds=2,native_endpoint_requests=1512,requested_model_forwards=4033,acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),diagnostic_only='All eight signed fixed probes evaluated on every84 state twice. No adaptive step, line search, optimizer, iterative fit, best point/radius/direction selection or deployment. Both signs and both steps retained, even if a direction is zero or proposals coincide.',linear_prediction='q_i(a_probe) approximately q_i(a0)+archived-at-runtime grad(q_i) dot actual coefficient displacement; reported only as a local AD linearization diagnostic. Max/tie changes, clipping, float32 steps and nonlinear effects are retained; discrepancies do not prove AD error.',scope='Exposed training-domain finite directional probes only. No independent derivative accuracy, feasibility/infeasibility, convergence, fresh local confirmation or excluded-input generalisation claim. All84 pass at a probe is descriptive and does not select a writer or close LC2.',new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT)

def qvalue(observed):return float(np.max(np.abs(farray(np.asarray(observed,float),(22,))[:6]))/.002)

def directions_from_rows(records,rank):
    need(len(records)==84 and [r['state_id'] for r in records]==[r['state_id'] for r in registry()],'All84 ordered scalar gradients required')
    original=farray(np.asarray([r['original_gradient'] for r in records],float),(84,rank));response=farray(np.asarray([r['response_gradient'] for r in records],float),(84,rank))
    loss=farray(np.asarray([r['original_objective'] for r in records],float),(84,));q=farray(np.asarray([r['response_ratio'] for r in records],float),(84,));need(np.all(loss>=0) and np.all(q>=0),'Nonnegative losses/ratios required')
    ga,oa=auditcore.aggregate(loss,original);e=np.exp(32*(q-np.max(q)));w=e/e.sum();gb=w@response
    need(np.isfinite(ga).all() and np.isfinite(gb).all(),'Nonfinite aggregate direction gradient')
    result={};details={}
    for name,g in [(DIRECTIONS[0],ga),(DIRECTIONS[1],gb)]:
        norm=float(np.max(abs(g),initial=0));available=norm>1e-14;d=-g/norm if available else np.zeros(rank)
        result[name]=d;details[name]=dict(gradient=g.tolist(),gradient_Linf=norm,gradient_L2=float(np.linalg.norm(g)),available=available,negative_gradient_Linf_normalized=d.tolist(),normalization='Linf',state_weights=(oa['weights'] if name==DIRECTIONS[0] else w.tolist()))
    return result,details

def probe_controls(a,directions):
    farray(a,(len(a),));need(np.max(abs(a),initial=0)<=.125,'Baseline gain bound')
    need(set(directions)==set(DIRECTIONS),'Exactly two registered directions');controls={};records=[]
    for cfg in configurations():
        if cfg['direction'] is None:raw=a.copy()
        else:
            d=farray(directions[cfg['direction']],a.shape);need(np.max(abs(d),initial=0)<=1+1e-12,'Direction normalization exceeds one');raw=a+cfg['sign']*cfg['step']*d
        control=np.clip(raw,-.125,.125);change=control-a;controls[cfg['config_id']]=control
        records.append(dict(**cfg,control_digest=row_digest(control),projected_coordinates=int(np.count_nonzero(raw!=control)),coefficient_change=change.tolist(),coefficient_change_Linf=float(np.max(abs(change),initial=0)),coefficient_change_L2=float(np.linalg.norm(change))))
    return controls,records

def frozen_digest(d):
    return dict(inputs=safe.head_digest({sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')}),V=row_digest(d['V']),baseline_a=row_digest(d['a']),C140_a=row_digest(d['C140_a']),writer=safe.head_digest(d['writer']),control=safe.head_digest(d['control']))

def tables(records):
    groups={'all84':84,'historical12':12,'C139_exposed_kernel24':24,'C141_exposed_confirmation48':48,'xor_00':28,'xor_01':28,'xor_11':28};result={}
    for group,n in groups.items():
        result[group]={}
        for rnd in (0,1):
            subset=[r for r in records if r['round']==rnd and (group=='all84' or r.get('training_origin')==group or r['case']==group)];baseline={r['state_id']:r for r in subset if r['config_id']=='C143_baseline'};need(len(baseline)==n,'Complete baseline denominator')
            result[group][str(rnd)]={}
            for cfg in configurations():
                rr=[r for r in subset if r['config_id']==cfg['config_id']];need(len(rr)==n and len({r['state_id'] for r in rr})==n,'Complete diagnostic denominator')
                result[group][str(rnd)][cfg['config_id']]=dict(n_requested=n,n_evaluated=n,passed=sum(r['gates']['complete_pass'] for r in rr),failed_ids=[r['state_id'] for r in rr if not r['gates']['complete_pass']],failed_gates=dict(sorted(Counter(g for r in rr for g in r['gates']['failed_gates']).items())),maximum_response_drift=max(r['metrics']['response_drift'] for r in rr),rescued_vs_baseline=[r['state_id'] for r in rr if r['gates']['complete_pass'] and not baseline[r['state_id']]['gates']['complete_pass']],destroyed_vs_baseline=[r['state_id'] for r in rr if not r['gates']['complete_pass'] and baseline[r['state_id']]['gates']['complete_pass']])
    return result

class Phase:
    def __init__(self,data):
        self.data=data;self.phase='preflight';self.active=None;self.used=set();self.parity=set();self.gradients=set();self.preflight={};self.paths={}
        for sp in registry():
            sid=sp['state_id'];p=data['points'][sid]
            for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('C140',parent.proposal(p['prepared'],p['raw'],data['V'],data['C140_a'])[0],True),('C143',parent.proposal(p['prepared'],p['raw'],data['V'],data['a'])[0],True)]:self.preflight[(sid,kind)]=(safe.tensor_hash(point),sp['case'],write)
    def authorize(self,key,point,case,write):
        expected=(safe.tensor_hash(point),case,write);table=self.preflight if self.phase=='preflight' else self.paths
        need(self.phase in ('preflight','paths') and self.active is None and table.get(key)==expected and (self.phase,key) not in self.used,'Unregistered/duplicate native point');self.active=expected;self.used.add((self.phase,key))
    def consume(self,point,case,write):need(self.active==(safe.tensor_hash(point),case,write),'Unauthorized native call');self.active=None
    def begin_parity(self):need(self.phase=='preflight' and self.active is None and len(self.used)==336,'All336 preflights before parity');self.phase='parity'
    def functional(self,sid,a,point,kind):
        need(self.phase==kind and kind in ('parity','gradient'),'Functional phase mismatch');need(row_digest(a)==row_digest(self.data['a']),'Only frozen C143 control may be differentiated');need(sid in self.data['points'],'Unregistered state')
        used=self.parity if kind=='parity' else self.gradients;need(sid not in used,'Duplicate functional state');used.add(sid);p=self.data['points'][sid];expected=parent.proposal(p['prepared'],p['raw'],self.data['V'],a)[0]
        return close(point,expected,'NumPy/Torch committed point parity',2e-6)
    def begin_gradients(self):need(self.phase=='parity' and len(self.parity)==84,'All84 parity records before gradients');self.phase='gradient'
    def freeze(self,maps):
        need(self.phase=='gradient' and len(self.gradients)==84,'All84 two-gradient records before probes');need(set(maps)=={(r['state_id'],cfg['config_id']) for r in registry() for cfg in configurations()},'All756 frozen points required')
        self.paths={(rnd,sid,cfg):(safe.tensor_hash(point),sid.split('__')[0],True) for (sid,cfg),point in maps.items() for rnd in (0,1)};self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.used)==336+1512,'All1512 fixed endpoints required')
