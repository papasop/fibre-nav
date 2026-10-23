"""Frozen positive path grid: no step selection, training or online derivatives."""
from pathlib import Path
import hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor133'))
from core133 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,ARMS,DIM,RADIUS,BUDGET,registry as registry12,gate_record

PROTOCOL='GPT2_LC2_C134_FIXED_WRITE_PATH_CPU_V1'
INPUT_KEYS=('C132','C133');TIMEOUT=3000
ALPHAS=(0.,1/64,1/16,1/4,1/2,3/4,1.)

def registry():return [r for r in registry12() if r['role']=='geometry_training']

def contract():
    return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source132=clean_json((ROOT/'trusted/C132_SOURCE_PINS.json').read_bytes()),source133=clean_json((ROOT/'trusted/C133_SOURCE_PINS.json').read_bytes()))

def design():
    return dict(protocol=PROTOCOL,registry=registry(),arms=list(ARMS),alphas=list(ALPHAS),rounds=2,path_points_requested=126,path_measurements_requested=252,
        path='float32(prepared + alpha * (C132_actual_endpoint - prepared)); alpha0/1 retain exact archived coordinate bytes',
        endpoint_source='Actual committed C132 endpoints, not preclip or ideal projected deltas',
        freeze='All126 coordinate outcomes frozen before model initialisation, preflight and path measurements',
        preflight='Six preparations and all18 available full endpoints checked against C132 observations; point identity and gate identity required; observed absolute tolerance2e-4, KL5e-6',
        intermediate_target='alpha0: prepared-state gates; every alpha>0: same final requested-write gates. All gate failures retained.',
        derivative='Only the six archived C132 local10-row Jacobians; none recomputed or imputed to other centers',
        analysis='Fresh path observable minus fresh prewrite observable minus archived_J @ actual_float32_displacement; also report nominal alpha*J@full_displacement and linear rounding effect',
        near_origin_slope='(fresh_observable(alpha)-fresh_prewrite)/alpha compared to archived_J@full_displacement at the two fixed small fractions1/64 and1/16; disagreement alone is not proof of AD error',
        descriptive_grid_departure='First sampled alpha whose response remainder Linf exceeds fixed response budget; descriptive only, not a selected deployment step or continuous threshold',
        response_budget=.002,KL_budget=.005,read_margin=.05,proposal_radius=20.,
        online_Jacobians=0,new_Jacobians=0,new_SVD_calls=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,linear_systems_solved=0,
        step_selected=False,candidate_selected=False,postwrite_response_repair=False,teacher=False,backtracking=False,excluded_input_states_evaluated=0,LC2_closed=False,
        scope='Historical training-center path diagnostic only. No fresh-state/excluded-input generalisation or finite error certificate. Archived derivative accuracy and physical numerical causes remain unproved.',worker_budget_seconds=TIMEOUT)

def path_point(prepared,endpoint,alpha):
    need(type(alpha) is float and alpha in ALPHAS,'Unregistered or non-float alpha')
    farray(prepared,(DIM,));farray(endpoint,(DIM,));need(prepared.dtype==endpoint.dtype==np.float32,'Float32 endpoints required')
    if alpha==0.:return prepared.copy()
    if alpha==1.:return endpoint.copy()
    return (prepared.astype(float)+alpha*(endpoint.astype(float)-prepared.astype(float))).astype(np.float32)

def freeze_paths(data):
    arrays={};outcomes=[];maps={}
    for spec in registry():
        sid=spec['state_id'];prepared=data['points'][sid]['prepared']
        for arm in ARMS:
            endpoint=data['maps'][(sid,arm)]['point']
            for step,alpha in enumerate(ALPHAS):
                name=sid+'__'+arm+f'__s{step:02d}';point=None if endpoint is None else path_point(prepared,endpoint,alpha)
                if point is not None:arrays[name]=point
                row=dict(**spec,arm=arm,step=step,alpha=alpha,array_key=name,available=point is not None,point_sha256=safe.tensor_hash(point) if point is not None else None,evaluation_target='prepared_state' if alpha==0. else 'requested_write')
                outcomes.append(row);maps[(sid,arm,step)]=point
    need(len(outcomes)==126 and len(maps)==126,'Full126 path registry required')
    return arrays,outcomes,maps

def path_gates(metrics,case,point,prepared,alpha):
    if alpha>0:return gate_record(metrics,case,point,prepared)
    passed=safe.metric_gates(metrics,case,point,False)
    radius=float(np.linalg.norm(point.astype(float)-prepared.astype(float)));valid=radius<=RADIUS+1e-5
    return dict(complete_pass=bool(passed and valid),failed_gates=[k for k,v in metrics['gates'].items() if not v]+([] if valid else ['proposal_radius']),actual_delta_norm=radius,proposal_radius_passed=bool(valid),response_drift=metrics['response_drift'])

def analyse(j,prepared,endpoint,point,alpha,base,observed):
    j=farray(j,(10,DIM));before=farray(np.asarray(base,float),(10,));after=farray(np.asarray(observed,float),(10,));d=endpoint.astype(float)-prepared.astype(float);actual=point.astype(float)-prepared.astype(float)
    direction=j@d;nominal=alpha*direction;linear=j@actual;rounding=j@(actual-alpha*d);change=after-before;remainder=change-linear
    close(nominal+rounding,linear,'Rounded linear identity',1e-9);close(before+linear+remainder,after,'Fresh path closure',1e-9)
    vectors=dict(prewrite=before,observed=after,finite_change=change,linear_actual=linear,linear_nominal=nominal,linear_rounding_effect=rounding,finite_step_remainder=remainder,first_order_endpoint=before+linear,full_directional_action=direction)
    if alpha>0:
        slope=change/alpha;vectors.update(finite_fraction_slope=slope,slope_minus_archived_AD=slope-direction,remainder_div_alpha_squared=remainder/(alpha*alpha))
    stats=dict(response_drift_Linf=float(np.max(abs(after[:6]))),response_linear_action_Linf=float(np.max(abs(linear[:6]))),response_remainder_Linf=float(np.max(abs(remainder[:6]))),linear_rounding_response_Linf=float(np.max(abs(rounding[:6]))),
        first_order_response_within_budget=bool(np.max(abs((before+linear)[:6]))<=BUDGET),finite_response_within_budget=bool(np.max(abs(after[:6]))<=BUDGET),remainder_exceeds_response_budget=bool(np.max(abs(remainder[:6]))>BUDGET),
        response_slope_disagreement_Linf=float(np.max(abs(vectors['slope_minus_archived_AD'][:6]))) if alpha>0 else None,closure_Linf=float(np.max(abs(before+linear+remainder-after))))
    need(all(np.isfinite(v).all() for v in vectors.values()),'Nonfinite path component')
    return vectors,stats

class Phase:
    """One-use measured points, fixed preflight before any path request."""
    def __init__(self,data,paths):
        self.phase='preflight';self.preflight={};self.path={};self.taken=set();self.active=None
        for spec in registry():
            sid=spec['state_id'];case=spec['case'];self.preflight[('prepared',sid)]=(safe.tensor_hash(data['points'][sid]['prepared']),case,False)
            for arm in ARMS:
                p=data['maps'][(sid,arm)]['point']
                if p is not None:self.preflight[('endpoint',sid,arm)]=(safe.tensor_hash(p),case,True)
                for step,alpha in enumerate(ALPHAS):
                    q=paths[(sid,arm,step)]
                    if q is not None:
                        for rnd in (0,1):self.path[(rnd,sid,arm,step)]=(safe.tensor_hash(q),case,alpha>0)
    def authorize(self,key,point,case,write):
        need(self.active is None,'Previous model call not consumed');table=self.preflight if self.phase=='preflight' else self.path
        tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(key in table and table[key]==expected and tag not in self.taken,'Unsealed, repeated or wrongly labelled model request')
        self.taken.add(tag);self.active=expected
    def consume(self,point,case,write):
        need(self.active==(safe.tensor_hash(point),case,write),'Model call outside authorized point');self.active=None
    def begin_paths(self):
        need(self.phase=='preflight' and self.active is None and len(self.taken)==len(self.preflight),'Complete known-point replay required before paths');self.phase='paths'
    def finish(self):
        need(self.phase=='paths' and self.active is None and len(self.taken)==len(self.preflight)+len(self.path),'All two-round path calls required')
