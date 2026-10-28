"""One frozen normal-multiplier family; historical development evidence only."""
from pathlib import Path
import hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor134'))
from core134 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,ARMS,DIM,RADIUS,BUDGET,registry12,gate_record

PROTOCOL='GPT2_LC2_C135_NORMAL_COEFFICIENT_CPU_V1'
INPUT_KEYS=('C132','C133','C134');TIMEOUT=3000
BETAS=(0.,7/8,15/16,31/32,1.,33/32,17/16)
ROLES=('all12','geometry_training','historical_seen_center_probe')

def registry():return registry12()

def contract():
    return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),
        source132=clean_json((ROOT/'vendor134/trusted/C132_SOURCE_PINS.json').read_bytes()),
        source133=clean_json((ROOT/'vendor134/trusted/C133_SOURCE_PINS.json').read_bytes()),
        source134=clean_json((ROOT/'trusted/C134_SOURCE_PINS.json').read_bytes()))

def design():
    return dict(protocol=PROTOCOL,registry=registry(),betas=list(BETAS),rounds=2,coefficient_points_requested=84,endpoint_measurements_requested=168,
        parameterisation='N=V.T@V, P=I-N for the fixed C132 shared response row space; u_beta=P*d+beta*N*d with d=archived_raw-prepared',
        controls='beta0: exact C132 shared_response_projected point; beta1: exact original raw point, retained byte-for-byte and never counted as a new improvement',
        normal_scope='Euclidean row-space component of the fixed C132 shared response stack, not a certified normal bundle or the local row space at all12 states',
        positive_beta_above_one='33/32 and17/16 amplify rather than merely retain the row-space component',
        radius_and_commit='One fixed radius20 cap on each noncontrol u_beta, then float32 commit; beta1 retains original raw bytes; report pre/post-cap and rounding diagnostics. Actual displacement radius checked with1e-5 rounding allowance.',
        freeze='All84 proposals, coefficients and roles frozen before any model initialisation or C135 observations; no C134 postwrite value enters proposal formula',
        preflight='All12 preparations plus raw and projected full endpoints:36 known points. Compare full22 observables and gates to C132; for geometry6 also compare raw/projected to both C134 final-point rounds. Observed tolerance2e-4, KL5e-6; any drift stops before new coefficient endpoints.',
        acceptance='Original complete content, preservation, response, KL, read-margin and radius gates at every coefficient; failures and all requested states retained',
        historical_feasibility_rule='List every beta!=1 passing all12 twice, repeatably and with stable known-point controls, rescuing at least one raw failure in each round. Do not choose among them or deploy. Partial improvements are descriptive only.',
        stop_rule='If no registered noncontrol coefficient meets the full historical rule, stop this fixed multiplier family; do not add coefficients after observing results.',
        local_derivatives='Only archived C132 Jacobians for geometry6; no local derivative or remainder imputed to the other6 states',
        response_budget=.002,KL_budget=.005,read_margin=.05,proposal_radius=20.,new_Jacobians=0,new_SVD_calls=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,linear_systems_solved=0,
        candidate_selected=False,coefficient_selected=False,teacher=False,backtracking=False,response_repair=False,excluded_input_states_evaluated=0,LC2_closed=False,worker_budget_seconds=TIMEOUT,
        scope='One registered shared-coefficient diagnostic on12 historical training centers. These centers and prior outcomes are known development data, not fresh confirmation or excluded inputs. Even complete historical acceptance does not establish generalisation or close LC2.')

def proposal(prepared,raw,v,beta):
    need(type(beta) is float and beta in BETAS,'Unregistered normal coefficient')
    farray(prepared,(DIM,));farray(raw,(DIM,));need(prepared.dtype==raw.dtype==np.float32,'Float32 archived coordinates required')
    need(v.ndim==2 and v.shape[1]==DIM and v.dtype.kind=='f' and np.isfinite(v).all(),'Invalid fixed shared row basis')
    d=raw.astype(float)-prepared.astype(float);normal=v.T@(v@d);tangent=d-normal
    u=d.copy() if beta==1. else tangent+beta*normal;norm=float(np.linalg.norm(u));factor=1. if beta==1. else min(1.,RADIUS/max(norm,1e-30));limited=u*factor
    point=raw.copy() if beta==1. else (prepared.astype(float)+limited).astype(np.float32);actual=point.astype(float)-prepared.astype(float)
    diag=dict(preclip_norm=norm,clip_factor=factor,radius_active=bool(factor<1),actual_delta_norm=float(np.linalg.norm(actual)),float32_commit_error_Linf=float(np.max(abs(actual-limited))),
        raw_control_identity=bool(beta==1.),normal_multiplier_before_radius_cap=beta,postcap_is_not_pure_normal_adjustment=bool(factor<1))
    return point,dict(tangent=tangent,normal=normal,ideal=u,limited=limited,actual=actual),diag

def freeze_coefficients(data):
    arrays={};outcomes=[];maps={};parts={};v=data['V']
    for spec in registry():
        sid=spec['state_id'];p=data['points'][sid]
        for i,beta in enumerate(BETAS):
            point,components,diag=proposal(p['prepared'],p['raw'],v,beta)
            if beta==0.:
                archived=data['maps'][(sid,ARMS[1])]['point'];exact(point,archived,'beta0 must reproduce C132 projected endpoint')
                point=archived.copy();components['actual']=point.astype(float)-p['prepared'].astype(float)
            if beta==1.:exact(point,p['raw'],'beta1 must retain original raw endpoint')
            name=sid+f'__b{i:02d}';arrays[name]=point;maps[(sid,i)]=point;parts[(sid,i)]=components
            outcomes.append(dict(**spec,coefficient_index=i,beta=beta,array_key=name,point_sha256=safe.tensor_hash(point),control='original_raw' if beta==1. else ('original_projected' if beta==0. else None),**diag))
    need(len(outcomes)==len(arrays)==84,'All84 fixed outcomes required');return arrays,outcomes,maps,parts

def local_diagnostic(j,parts,beta,base,observed):
    if j is None:return None,dict(local_derivative_available=False,finite_remainder_claimed=False)
    j=farray(j,(10,DIM));before=farray(np.asarray(base,float),(10,));after=farray(np.asarray(observed,float),(10,))
    lt=j@parts['tangent'];ln=j@parts['normal'];ideal=j@parts['ideal'];actual=j@parts['actual'];cap=j@(parts['limited']-parts['ideal']);rounding=j@(parts['actual']-parts['limited']);remainder=after-before-actual
    close(lt+beta*ln,ideal,'Tangent/normal first-order identity',1e-9);close(ideal+cap+rounding,actual,'Commit first-order identity',1e-9);close(before+actual+remainder,after,'Finite diagnostic closure',1e-9)
    vectors=dict(prewrite=before,endpoint=after,linear_tangent=lt,linear_normal=ln,linear_ideal=ideal,linear_actual=actual,linear_cap_effect=cap,linear_commit_rounding_effect=rounding,finite_step_remainder=remainder)
    stats=dict(local_derivative_available=True,finite_remainder_claimed=True,response_linear_Linf=float(np.max(abs(actual[:6]))),response_remainder_Linf=float(np.max(abs(remainder[:6]))),linear_cap_effect_response_Linf=float(np.max(abs(cap[:6]))),linear_rounding_effect_response_Linf=float(np.max(abs(rounding[:6]))),physical_curvature_cause_established=False)
    return vectors,stats

class Phase:
    def __init__(self,data,maps):
        self.phase='preflight';self.preflight={};self.path={};self.taken=set();self.active=None
        for spec in registry():
            sid=spec['state_id'];case=spec['case'];self.preflight[('prepared',sid)]=(safe.tensor_hash(data['points'][sid]['prepared']),case,False)
            for arm in (ARMS[0],ARMS[1]):self.preflight[('endpoint',sid,arm)]=(safe.tensor_hash(data['maps'][(sid,arm)]['point']),case,True)
            for i in range(len(BETAS)):
                for rnd in (0,1):self.path[(rnd,sid,i)]=(safe.tensor_hash(maps[(sid,i)]),case,True)
    def authorize(self,key,point,case,write):
        need(self.active is None,'Previous model request not consumed');table=self.preflight if self.phase=='preflight' else self.path;tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(key in table and table[key]==expected and tag not in self.taken,'Unsealed, repeated or relabelled request');self.taken.add(tag);self.active=expected
    def consume(self,point,case,write):
        need(self.active==(safe.tensor_hash(point),case,write),'Model call outside authorized point');self.active=None
    def begin_measurements(self):
        need(self.phase=='preflight' and self.active is None and len(self.taken)==len(self.preflight),'All36 preflight controls required');self.phase='paths'
    def finish(self):need(self.phase=='paths' and self.active is None and len(self.taken)==len(self.preflight)+len(self.path),'All168 two-round coefficient measurements required')
