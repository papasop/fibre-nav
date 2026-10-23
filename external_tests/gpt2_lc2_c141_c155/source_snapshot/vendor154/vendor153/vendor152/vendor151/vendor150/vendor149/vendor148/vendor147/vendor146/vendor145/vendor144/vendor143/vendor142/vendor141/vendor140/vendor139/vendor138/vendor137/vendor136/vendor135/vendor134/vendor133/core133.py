"""Independent arithmetic for the fixed C132 finite-write decomposition."""
from pathlib import Path
import hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'support'))
import audit124 as safe
from audit124 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest

PROTOCOL='GPT2_LC2_C133_FINITE_STEP_DECOMPOSITION_CPU_V1'
INPUT_KEYS=('C132',)
PARENT='GPT2_LC2_C132_SHARED_KERNEL_FEASIBILITY_CPU_V1'
ARMS=('learned_raw','shared_response_projected','shared_projected_read_preserved')
ROLES=('all12','geometry_training','historical_seen_center_probe')
GATES=('preparation','bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin','proposal_radius')
DIM=36864;RADIUS=20.;BUDGET=.002;TIMEOUT=1200;ALG_TOL=1e-8

def registry():
    return [dict(state_id=f'{c}__{o.lower()}_training_center{i:02d}',case=c,origin=o,role=('geometry_training' if i==0 else 'historical_seen_center_probe')) for c in ('xor_00','xor_01','xor_11') for o in ('C82','C90') for i in range(2)]

def contract():
    return dict(production=True,pin=clean_json((ROOT/'INPUT_PINS.json').read_bytes())['C132'],source=clean_json((ROOT/'trusted/C132_SOURCE_PINS.json').read_bytes()))

def design():
    return dict(protocol=PROTOCOL,input='One whole-hash-pinned original C132 scientific ZIP',registry=registry(),arms=list(ARMS),rounds=2,
        endpoint_records_requested=72,local_decomposition_records_requested=36,arm_vs_raw_contrasts_requested=24,
        local_derivative_scope='Only the six geometry-training centers have local10-row Jacobians. No derivative is imputed to the six other centers.',
        identity='observed_endpoint = observed_prewrite + J_local @ actual_committed_delta + finite_step_remainder',
        linear_stages='pre-cap proposal; after radius cap; actual float32 displacement; cap and rounding effects are linear diagnostics only',
        contrast='endpoint_arm-endpoint_raw = J_local@(delta_arm-delta_raw) + remainder_arm-remainder_raw',
        cancellation='Coordinatewise |linear|+|remainder|-|finite change|, descriptive only; not proof of a physical mechanism',
        response_budget=BUDGET,KL_budget=.005,read_margin=.05,proposal_radius=RADIUS,
        arithmetic_tolerance=1e-9,operator_tolerance=ALG_TOL,
        branch_rule='Among evaluated projected geometry endpoints: all first-order endpoints within budget but finite failures -> FIRST_ORDER_ENDPOINT_WITHIN_BUDGET_FINITE_ENDPOINT_FAILURE; none -> LINEAR_ACTION_OR_BASE_RESIDUAL_ALREADY_OUTSIDE_BUDGET; otherwise MIXED_LINEAR_AND_FINITE_STEP_FAILURE. All finite passes -> NO_PROJECTED_GEOMETRY_FAILURE_RECORDED.',
        finite_step_remainder_is_not_pure_curvature=True,AD_accuracy_independently_checked=False,forward_accuracy_bound_established=False,
        new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,
        archived_source_executed=False,candidate_selected=False,thresholds_changed=False,LC2_closed=False,worker_budget_seconds=TIMEOUT)

def gate_record(m,case,point,prepared):
    g=safe.metric_gates(m,case,point,True,True)
    radius=float(np.linalg.norm(point.astype(float)-prepared.astype(float)));valid=radius<=RADIUS+1e-5
    g.update(actual_delta_norm=radius,proposal_radius_passed=bool(valid),complete_pass=bool(g['complete_pass'] and valid))
    if not valid:g['failed_gates'].append('proposal_radius')
    return g

def reconstruct(theta,raw,arm,op,available):
    """Reconstruct the documented map independently; do not execute C132 source."""
    d=raw.astype(np.float64)-theta.astype(np.float64)
    if arm==ARMS[0]:return raw.copy(),d,d,dict(available=True,radius_active=False,delta_norm=float(np.linalg.norm(d)),source='Original exact archived proposal')
    if arm==ARMS[2] and not available:return None,None,None,dict(available=False,reason='COMMON_READ_RANK_OR_ALGEBRA_UNAVAILABLE')
    v,a,b,stack=(op[k] for k in ('V','A','B','response_stack'))
    projected=d-v.T@(v@d)
    u=projected.copy()
    if arm==ARMS[2]:u+=b@(a@d-a@projected)
    norm=float(np.linalg.norm(u));scale=min(1.,RADIUS/max(norm,1e-30));limited=u*scale
    point=(theta.astype(np.float64)+limited).astype(np.float32);actual=point.astype(float)-theta.astype(float)
    diag=dict(available=True,radius_active=bool(scale<1),preclip_norm=norm,clip_factor=scale,delta_norm=float(np.linalg.norm(actual)),
        float32_commit_error_Linf=float(np.max(abs(actual-limited))),preclip_stacked_response_action_Linf=float(np.max(abs(stack@u))),
        postclip_stacked_response_action_Linf=float(np.max(abs(stack@limited))),actual_stacked_response_action_Linf=float(np.max(abs(stack@actual))),
        preclip_common_read_action_change_Linf=float(np.max(abs(a@u-a@d))),postclip_common_read_action_change_Linf=float(np.max(abs(a@limited-a@d))),
        actual_common_read_action_change_Linf=float(np.max(abs(a@actual-a@d))),first_order_quantities_are_not_finite_endpoint_bounds=True)
    return point,u,limited,diag

def vector_stats(v):
    a=np.asarray(v,float);return dict(Linf=float(np.max(abs(a))),L2=float(np.linalg.norm(a)))

def decompose(j,prepared,point,prewrite,endpoint,precap,postcap):
    """Residual of archived differential prediction, not a curvature certificate."""
    j=farray(np.asarray(j),(10,DIM));d=point.astype(float)-prepared.astype(float)
    before=farray(np.asarray(prewrite,float),(10,));after=farray(np.asarray(endpoint,float),(10,));change=after-before
    lp=j@precap;lc=j@postcap;la=j@d;rem=change-la
    cap=j@(postcap-precap);rounding=j@(d-postcap)
    close(lp+cap+rounding,la,'Linear stage decomposition',1e-9)
    close(before+la+rem,after,'Endpoint arithmetic closure',1e-9)
    vals=dict(prewrite=before,endpoint=after,finite_change=change,linear_precap=lp,linear_postcap=lc,linear_actual=la,
        linear_cap_effect=cap,linear_float32_commit_effect=rounding,finite_step_remainder=rem,
        first_order_endpoint=before+la,coordinate_cancellation_amount=abs(la)+abs(rem)-abs(change))
    for v in vals.values():need(np.isfinite(v).all(),'Nonfinite decomposed component')
    stats={k:dict(response=vector_stats(v[:6]),exact_read=vector_stats(v[6:])) for k,v in vals.items()}
    stats.update(first_order_response_within_budget=bool(np.max(abs((before+la)[:6]))<=BUDGET),finite_response_within_budget=bool(np.max(abs(after[:6]))<=BUDGET),
        response_opposed_coordinates=int(np.sum(la[:6]*rem[:6]<0)),arithmetic_closure_Linf=float(np.max(abs(before+la+rem-after))))
    return vals,stats

def contrast(raw,arm):
    actual=arm['endpoint']-raw['endpoint'];linear=arm['linear_actual']-raw['linear_actual'];remainder=arm['finite_step_remainder']-raw['finite_step_remainder']
    error=close(linear+remainder,actual,'Arm-vs-raw decomposition',1e-9)
    return dict(endpoint_difference=actual,linear_action_difference=linear,remainder_difference=remainder),dict(arithmetic_closure_Linf=error,response_endpoint_difference_Linf=float(np.max(abs(actual[:6]))),response_linear_difference_Linf=float(np.max(abs(linear[:6]))),response_remainder_difference_Linf=float(np.max(abs(remainder[:6]))))

def endpoint_summaries(records):
    from collections import Counter
    summaries={};pairs={}
    for role in ROLES:
        summaries[role]={};pairs[role]={}
        for rnd in (0,1):
            summaries[role][str(rnd)]={};pairs[role][str(rnd)]={}
            for arm in ARMS:
                q=[r for r in records if r['arm']==arm and r['round']==rnd and (role=='all12' or r['role']==role)];ev=[r for r in q if r['status']=='EVALUATED'];bad=Counter(g for r in ev for g in r['recomputed_gates']['failed_gates'])
                summaries[role][str(rnd)][arm]=dict(n_requested=len(q),n_evaluated=len(ev),n_unavailable=len(q)-len(ev),passed=sum(r['recomputed_gates']['complete_pass'] for r in ev),maximum_response_drift=max((r['metrics']['response_drift'] for r in ev),default=None),failed_gates={g:bad[g] for g in GATES},failed_ids=[r['state_id'] for r in ev if not r['recomputed_gates']['complete_pass']])
                # C132 pairs all arms, including raw with itself.
                raw={r['state_id']:r for r in records if r['arm']==ARMS[0] and r['round']==rnd and (role=='all12' or r['role']==role) and r['status']=='EVALUATED'}
                matches=[r for r in ev if r['state_id'] in raw]
                pairs[role][str(rnd)][arm]=dict(n_requested=len(q),n_matching_evaluated=len(matches),rescued_ids=[r['state_id'] for r in matches if r['recomputed_gates']['complete_pass'] and not raw[r['state_id']]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in matches if not r['recomputed_gates']['complete_pass'] and raw[r['state_id']]['recomputed_gates']['complete_pass']])
    return summaries,pairs

def choose_branch(records):
    q=[r for r in records if r['arm']!=ARMS[0] and r['status']=='DECOMPOSED' and not r['statistics']['finite_response_within_budget']]
    n=sum(r['statistics']['first_order_response_within_budget'] for r in q)
    if not q:return 'NO_PROJECTED_GEOMETRY_FAILURE_RECORDED'
    if n==len(q):return 'FIRST_ORDER_ENDPOINT_WITHIN_BUDGET_FINITE_ENDPOINT_FAILURE'
    if n==0:return 'LINEAR_ACTION_OR_BASE_RESIDUAL_ALREADY_OUTSIDE_BUDGET'
    return 'MIXED_LINEAR_AND_FINITE_STEP_FAILURE'
