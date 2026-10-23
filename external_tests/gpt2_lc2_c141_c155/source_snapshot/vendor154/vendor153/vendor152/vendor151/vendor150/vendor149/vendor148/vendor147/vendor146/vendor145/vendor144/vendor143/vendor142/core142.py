"""C142 pure-NumPy archival audit contract and independently expressed gates."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor141'))
import core141 as prior
from core141 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM
PROTOCOL='GPT2_LC2_C142_READONLY_CONFIRMATION_AUDIT_CPU_V1'
PARENT=prior.PROTOCOL
PARENT_DECISION='C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE'
ARMS=prior.ARMS
TIMEOUT=1200

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source141=clean_json((ROOT/'trusted/C141_R1_SOURCE_PINS.json').read_bytes()))

def design():return dict(protocol=PROTOCOL,parent_protocol=PARENT,input_archives=['C141','C140'],states=48,preparation_records=96,endpoint_records=288,rounds=2,arms=list(ARMS),norms=[.125,.375],acceptance_budgets=prior.design()['acceptance_budgets'],scope='Archive/tensor/arithmetic audit only. Same48 states in two rounds; norm strata are descriptive, cannot replace all48 confirmation. C141 failure retained. No model replay, fit, solver, derivatives, candidate/radius selection or LC2 claim.',worker_budget_seconds=TIMEOUT)

def native_gates(m,case,point,write):
    need(case in ('xor_00','xor_01','xor_11') and type(write) is bool,'Invalid/excluded case or write role')
    o=farray(np.asarray(m['observed'],float),(22,));kl=m['KL']
    need(type(kl) in (int,float) and np.isfinite(kl) and kl>=0,'Invalid KL')
    truth=[int(case[-2]),int(case[-1]),1,int(case[-2])^int(case[-1])]
    if not write:truth[3]=1-truth[3]
    bits=(o[6:10]>=0).astype(int).tolist();margins=(2*np.tile(truth,4)-1)*o[6:];drift=float(np.max(np.abs(o[:6])))
    compare(m['theta_sha256'],safe.tensor_hash(point),'Coordinate/metric binding')
    compare(m['bits'],bits,'Bits recomputed');close(m['exact_margins'],margins[:4],'Exact signed margins',1e-10);close(m['development_margins'],margins[4:],'Development signed margins',1e-10);compare(m['response_drift'],drift)
    gates=dict(bits_match=bits==truth,response=bool(drift<=.002),KL=bool(kl<=.005),exact_read_margin=bool(np.all(margins[:4]>=.05)),development_read_margin=bool(np.all(margins[4:]>=.05)))
    if write:gates.update(output_flip=bits[3]!=(1-(int(case[-2])^int(case[-1]))),unselected_preserved=bits[:3]==[int(case[-2]),int(case[-1]),1])
    compare(m['gates'],gates,'Native gates independently recomputed');compare(m['pass'],bool(all(gates.values())))
    if 'per_anchor_KL' in m:
        a=farray(np.asarray(m['per_anchor_KL'],float),(12,))
        # Native float32 KL summands may be slightly negative; only the archived aggregate is clamped nonnegative.
    return gates,drift,margins

def endpoint_gates(m,case,point,prepared,valid):
    g,drift,margin=native_gates(m,case,point,True);order=('bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin')
    radius=float(np.linalg.norm(point.astype(float)-prepared.astype(float)));within=radius<=20.+1e-5
    failed=[k for k in order if not g[k]]+([] if within else ['proposal_radius'])
    if not valid:failed.insert(0,'preparation')
    return dict(complete_pass=bool(not failed),failed_gates=failed,response_drift=drift,response_residual=np.asarray(m['observed'][:6],float).tolist(),KL=m['KL'],minimum_exact_margin=float(np.min(margin[:4])),minimum_development_margin=float(np.min(margin[4:])),actual_delta_norm=radius,proposal_radius_passed=bool(within))

def repeat_check(a,b):
    e=float(np.max(np.abs(np.array(a['observed'])-np.array(b['observed']))));k=abs(a['KL']-b['KL']);same=a['gates']==b['gates']
    return dict(observed_error_Linf=e,KL_absolute_error=k,gate_identity=same,same_point_identity=a['theta_sha256']==b['theta_sha256'],within_replay_tolerance=bool(e<=.0002 and k<=5e-6 and same))

def summary_tables(records):
    result={};pairs={}
    groups={'all48':48,'xor_00':16,'xor_01':16,'xor_11':16,'norm_0.125':24,'norm_0.375':24}
    for group,n in groups.items():
        result[group]={};pairs[group]={}
        for rnd in (0,1):
            rr=[r for r in records if r['round']==rnd and (group=='all48' or r['case']==group or group=='norm_'+str(r['norm']))];by={(r['state_id'],r['arm']):r for r in rr};result[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in ARMS:
                rows=[r for r in rr if r['arm']==arm];need(len(rows)==n,'Missing requested state/arm/stratum')
                result[group][str(rnd)][arm]=dict(n_requested=n,n_evaluated=n,passed=sum(r['recomputed_gates']['complete_pass'] for r in rows),failed_ids=[r['state_id'] for r in rows if not r['recomputed_gates']['complete_pass']],failed_gates=dict(sorted(Counter(k for r in rows for k in r['recomputed_gates']['failed_gates']).items())),maximum_response_drift=max(r['metrics']['response_drift'] for r in rows),within_C140_training_response_safety=sum(r['metrics']['response_drift']<=.00125 for r in rows));pairs[group][str(rnd)][arm]={}
                for control in ARMS[:2]:pairs[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in rows if r['recomputed_gates']['complete_pass'] and not by[(r['state_id'],control)]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in rows if not r['recomputed_gates']['complete_pass'] and by[(r['state_id'],control)]['recomputed_gates']['complete_pass']])
    return result,pairs

def residual_diagnostic(raw,candidate,delta):
    r=farray(np.asarray(raw,float),(6,));c=farray(np.asarray(candidate,float),(6,));d=farray(np.asarray(delta,float),(DIM,))
    effect=c-r;required=-r;dot=float(r@effect);normr=float(np.linalg.norm(r));norme=float(np.linalg.norm(effect))
    return dict(raw_response=r.tolist(),candidate_response=c.tolist(),measured_finite_response_effect=effect.tolist(),required_effect_for_zero_response=required.tolist(),remaining_response=c.tolist(),raw_Linf=float(np.max(abs(r))),candidate_Linf=float(np.max(abs(c))),measured_effect_Linf=float(np.max(abs(effect))),response_budget_excess=max(0.,float(np.max(abs(c)))-.002),cancellation_projection_fraction=(-dot/(normr*normr) if normr>0 else None),cancellation_cosine=(-dot/(normr*norme) if normr>0 and norme>0 else None),committed_correction_L2=float(np.linalg.norm(d)),committed_correction_Linf=float(np.max(abs(d))),identity_closure_Linf=float(np.max(abs(r+effect-c))),causal_derivative_or_control_accuracy_bound_established=False)
