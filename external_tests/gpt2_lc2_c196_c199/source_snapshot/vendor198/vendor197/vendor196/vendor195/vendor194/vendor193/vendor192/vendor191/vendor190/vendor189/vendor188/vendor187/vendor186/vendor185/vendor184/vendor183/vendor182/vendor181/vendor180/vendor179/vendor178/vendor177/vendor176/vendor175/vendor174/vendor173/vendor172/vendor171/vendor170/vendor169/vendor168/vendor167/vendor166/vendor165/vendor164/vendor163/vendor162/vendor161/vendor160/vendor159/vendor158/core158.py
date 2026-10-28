"""Independent same-run FP64 diagnostic; C157 failure and original LC2 gates remain unchanged."""
from pathlib import Path
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor157'))
import core157 as previous
from core157 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,chord,point_hash,describe
PROTOCOL='GPT2_LC2_C158_SAME_RUN_FP64_AD_CPU_V1'
INPUT_KEYS=('C155','C156','C157');FOCUS=previous.FOCUS;LABELS=previous.LABELS;SCALES=previous.SCALES
REFERENCES=('current_FP64','archived_FP64','archived_FP32');TIMEOUT=7200
WITHIN_RUN_TOL=2e-9
REQUESTED_FORWARDS=27;REQUESTED_BACKWARDS=18
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source157=clean_json((ROOT/'trusted/C157_R2_SOURCE_PINS.json').read_bytes()))
def design():
    return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),all84_remain_exposed_training=True,focus_state_ids=list(FOCUS),
        new_protocol_after_C157=True,C157_decision_retained='C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION',
        path='Same fixed physical chord as C157: exact FP64 casts of C155 zero/+1 physical endpoints; no float32 cast inside theta(alpha). Original intermediate C155 coefficient-map points may differ.',
        labels=LABELS,scales=[h for h,_,_ in SCALES],
        references=dict(current_FP64='First current-run FP64 all-zero LoRA-B chart baseline response; NOT the nonzero chord alpha0 output.',archived_FP64='Unchanged original C155 FP64 response reference',archived_FP32='Unchanged original C155 cached FP32 response reference'),
        reference_rule='Fixed before model loading: take first current FP64 zero-chart forward mean at the original six response IDs. Repeat once, save both full logits, freeze all three references and hashes before focus-state probes. No measured focus state is used to choose or recenter a reference.',
        cross_run='All FP32/FP64 baseline and four endpoint differences from original C155 are reported, including the historical 2e-9 comparison. These differences do not gate this separate same-run derivative experiment. They do not satisfy or replace C157 replay, writer acceptance or LC2 gates.',
        mandatory_checks='Original archive/source/tensor bindings and model identity; frozen physical/double parameters, buffers and references; finite FP64 actual forwards; fixed within-run zero-baseline repetition, endpoint/preflight, AD/no-grad center and final zero/+1 repetitions within 2e-9.',
        AD='Only scalar alpha requires grad. Six coordinate derivatives and three response-loss derivatives at alpha0 per state: nine scalar reverse calls per state. No parameter Jacobian, optimizer or fit.',
        response_loss='relu(max(abs(absolute_response-fixed_reference))/.0018-1)^2, separately for each of three references. No KL/read/full84 objective or FP64 qualification.',
        counts=dict(native_initialization_forwards=1,current_FP64_zero_chart_baseline_forwards=2,endpoint_preflight_forwards=4,AD_forwards=2,finite_panel_and_repeat_forwards=18,total_model_forwards=27,scalar_backward_calls=18,state_records=24),
        derivative_comparison='Retain all central/positive/negative finite slopes and signed discrepancies from AD at h=.25,.5,1. No best scale, derivative-agreement gate, AD certification or precision superiority claim.',
        within_run_tolerance=WITHIN_RUN_TOL,historical_C157_cross_run_tolerance=previous.REPLAY_TOL,LC2_acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05),
        protocol_change_disclosed=True,LC2_acceptance_thresholds_changed=False,worker_budget_seconds=TIMEOUT,
        scope='Separate training-only two-state numerical diagnostic in one newly initialized fixed runtime. Cross-archive drift is descriptive, not silently accepted as C157 replay. Fresh zero-chart FP64 reference is an additional condition, not replacement scoring for historical runs. No training, selection, deployment, unseen confirmation, excluded-class pass, full84 FP64 qualification, causal error explanation, accuracy/error/curvature bound or LC2 closure.',
        candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False)
def freeze(data,out):
    out=Path(out);points={};directions={};rows=[]
    for sid in FOCUS:
        z=data['points']['zero__'+sid];p=data['points']['plus_one__'+sid];d=p.astype(float)-z.astype(float);need(np.any(d!=0),'Nonzero archived direction required');directions[sid]=d
        for n,a in LABELS.items():
            value=chord(z,p,a);points[(sid,n)]=value;old=data['points'][n+'__'+sid].astype(float)
            rows.append(dict(state_id=sid,label=n,alpha=a,FP64_point_sha256=point_hash(value),original_C155_point_difference_Linf=float(max(abs(value-old))),direction_sha256=point_hash(d)))
    requests=[dict(stage='preflight',state_id=sid,label=n) for sid in FOCUS for n in ('zero','plus_one')]
    for sid in FOCUS:
        requests.append(dict(stage='AD',state_id=sid,label='zero'));requests.extend(dict(stage='finite',state_id=sid,label=n) for n in LABELS);requests.extend(dict(stage='repeat',state_id=sid,label=n) for n in ('zero','plus_one'))
    np.savez_compressed(out/'ALL14_FP64_CHORD_POINTS.npz',**{sid+'__'+n:p for (sid,n),p in points.items()});np.savez_compressed(out/'TWO_FIXED_DIRECTIONS.npz',**directions);dump(out/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json',rows);dump(out/'REQUESTS_FROZEN.json',requests)
    names=['DESIGN_FROZEN_BEFORE_MODEL.json','ALL14_FP64_CHORD_POINTS.npz','TWO_FIXED_DIRECTIONS.npz','POINT_BINDINGS_AND_PATH_DIFFERENCES.json','REQUESTS_FROZEN.json'];seals={n:sha(out/n) for n in names};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,model_forwards_so_far=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');return points,directions,requests,seals
def response_loss(r):return previous.parent.response_loss(r)
def analyse(panels,ad,references):
    compare(set(panels),set(LABELS),'All seven finite panels');compare(set(references),set(REFERENCES),'All three fixed references');d=farray(np.asarray(ad['coordinate_derivatives'],float),(6,));rows=[]
    for h,m,p in SCALES:
        z=np.asarray(panels['zero']['absolute_response'],float);minus=np.asarray(panels[m]['absolute_response'],float);plus=np.asarray(panels[p]['absolute_response'],float);fd=previous.parent.fd(z,minus,plus,h);losses={}
        for name in REFERENCES:
            b=np.asarray(references[name],float);v=previous.parent.fd(response_loss(z-b),response_loss(minus-b),response_loss(plus-b),h);a=ad['response_loss_derivatives'][name];losses[name]=dict(**v,AD=a,central_minus_AD=float(v['central']-a),positive_minus_AD=float(v['positive']-a),negative_minus_AD=float(v['negative']-a))
        rows.append(dict(**fd,AD=d.tolist(),central_minus_AD=(np.asarray(fd['central'])-d).tolist(),central_minus_AD_Linf=float(max(abs(np.asarray(fd['central'])-d))),positive_minus_AD=(np.asarray(fd['positive'])-d).tolist(),negative_minus_AD=(np.asarray(fd['negative'])-d).tolist(),response_loss=losses))
    return dict(scales=rows,panels={n:{ref:describe(np.asarray(row['absolute_response'])-references[ref]) for ref in REFERENCES} for n,row in panels.items()},all_scales_retained=True,all_references_retained=True,no_derivative_agreement_threshold=True)
