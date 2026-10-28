"""Fixed two-state FP64 physical-chord directional derivative diagnostic."""
from pathlib import Path
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor156'))
import core156 as parent
from core156 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM
PROTOCOL='GPT2_LC2_C157_FP64_DIRECTIONAL_AD_CPU_V1'
INPUT_KEYS=('C155','C156');FOCUS=parent.FOCUS;LABELS=parent.LABELS
SCALES=parent.SCALES;TIMEOUT=7200
REPLAY_TOL=2e-9
REQUESTED_FORWARDS=26;REQUESTED_BACKWARDS=16
def contract():
    return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),
        source156=clean_json((ROOT/'trusted/C156_SOURCE_PINS.json').read_bytes()),parent=parent.contract())
def design():
    return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),focus_training_state_ids=list(FOCUS),
        all84_remain_exposed_training=True,model='Original pinned GPT-2 CPU float32 initialization; frozen parameters/buffers cast exactly to FP64 for temporary functional replacement',
        path='For each fixed state: theta(alpha)=float64(C155 zero point)+alpha*(float64(C155 plus_one point)-float64(C155 zero point)). Fixed continuous FP64 physical chord, with no float32 cast inside its differentiable path.',
        path_boundary='This is NOT the original coefficient-update/write-map path: intermediate sealed C155 float32 points can differ. All seven original/chord coordinate differences are recorded before model loading. alpha1 uses the exact archived plus_one endpoint bytes cast to FP64.',
        scales=[h for h,_,_ in SCALES],labels=LABELS,AD='Only scalar alpha requires grad. Six scalar response-coordinate derivatives and two response-loss derivatives (own FP64 / cached FP32 reference) at alpha0; eight scalar reverse calls per state. No parameter Jacobian, parameter gradients, optimizer or writer fit.',
        response_loss='relu(max(abs(response))/.0018-1)^2 under each fixed reference. Original FP32 acceptance gates unchanged; FP64 response-only diagnostic has no qualification gates or KL/read/aggregate84 objective.',
        baseline='Native FP32 zero-chart initialization and one FP64 zero-chart baseline. Compare pinned model identity, cached FP32 response/logits and archived FP64 response/logits before probes. Never replace archived references to make replay pass.',
        order='All14 chord points, direction tensors and requests sealed before model initialization; baseline then four exact C155 endpoint replays (zero/+1, each state), then AD at zero and all seven finite panels plus zero/+1 repeats per state.',
        derivative_agreement='Report every signed/absolute AD-minus-FD coordinate/loss discrepancy at all three registered scales. No agreement threshold, best scale, precision superiority or derivative certification claim.',
        sampled_nonsmoothness='Record maximum-coordinate ties, signs and response-hinge activity at every panel; no between-point smoothness certificate.',
        counts=dict(model_forwards=26,scalar_backward_calls=16,FP64_baseline_forwards=1,native_initialization_forwards=1,endpoint_replay_forwards=4,AD_forwards=2,finite_panel_and_repeat_forwards=18),
        numeric_replay_tolerance=REPLAY_TOL,replay_tolerance_is_not_acceptance_or_derivative_threshold=True,
        original_response_budget=.002,original_KL_budget=.005,original_margin_budget=.05,
        worker_budget_seconds=TIMEOUT,scope='Two previously exposed fixed training states on a declared new continuous FP64 physical chord. No deployment, training, new feature acquisition, full84 FP64 objective, unseen confirmation or excluded-input acceptance. No AD-error/correctness certification, systematic-error bound, curvature/convergence or causal failure explanation. Archived source is data, not executed. Parent decisions and original FP32 gates retained.',
        candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False)
def point_hash(a):
    a=np.asarray(a);need(a.dtype==np.float64 and a.shape==(DIM,) and np.isfinite(a).all(),'Finite FP64 chart');return hashlib.sha256(a.tobytes()).hexdigest()
def chord(p0,p1,alpha):
    need(type(alpha) in (int,float) and alpha in LABELS.values(),'Registered scalar alpha')
    a=farray(np.asarray(p0),(DIM,));b=farray(np.asarray(p1),(DIM,));need(a.dtype==b.dtype==np.float32,'Archived physical endpoints must be FP32')
    return b.astype(np.float64) if alpha==1. else a.astype(np.float64)+float(alpha)*(b.astype(np.float64)-a.astype(np.float64))
def freeze(data,out):
    out=Path(out);points={};directions={};rows=[]
    for sid in FOCUS:
        z=data['points']['zero__'+sid];p=data['points']['plus_one__'+sid]
        delta=p.astype(np.float64)-z.astype(np.float64);need(np.any(delta!=0),'No arbitrary zero direction');directions[sid]=delta
        for label,alpha in LABELS.items():
            value=chord(z,p,alpha);points[(sid,label)]=value;old=data['points'][label+'__'+sid].astype(np.float64)
            rows.append(dict(state_id=sid,label=label,alpha=alpha,FP64_point_sha256=point_hash(value),C155_FP32_point_sha256=safe.tensor_hash(data['points'][label+'__'+sid]),original_point_difference_Linf=float(np.max(abs(value-old))),coordinates_different_from_archived=int(np.count_nonzero(value!=old)),displacement_from_zero_L2=float(np.linalg.norm(value-z.astype(float))),direction_L2=float(np.linalg.norm(delta)),direction_sha256=point_hash(delta)))
    np.savez_compressed(out/'ALL14_FP64_CHORD_POINTS.npz',**{sid+'__'+n:p for (sid,n),p in points.items()})
    np.savez_compressed(out/'TWO_FIXED_DIRECTIONS.npz',**directions);dump(out/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json',rows)
    requests=[dict(stage='endpoint_replay',state_id=sid,label=n) for sid in FOCUS for n in ('zero','plus_one')]
    for sid in FOCUS:
        requests.append(dict(stage='AD',state_id=sid,label='zero'))
        requests.extend(dict(stage='finite',state_id=sid,label=n) for n in LABELS)
        requests.extend(dict(stage='repeat',state_id=sid,label=n) for n in ('zero','plus_one'))
    dump(out/'REQUESTS_FROZEN.json',requests)
    names=['DESIGN_FROZEN_BEFORE_MODEL.json','ALL14_FP64_CHORD_POINTS.npz','TWO_FIXED_DIRECTIONS.npz','POINT_BINDINGS_AND_PATH_DIFFERENCES.json','REQUESTS_FROZEN.json']
    seals={n:sha(out/n) for n in names};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,model_forwards_so_far=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json')
    return points,directions,requests,seals
def losses(absolute,local,shared):
    a=farray(np.asarray(absolute,float),(6,));return dict(local=parent.response_loss(a-local),shared=parent.response_loss(a-shared))
def describe(response):
    r=farray(np.asarray(response,float),(6,));m=float(max(abs(r)));return dict(response_Linf=m,max_coordinates=np.flatnonzero(abs(r)==m).tolist(),coordinate_signs=np.sign(r).astype(int).tolist(),hinge_active=m>.0018,max_coordinate_tie=int(np.sum(abs(r)==m))>1)
def analyse(records,ad,local,shared):
    need(set(records)==set(LABELS),'Every finite panel retained');need(np.shape(ad['coordinate_derivatives'])==(6,),'Six directional scalars')
    output=[]
    for h,minus,plus in SCALES:
        z=np.asarray(records['zero']['absolute_response']);m=np.asarray(records[minus]['absolute_response']);p=np.asarray(records[plus]['absolute_response']);fd=parent.fd(z,m,p,h);central=np.array(fd['central']);d=np.asarray(ad['coordinate_derivatives'])
        loss_fd={}
        for ref,baseline in [('local',local),('shared',shared)]:
            l=parent.fd(parent.response_loss(z-baseline),parent.response_loss(m-baseline),parent.response_loss(p-baseline),h);a=ad['response_loss_derivatives'][ref]
            loss_fd[ref]=dict(**l,AD=a,central_minus_AD=float(l['central']-a),positive_minus_AD=float(l['positive']-a),negative_minus_AD=float(l['negative']-a))
        output.append(dict(**fd,AD=d.tolist(),central_minus_AD=(central-d).tolist(),central_minus_AD_Linf=float(max(abs(central-d))),positive_minus_AD=(np.array(fd['positive'])-d).tolist(),negative_minus_AD=(np.array(fd['negative'])-d).tolist(),response_loss=loss_fd))
    return dict(scales=output,panels={n:dict(local=describe(np.asarray(r['absolute_response'])-local),shared=describe(np.asarray(r['absolute_response'])-shared)) for n,r in records.items()},all_scales_retained=True,no_derivative_agreement_threshold=True)
