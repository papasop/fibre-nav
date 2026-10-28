"""Fixed late-position paired forward / precision diagnostic. No optimization."""
from pathlib import Path
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor154'))
import core154 as prev
from core154 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,endpoint_gates,repeat_check
prior=prev.prior
PROTOCOL='GPT2_LC2_C155_PAIRED_FORWARD_PRECISION_CPU_V1'
INPUT_KEYS=('C154',)+prev.INPUT_KEYS
ANCHOR='small_step12';ARM=prev.ARM;TIMEOUT=7200
LABELS=prev.LABELS;N=84
FOCUS=('xor_11__c141_confirmation_p09_n01_d00','xor_11__c141_confirmation_p08_n01_d00')
REQUESTED_METRICS=1530;REQUESTED_FORWARDS=3062
def registry():return prev.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prev.contract(),source154=clean_json((ROOT/'trusted/C154_SOURCE_PINS.json').read_bytes()))
def requests():
    return [('zero',s['state_id'],'preflight') for s in registry()]+[(n,s['state_id'],'probe') for n,_ in prev.PROBES for s in registry()]+[(n,s['state_id'],'repeat') for n in ('zero','plus_one') for s in registry()]
def design():
    return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),all84_are_exposed_training=True,anchor=ANCHOR,coefficient_source='C152 cyclic_rate_1e4 step12 UPDATE before/after; exact C154 stencil',alphas=LABELS,training_registry=registry(),focus_FP64_state_ids=list(FOCUS),focus_selection='Post-C154 diagnostic targets fixed before C155: late worst-response state and additional C152 failed state. No unseen or candidate-selection claim.',FP32='All84 at seven fixed signed panels, plus zero/+1 repeats; native physical commit followed by functional replacement on identical sealed float32 coordinates. Same pinned FP32 backbone, LoRA-A, buffers and zero-chart baseline.',FP64='Two focus states only; full FP32 parameters/buffers are cast to float64 for temporary functional calls. The already committed/sealed float32 chart is cast exactly to float64, not recomputed from a float64 write map. Integer/bool buffers are retained. All physical FP32 tensors are restored after every call.',FP64_baselines='One separate zero-chart FP64 anchor forward. Report own-precision response/KL and FP64 outputs against original cached FP32 response reference / FP32 baseline logits recomputed in FP64. Neither is a substitute acceptance metric. No full84 FP64 objective.',preflight='All84 native zero replay original C154 under original native repeat/gate rules. Native/functional observed <=2e-6, KL<=2e-7 required before signed probes. FP64 records have no pass/qualification gates.',rounding='Identical finite float32 chart values in every route; no quantization-free derivative claim. FP64 is a diagnostic condition, not ground truth or a certified error bound.',counts=dict(FP32_native_metrics=756,FP32_functional_metrics=756,FP64_focus_metrics=18,total_metric_records=1530,native_FP32_baseline_forwards=1,FP64_zero_chart_baseline_forwards=1,total_model_forwards=3062,backward_calls=0),acceptance_budgets=prev.design()['acceptance_budgets'],repeat='All routes repeat zero/+1. Native repeats use original tolerances; functional parity and FP64 changes are recorded separately. No best alpha/route/state is chosen.',worker_budget_seconds=TIMEOUT,scope='Training-only fixed late-direction forward diagnostics. All84 FP32 denominators retained. FP64 covers only two registered exposed states, with separate baseline provenance and coordinate trajectories, no full84 FP64 objective or qualification. No AD correctness/error certification, systematic-error bound, curvature bound, convergence or architecture-feasibility conclusion. No gradients, training, feature acquisition, router/head fitting, selection, deployment, preparation, confirmation or excluded-class evaluation. Original parent decision and FP32 acceptance remain unchanged.',LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False)

def point_key(label,sid):return ANCHOR+'__'+label+'__'+sid
def freeze(d,anchors,out,parent_points,parent_params):
    out=Path(out);h=anchors[ANCHOR];params={};points={};rows=[];need(np.any(h['after']!=h['before']) and np.any(h['before'][:,1,:]!=0),'Nonzero archived coupled head/direction required')
    for label,alpha in LABELS.items():
        p=prev.interpolate(h['before'],h['after'],alpha);exact(p,parent_params[ANCHOR+'__'+label],'Exact C154 coefficient tensor including inactive coordinates');params[label]=p
        for sp in registry():
            sid=sp['state_id'];q=d['points'][sid];e=prior.effective(p,d['weights'][sid],ARM);pt=prior.proposal(q['prepared'],q['raw'],d['V'],e)
            exact(pt,parent_points[point_key(label,sid)],'Exact C154 sealed float32 point');need(pt.dtype==np.float32,'Float32 physical coordinate required');points[(label,sid)]=pt
            rows.append(dict(label=label,state_id=sid,point_sha256=safe.tensor_hash(pt),coefficient_digest=prev.row_digest(p),alpha=alpha))
    need(len(points)==588,'All588 late point entries');np.savez_compressed(out/'ALL588_FIXED_POINTS.npz',**{n+'__'+sid:p for (n,sid),p in points.items()});np.savez_compressed(out/'COEFFICIENTS.npz',**params);dump(out/'POINT_BINDINGS.json',rows);dump(out/'ALL756_FIXED_REQUESTS.json',[dict(label=n,state_id=sid,stage=st,FP64_focus=sid in FOCUS) for n,sid,st in requests()]);names=['DESIGN_FROZEN_BEFORE_MODEL.json','ALL588_FIXED_POINTS.npz','COEFFICIENTS.npz','POINT_BINDINGS.json','ALL756_FIXED_REQUESTS.json'];seals={n:sha(out/n) for n in names};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,point_entries=588,FP32_native_functional_pairs=756,FP64_focus_metrics=18,model_forwards_so_far=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');return params,points,seals

class Phase:
    def __init__(self,points):self.points=points;self.ids={k:safe.tensor_hash(v) for k,v in points.items()};self.order=requests();self.at=0;self.active=None;self.phase='preflight'
    def authorize(self,label,sid,stage):
        need(self.active is None and self.at<len(self.order) and self.order[self.at]==(label,sid,stage),'Fixed C155 request order');self.phase='preflight' if stage=='preflight' else 'paths';compare(safe.tensor_hash(self.points[(label,sid)]),self.ids[(label,sid)],'Actual sealed point');self.active=(self.ids[(label,sid)],sid.split('__')[0],True)
    def consume(self,p,case,write):need(self.active==(safe.tensor_hash(p),case,write),'Unauthorized physical commit');self.active=None;self.at+=1
    def verify(self):
        compare(set(self.points),set(self.ids),'Fixed point inventory')
        for k,p in self.points.items():compare(safe.tensor_hash(p),self.ids[k],'Actual frozen point/'+str(k))
    def finish(self):need(self.at==756 and self.active is None,'All756 native pairs required')

def metric(observed,kl,point,case):
    o=farray(np.asarray(observed,float),(22,));need(np.isfinite(kl) and kl>=0,'Finite nonnegative KL');x,y=int(case[-2]),int(case[-1]);truth=[x,y,1,x^y];bits=(o[6:10]>=0).astype(int).tolist();m=(2*np.tile(truth,4)-1)*o[6:];R=float(np.max(abs(o[:6])));g=dict(bits_match=bits==truth,response=R<=.002,KL=kl<=.005,exact_read_margin=bool(np.all(m[:4]>=.05)),development_read_margin=bool(np.all(m[4:]>=.05)),output_flip=bits[3]!=1-(x^y),unselected_preserved=bits[:3]==[x,y,1]);return dict(observed=o.tolist(),KL=float(kl),bits=bits,response_drift=R,exact_margins=m[:4].tolist(),development_margins=m[4:].tolist(),gates=g,pass_=bool(all(g.values())),theta_sha256=safe.tensor_hash(point))
def full_metric(observed,kl,point,case):
    m=metric(observed,kl,point,case);m['pass']=m.pop('pass_');return m
def parity(a,b):return dict(observed_Linf=float(np.max(abs(np.array(a['observed'])-b['observed']))),KL_absolute=abs(a['KL']-b['KL']),gates_equal=a['gates']==b['gates'],same_point=a['theta_sha256']==b['theta_sha256'])
def coordinate_fd(z,m,p,h):
    need(type(h) in (int,float) and h in (.25,.5,1.),'Registered scale');z=farray(np.asarray(z,float),(22,));m=farray(np.asarray(m,float),(22,));p=farray(np.asarray(p,float),(22,));return dict(h=h,central=((p-m)/(2*h)).tolist(),positive=((p-z)/h).tolist(),negative=((z-m)/h).tolist(),response_max_coordinate_zero=np.flatnonzero(abs(z[:6])==max(abs(z[:6]))).tolist(),response_max_coordinate_minus=np.flatnonzero(abs(m[:6])==max(abs(m[:6]))).tolist(),response_max_coordinate_plus=np.flatnonzero(abs(p[:6])==max(abs(p[:6]))).tolist(),zero_response_hinge_active=bool(max(abs(z[:6]))>.0018),minus_response_hinge_active=bool(max(abs(m[:6]))>.0018),plus_response_hinge_active=bool(max(abs(p[:6]))>.0018))
