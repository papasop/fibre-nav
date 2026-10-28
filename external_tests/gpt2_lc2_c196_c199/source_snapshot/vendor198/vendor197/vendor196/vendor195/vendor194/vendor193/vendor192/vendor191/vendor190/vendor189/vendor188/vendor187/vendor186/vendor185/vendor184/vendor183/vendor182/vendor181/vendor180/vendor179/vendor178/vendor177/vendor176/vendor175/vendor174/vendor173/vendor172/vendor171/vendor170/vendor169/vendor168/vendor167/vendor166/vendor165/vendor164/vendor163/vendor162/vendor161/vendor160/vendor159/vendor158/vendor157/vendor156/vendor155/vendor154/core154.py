"""Registered native-forward directional diagnostic; all84 exposed training states."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor153'))
import core153 as prior
from core153 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,native_gates,endpoint_gates,repeat_check
PROTOCOL='GPT2_LC2_C154_FIXED_DIRECTION_CPU_V1'
INPUT_KEYS=('C153',)+prior.INPUT_KEYS;N=84;TIMEOUT=7200
ARM='cyclic_rate_1e4';ANCHORS=(('small_step01',1),('small_step12',12))
PROBES=(('minus_quarter',-.25),('plus_quarter',.25),('minus_half',-.5),('plus_half',.5),('minus_one',-1.),('plus_one',1.))
LABELS=dict(zero=0.,**dict(PROBES));HALVES=(.25,.5,1.)
REQUESTED_METRICS=1512;REQUESTED_FORWARDS=3025

def registry():return prior.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),audit153=prior.contract(),source153=clean_json((ROOT/'trusted/C153_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),anchor_registry=[dict(anchor=a,C152_arm=ARM,step=k,position='before archived update',direction='exact archived after minus before, no normalization') for a,k in ANCHORS],training_registry=registry(),all84_are_exposed_training=True,coefficient_stencil=LABELS,probe_order=[n for n,_ in PROBES],repeat_points=['zero','plus_one'],interpolation='alpha0 uses archived before bytes; alpha1 uses archived after bytes; others float64(before+alpha*(after-before)). No projection or adaptive radius. Any coefficient outside original .125 box stops before model.',objective='Original C152 J=.9*max(R)+.1*mean(R)+mean(G), all84; R=relu(max(abs(response))/.0018-1)^2; G retains read .075, KL .0045, radius20, .001*sum(e^2)/36.',map='Original frozen router/V and explicit-left float64 cyclic coupled map; commit float32. No new addressing, features or data.',preflight='Both zero panels, all168 native metrics, must match archived functional observations <=2e-6 and KL<=2e-7 before probes. First anchor additionally replays C152 C150 native baseline.',replay='Both zero and plus-one panels repeated after all1008 probes. Original native repeat tolerances and gates retained. Last-anchor plus-one replays original C152 step12 endpoints; first-anchor plus-one is compared to archived step02 functional observations.',counts=dict(unique_coefficient_panels=14,unique_points=1176,preflight_metrics=168,probe_metrics=1008,repeat_metrics=336,total_metrics=1512,model_forwards_including_initialization=3025,new_backward_calls=0),frozen_before_model=True,acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,radius=20.,radius_rounding_allowance=1e-5),native_functional_parity=dict(observed=2e-6,KL=2e-7),worker_budget_seconds=TIMEOUT,scope='Training-only fixed native-forward stencil. Archived gradients remain unverified derivatives. Central and one-sided finite differences at three finite scales, maximizer/hinge switches and float32 point changes are descriptive; no AD error, curvature bound, systematic-error bound, convergence, feasibility or architecture failure is certified. No pass/fail tolerance for derivative agreement is invented. Probes are not selected, deployed or used for online repair. No new states, fitting, backward, optimizer update, Jacobian, SVD, teacher, backtracking, confirmation or excluded-input evaluation. All requested points retained. C153 remains an archive audit, not an independent model replay.',candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False)

def interpolate(before,after,alpha):
    need(type(alpha) in (int,float) and float(alpha) in LABELS.values(),'Registered alpha only');farray(before,(3,2,before.shape[-1]));farray(after,before.shape)
    need(before.dtype==after.dtype==np.float64,'Original float64 coefficient tensors required');p=before.copy() if alpha==0 else after.copy() if alpha==1 else before+float(alpha)*(after-before)
    need(np.max(abs(p),initial=0)<=.125+1e-15,'Registered stencil outside original coefficient box');return p

def request_order():
    pre=[(a,'zero',sp['state_id'],'preflight') for a,_ in ANCHORS for sp in registry()]
    probes=[(a,n,sp['state_id'],'probe') for a,_ in ANCHORS for n,_ in PROBES for sp in registry()]
    repeats=[(a,n,sp['state_id'],'repeat') for a,_ in ANCHORS for n in ('zero','plus_one') for sp in registry()]
    return pre+probes+repeats

def freeze_stencil(d,anchors,out):
    out=Path(out);params={};points={};rows=[];specs=registry()
    compare(list(anchors),[a for a,_ in ANCHORS],'Both registered anchors')
    for a,_ in ANCHORS:
        h=anchors[a];farray(h['before'],(3,2,d['V'].shape[0]));farray(h['after'],h['before'].shape);need(np.any(h['before'][:,1,:]!=0),'Nonzero archived coupled head');need(np.any(h['after']!=h['before']),'Nonzero archived actual update')
        for label,alpha in LABELS.items():
            p=interpolate(h['before'],h['after'],alpha);params[(a,label)]=p
            for sp in specs:
                sid=sp['state_id'];q=d['points'][sid];e=prior.effective(p,d['weights'][sid],ARM);point=prior.proposal(q['prepared'],q['raw'],d['V'],e);farray(point,(DIM,));points[(a,label,sid)]=point;baseline=points[(a,'zero',sid)]
                rows.append(dict(anchor=a,label=label,alpha=alpha,**sp,parameters_digest=row_digest(p),point_sha256=safe.tensor_hash(point),same_float32_point_as_zero=bool(np.array_equal(point,baseline)),point_change_Linf=float(np.max(abs(point.astype(float)-baseline.astype(float)))),coefficient_interpolation_roundoff_Linf=float(np.max(abs(p-(h['before']+alpha*(h['after']-h['before'])))))))
    need(len(points)==1176 and len(params)==14,'Full fixed stencil inventory');np.savez_compressed(out/'STENCIL_PARAMETERS.npz',**{a+'__'+n:p for (a,n),p in params.items()});np.savez_compressed(out/'ALL1176_STENCIL_POINTS.npz',**{a+'__'+n+'__'+sid:p for (a,n,sid),p in points.items()});dump(out/'ALL1176_STENCIL_POINT_BINDINGS.json',rows);dump(out/'ALL1512_REQUESTS_FROZEN.json',[dict(anchor=a,label=n,state_id=sid,phase=stage) for a,n,sid,stage in request_order()]);seals={n:sha(out/n) for n in ['DESIGN_FROZEN_BEFORE_MODEL.json','STENCIL_PARAMETERS.npz','ALL1176_STENCIL_POINTS.npz','ALL1176_STENCIL_POINT_BINDINGS.json','ALL1512_REQUESTS_FROZEN.json']};dump(out/'STENCIL_FROZEN_BEFORE_MODEL.json',dict(artifact_seals=seals,unique_points=1176,requests=1512,model_forwards_so_far=0,no_probe_selected=True));return params,points,rows,seals

class Phase:
    def __init__(self,points):self.points=points;self.ids={k:safe.tensor_hash(p) for k,p in points.items()};self.order=request_order();self.at=0;self.active=None;self.phase='preflight'
    def authorize(self,a,n,sid,stage):
        need(self.at<len(self.order) and self.active is None and self.order[self.at]==(a,n,sid,stage),'Fixed complete request order');self.phase='preflight' if stage=='preflight' else 'paths';need(safe.tensor_hash(self.points[(a,n,sid)])==self.ids[(a,n,sid)],'Actual frozen stencil point changed');self.active=(self.ids[(a,n,sid)],sid.split('__')[0],True)
    def consume(self,p,case,write):need(self.active==(safe.tensor_hash(p),case,write),'Unauthorized native stencil point');self.active=None;self.at+=1
    def verify(self):
        need(set(self.points)==set(self.ids),'Actual frozen point inventory changed')
        for k,p in self.points.items():compare(safe.tensor_hash(p),self.ids[k],'Actual frozen point/'+str(k))
    def finish(self):need(self.at==1512 and self.active is None,'All1512 native metrics required')

def objective_panel(rows,params,d):
    compare([r['state_id'] for r in rows],[sp['state_id'] for sp in registry()],'All84 ordered panel records');R=[];G=[];active=[]
    for sp,r in zip(registry(),rows):
        sid=sp['state_id'];q=d['points'][sid];e=prior.effective(params,d['weights'][sid],ARM);point=prior.proposal(q['prepared'],q['raw'],d['V'],e);part=prior.loss(r['metrics']['observed'],r['metrics']['KL'],point,q['prepared'],e,sp['case']);compare(r['objective_parts'],part,'Recomputed actual native scalar parts');R.append(part['response_loss']);G.append(sum(v for k,v in part.items() if k!='response_loss'));o=np.asarray(r['metrics']['observed'][:6]);active.append(dict(state_id=sid,response_max_coordinates=np.flatnonzero(abs(o)==np.max(abs(o))).tolist(),response_hinge_active=part['response_loss']>0,read_hinge_active=part['read_loss']>0,KL_hinge_active=part['KL_loss']>0,radius_hinge_active=part['radius_loss']>0))
    R=np.array(R);G=np.array(G);rm=float(.9*R.max()+.1*R.mean());return dict(objective=rm+float(G.mean()),response_objective=rm,mean_guard=float(G.mean()),maximum_response_loss=float(R.max()),maximizing_state_ids=[registry()[i]['state_id'] for i in np.flatnonzero(R==R.max())],passed_native_gates=sum(r['gates']['complete_pass'] for r in rows),maximum_response_drift=max(r['metrics']['response_drift'] for r in rows),active_sets=active)

def finite_difference(zero,minus,plus,h,prediction):
    need(h in HALVES,'Registered FD scale');v=np.array([zero,minus,plus,h,prediction],float);need(np.isfinite(v).all(),'Finite FD values');central=(plus-minus)/(2*h);return dict(h=h,central_slope=central,positive_slope=(plus-zero)/h,negative_slope=(zero-minus)/h,recorded_gradient_prediction=prediction,central_minus_recorded_prediction=central-prediction,AD_accuracy_certified=False,derivative_agreement_acceptance_rule=None)
