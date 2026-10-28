"""Complete frozen84-state FP64 objective/gradient baseline; no training or head selection."""
from pathlib import Path
from collections import Counter
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor159'))
import core159 as previous
from core159 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash
archive155=previous.previous.previous.parent.parent
arithmetic=archive155.prior
PROTOCOL='GPT2_LC2_C160_FULL84_FP64_BASELINE_CPU_V1'
INPUT_KEYS=('C155','C156','C157','C158','C159');N=84;LABELS=previous.LABELS;SCALES=previous.SCALES;TIMEOUT=7200
WITHIN_RUN_TOL=2e-9;MAP_TOL=1e-12
REQUESTED_FORWARDS=2019;REQUESTED_BACKWARDS=168

def registry():return archive155.registry()
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source159=clean_json((ROOT/'trusted/C159_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=registry(),training_states=84,all84_remain_exposed_training=True,anchor='Original C155 small_step12 cyclic_rate_1e4 BEFORE tensor, including all3x2x36 entries. Entire seven-panel original update direction retained, no optimizer or head update.',map=previous.design()['path'],boundary=previous.design()['rounding_boundary'],router='Frozen original84 prewrite weights; no route/feature recomputation.',FP64_reference='Primary response is unchanged archived C155 cached FP32 response cast exactly to FP64. KL uses unchanged archived C155 FP32 zero-baseline logits cast exactly to FP64, softmax/log_softmax recomputed in FP64. These are fixed before model load; fresh FP64 reference is descriptive only, never objective selection.',native_reference='Actual physical FP32 chart commits at alpha0, original archived FP32 response and logits, FP32 mean/subtraction/KL arithmetic. Fresh runtime baseline differences are reported and never substituted. All original content/response/KL/read/radius budgets retained; preparation validity is inherited archived provenance, not newly re-prepared.',loss=dict(response='R_i=relu(max(abs(response_i))/.0018-1)^2',read='mean(relu(1-signed_margin/.075)^2) over16 reads',KL='relu(KL/.0045-1)^2',radius='relu(norm(theta64-prepared64)/20-1)^2',regularizer='.001*sum(e^2)/rank',aggregate='J=.9*max(R)+.1*mean(R)+mean(read+KL+radius+regularizer); average exact maximizing ties for gradient aggregation.'),AD='One full FP64 endpoint pair per state at alpha0; separate response and summed-guard scalar backward to the complete3x2xrank coefficient tensor.168 scalar reverse calls. No model-parameter gradients, Jacobian or optimizer.',finite_difference='All84 at alpha0,±.25,±.5,±1 on continuous FP64 map, in multiples of the unnormalised original step12 deltaP; full objective J recomputed per panel. Retain central/one-sided slopes versus aggregate gradient dot original deltaP at all3 scales. No derivative agreement threshold or best scale.',order='Seal all588 current Torch FP64 points, original588 FP32 points, seven coefficients and full1008 state-request registry before model.3 initialization forwards; native alpha0 round0; all84 AD rows; seven full84 FP64 panels; full84 FP64 zero/+1 repeats; native alpha0 round1.',counts=dict(native_initialization_forwards=1,FP64_zero_baseline_forwards=2,native_FP32_metrics=168,FP64_AD_metrics=84,FP64_panel_metrics=588,FP64_repeat_metrics=168,state_metric_records=1008,state_requests=1008,total_model_forwards=2019,functional_FP64_forwards=1682,native_FP32_forwards_including_initialization=337,scalar_reverse_calls=168),within_run_FP64_tolerance=WITHIN_RUN_TOL,map_tolerance=MAP_TOL,native_repeat_tolerances=dict(observed=.0002,KL=5e-6,same_gates_required=True),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),no_FP64_acceptance_score=True,worker_budget_seconds=TIMEOUT,scope='Frozen full84 training-only full-objective and coefficient-gradient baseline, plus repeated native FP32 qualification of the unchanged archived anchor. No head training or update, fitting, schedule/step/route selection, new states, deployment, fresh confirmation or excluded-input test. No proof of derivative accuracy, discrete-FP32 derivative, error/curvature bound, optimizer causality, convergence or generalisation. All earlier decisions retained; LC2 remains open.',new_training_steps=0,candidate_selected=False,LC2_closed=False,LC2_single_fold_passed=False)

def parameter_hash(p):return array_digest(farray(np.asarray(p),p.shape))
def loss(o,kl,point,prepared,e,case):return arithmetic.loss(o,float(kl),point,prepared,e,case)
def objective(rows):
 compare([r['state_id'] for r in rows],[s['state_id'] for s in registry()],'All84 objective denominator and order');R=farray(np.asarray([r['objective_parts']['response_loss'] for r in rows],float),(N,));G=np.array([sum(v for k,v in r['objective_parts'].items() if k!='response_loss') for r in rows]);need(np.isfinite(G).all() and np.all(R>=0) and np.all(G>=0),'Finite nonnegative complete losses');top=R==R.max();weights=.1/N+.9*top/int(top.sum());return dict(objective=float(.9*R.max()+.1*R.mean()+G.mean()),maximum_response_loss=float(R.max()),mean_response_loss=float(R.mean()),mean_guard_loss=float(G.mean()),response_weights=weights.tolist(),guard_weights=[1/N]*N,maximizing_state_ids=[rows[i]['state_id'] for i in np.flatnonzero(top)],exact_max_tie_count=int(top.sum()))
def aggregate(rows,shape):
 a=objective(rows);rg=np.array([farray(np.asarray(r['response_gradient'],float),shape) for r in rows]);gg=np.array([farray(np.asarray(r['guard_gradient'],float),shape) for r in rows]);w=np.asarray(a['response_weights']);g=np.sum(w[:,None,None,None]*rg+gg/N,axis=0);return g,a

def constants(data):
 d=data['d'];P=data['before'];farray(P,(3,2,d['V'].shape[0]));need(P.dtype==np.float64 and data['after'].dtype==np.float64,'Archived FP64 head');c=dict(P0=P.copy(),deltaP=data['after']-P,V=d['V'].astype(np.float64).copy())
 compare(set(d['points']),{s['state_id'] for s in registry()},'Exact84 state inventory')
 for sp in registry():
  sid=sp['state_id'];q=d['points'][sid];w=farray(d['weights'][sid],(3,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14,'Original convex routing')
  for k in ('prepared','raw'):need(q[k].dtype==np.float32,'Archived FP32 physical state');c[sid+'__'+k]=farray(q[k],(DIM,)).astype(np.float64)
  c[sid+'__weights']=w.astype(np.float64).copy()
 return c

def requests():
 rr=[dict(stage='native0',state_id=s['state_id'],label='zero') for s in registry()]
 rr += [dict(stage='AD',state_id=s['state_id'],label='zero') for s in registry()]
 rr += [dict(stage='finite',state_id=s['state_id'],label=n) for n in LABELS for s in registry()]
 rr += [dict(stage='repeat',state_id=s['state_id'],label=n) for n in ('zero','plus_one') for s in registry()]
 rr += [dict(stage='native1',state_id=s['state_id'],label='zero') for s in registry()];return rr

def freeze(context,out):
 import torch
 from map159 import torch_constants,torch_map
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
 data=context['data'];c=constants(data);tc=torch_constants(c);out=Path(out);points={};native={};coefs={};rows=[]
 for n,a in LABELS.items():
  p=c['P0']+float(a)*c['deltaP'];need(np.max(abs(p))<=.125+1e-15,'Fixed coefficient box');coefs[n]=p
  for sp in registry():
   sid=sp['state_id'];want,_,_=previous.numpy_map(c,sid,a)
   with torch.no_grad():pt,pp,e=torch_map(tc,sid,torch.tensor(float(a),dtype=torch.float64))
   pt=pt.numpy().copy();err=float(max(abs(pt-want)));need(err<=MAP_TOL,'Independent complete84 Torch/NumPy map');points[(sid,n)]=pt;old=data['points'][n+'__'+sid];q=data['d']['points'][sid];original=arithmetic.proposal(q['prepared'],q['raw'],data['d']['V'],arithmetic.effective(data['coefs'][n],data['d']['weights'][sid],archive155.ARM));exact(old,original,'Exact original FP32 commit map');native[(sid,n)]=old.copy();rows.append(dict(state_id=sid,label=n,FP64_point_sha256=point_hash(pt),original_FP32_point_sha256=safe.tensor_hash(old),independent_numpy_map_Linf=err,FP64_vs_original_FP32_Linf=float(max(abs(pt-old.astype(float)))),new_rounded_map_vs_original_FP32_coordinates=int(np.count_nonzero(pt.astype(np.float32)!=old))))
 np.savez_compressed(out/'FROZEN_FULL84_CHAIN_CONSTANTS.npz',**c);np.savez_compressed(out/'ALL588_FP64_POINTS.npz',**{sid+'__'+n:p for (sid,n),p in points.items()});np.savez_compressed(out/'ALL588_ORIGINAL_FP32_POINTS.npz',**{sid+'__'+n:p for (sid,n),p in native.items()});np.savez_compressed(out/'SEVEN_COEFFICIENT_TENSORS.npz',**coefs);np.savez_compressed(out/'FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz',response=np.asarray(context['baseline']['FP32_cached_response_reference'],float),baseline_logits=context['baseline_logits']['FP32_logits_cast64']);dump(out/'POINT_BINDINGS.json',rows);dump(out/'ALL1008_REQUESTS_FROZEN.json',requests());names=['DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_FULL84_CHAIN_CONSTANTS.npz','ALL588_FP64_POINTS.npz','ALL588_ORIGINAL_FP32_POINTS.npz','SEVEN_COEFFICIENT_TENSORS.npz','FIXED_ARCHIVED_OBJECTIVE_REFERENCES.npz','POINT_BINDINGS.json','ALL1008_REQUESTS_FROZEN.json'];seals={n:sha(out/n) for n in names};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,model_forwards_so_far=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');return c,points,native,coefs,seals

def native_summary(rows):
 compare([r['state_id'] for r in rows],[s['state_id'] for s in registry()],'All84 native denominator');bad=Counter(k for r in rows for k in r['qualification']['failed_gates']);return dict(n_requested=N,n_evaluated=len(rows),passed=sum(r['qualification']['complete_pass'] for r in rows),failed_ids=[r['state_id'] for r in rows if not r['qualification']['complete_pass']],failed_gates=dict(bad),maximum_response_drift=max(r['metrics']['response_drift'] for r in rows))
