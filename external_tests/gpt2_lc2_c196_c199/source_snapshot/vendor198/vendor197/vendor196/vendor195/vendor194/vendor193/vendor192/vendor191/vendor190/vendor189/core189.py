"""Read-only independent arithmetic audit of the fixed C188 continuation."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor188'))
import core188 as prior
from core188 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
import core187 as independent
control=prior.control
PROTOCOL='GPT2_LC2_C189_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C188',)
TIMEOUT=3600
DECISION='C189_READONLY_CONTINUATION_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C189_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy thirty-four-original archival audit of C188 fixed eight-step all-coordinate offline continuation from the predetermined C186 final head. Independently bind all33 upstream originals and original C187 audit through delivered immutable source before re-auditing C188. Rebuild every1008-row LP from archived coordinate gradients using the independent C175 arithmetic checker, raw-primal mapping, exact recorded direction and actual binary64 endpoint-difference certificates, all eight fixed updates,840 saved FP64 points,1176 records, final FP64 repeat and336 native gates. Retain all84 trajectories over nine frames/eight transitions, guard-part decomposition,4032 response-coordinate affine residuals and actual cap violations. Independently reconstruct and bind every336 original precision-margin row, CSV and summary; report both precision response classifications without adding an FP64 full gate, and no assumed acceptance count or precision agreement. No model/Torch/HF/SciPy import, new forward/backward/fit/training/LP/solve/Jacobian/SVD, state preparation, confirmation or excluded input. Archived gradients, outputs and KL are not independently measured. Recorded arithmetic and precision agreement do not certify derivative truth, measurement accuracy, finite error bounds, solver optimality/infeasibility, chronology, non-access, saved-point consumption, curvature, convergence, causality or generalisation. Activity entry/exit/re-entry and1.8e-3 safety crossings reported for all84 states over all8 transitions, distinct from0.002 response gates and full native acceptance. Dynamic all84 endpoint accounting and full trajectories for the union of initial/final native failures, including rescues, destructions and remaining failures. No preset rescued count, failed count, monotonicity, crossing step or final class membership. Independently recompute168 native and84 FP64 cross-run C186-final/C188-initial replay pairs under original tolerances; distinguish point identity, exact observable equality and tolerance acceptance; retain all differences and compare training within the same run without recentring or hardware-cause claims. No head/step selection, deployment, changed thresholds or LC2 claim; original C188 decision retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source188=clean_json((ROOT/'trusted/C188_SOURCE_PINS.json').read_bytes()))
def no_compute():return independent.no_compute()
def design():
 d=independent.design()
 d.pop('C186_zip_hash_known_before_runtime',None)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C188_zip_hash_known_before_runtime=True,anchor='Exact original C186 predetermined final local step8; C188 contributes8 slots, cumulative C176/C178/C180/C182/C184/C186/C18856 since C170 fixed-final anchor, excluding C170 own eight slots.',arithmetic='All34 whole original ZIP hashes fixed before payload. All33 upstream originals and original C187 audit rebound through immutable C188 loader. Independent C175 problem/exact rational checker, not C174 production problem/certificate; all1008-row problems, raw-primal mapping,16 certificates and fixed binary640.001 updates. Solver-internal original count unknown/null; new solver calls0.',diagnostics='All84 states and all9 FP64 frames,8 transitions,4032 coordinate cap comparisons,168 paired full native outcomes and remaining failure trajectories. Reconstruct336 endpoint precision rows and CSV using independent arithmetic and original0.002 budget. Keep negative slacks, all disagreements, response-only vs full-native distinctions, initial/final summaries and regularizer/other guard parts. Record activity exits, entries, re-entries and safety crossings over all672 transitions, with endpoint native full gates; do not assume monotonicity, counts, causes or future-step requirements.',scope=SCOPE)
 d['counts'].update(endpoint_precision_rows_independently_reconstructed=336,activity_transition_rows=672,activity_state_trajectories=84,focus_full84_endpoint_accounting=84)
 d['counts'].update(native_anchor_replay_pairs=168,FP64_anchor_replay_pairs=84)
 d['cross_run_replay']='All84 states, both native rounds and first FP64 batch against original C186 final records; original native all22Linf<=0.0002,KL<=5e-6,equal metric gates and equal physical point hashes; FP64 all22Linf<=2e-9,KL<=2e-9. Report exact equality separately. No revised tolerances or acceptance gates.'
 d['focus']='All84 ordered state endpoints; dynamic union of initial/final native complete failures, both rounds, all9 FP64 response frames; no prescribed rescued or failed count.'
 return d
recorded_problem=independent.recorded_problem
certificate=independent.certificate
frame_objective=control.objective
chain_check=independent.chain_check
SAFETY=.0018
STEP=.001
def stripped(rr):return [{k:v for k,v in r.items() if k not in ('stage','step')} for r in rr]
def guard(r):return sum(float(r['objective_parts'][k]) for k in PARTS[1:])
def contraction(g,d):return sum((rational(x)*rational(y) for x,y in zip(np.asarray(g).ravel(),np.asarray(d).ravel())),Fraction(0))
