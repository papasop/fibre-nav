"""Read-only independent arithmetic audit of the fixed C180 continuation."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor180'))
import core180 as prior
from core180 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
import core179 as independent
control=prior.control
PROTOCOL='GPT2_LC2_C181_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C180',)
TIMEOUT=3600
DECISION='C181_READONLY_CONTINUATION_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C181_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy twenty-six-original archival audit of C180 fixed eight-step all-coordinate offline continuation from the predetermined C178 final head. Independently bind all25 upstream originals and original C179 audit through delivered immutable source before re-auditing C180. Rebuild every1008-row LP from archived coordinate gradients using the independent C175 arithmetic checker, raw-primal mapping, exact recorded direction and actual binary64 endpoint-difference certificates, all eight fixed updates,840 saved FP64 points,1176 records, final FP64 repeat and336 native gates. Retain all84 trajectories over nine frames/eight transitions, guard-part decomposition,4032 response-coordinate affine residuals and actual cap violations. Independently reconstruct and bind every336 original precision-margin row, CSV and summary; report both precision response classifications without adding an FP64 full gate, and no assumption that disagreements disappear or that80/84 pass. No model/Torch/HF/SciPy import, new forward/backward/fit/training/LP/solve/Jacobian/SVD, state preparation, confirmation or excluded input. Archived gradients, outputs and KL are not independently measured. Recorded arithmetic and precision agreement do not certify derivative truth, measurement accuracy, finite error bounds, solver optimality/infeasibility, chronology, non-access, saved-point consumption, curvature, convergence, causality or generalisation. No head/step selection, deployment, changed thresholds or LC2 claim; original C180 decision retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source180=clean_json((ROOT/'trusted/C180_SOURCE_PINS.json').read_bytes()))
def no_compute():return independent.no_compute()
def design():
 d=independent.design()
 d.pop('C178_zip_hash_known_before_runtime',None)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C180_zip_hash_known_before_runtime=True,anchor='Exact original C178 predetermined final local step8; C180 contributes8 slots, cumulative C176/C178/C18024 since C170 fixed-final anchor, excluding C170 own eight slots.',arithmetic='All26 whole original ZIP hashes fixed before payload. All25 upstream originals and original C179 audit rebound through immutable C180 loader. Independent C175 problem/exact rational checker, not C174 production problem/certificate; all1008-row problems, raw-primal mapping,16 certificates and fixed binary640.001 updates. Solver-internal original count unknown/null; new solver calls0.',diagnostics='All84 states and all9 FP64 frames,8 transitions,4032 coordinate cap comparisons,168 paired full native outcomes and remaining failure trajectories. Reconstruct336 endpoint precision rows and CSV using independent arithmetic and original0.002 budget. Keep negative slacks, all disagreements, response-only vs full-native distinctions, initial/final summaries and regularizer/other guard parts. Do not assume monotonicity, counts, causes or future-step requirements.',scope=SCOPE)
 d['counts'].update(endpoint_precision_rows_independently_reconstructed=336)
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
