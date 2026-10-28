"""Read-only independent arithmetic audit of the C178 continuation."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor178'))
import core178 as prior
from core178 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
control=prior.control
PROTOCOL='GPT2_LC2_C179_READONLY_CONTINUATION_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C178',)
TIMEOUT=3600
DECISION='C179_READONLY_CONTINUATION_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C179_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy twenty-four-original archival audit of C178 fixed eight-step all-coordinate offline continuation from the predetermined C176 final head. Independently bind all23 upstream originals and original C177 audit using delivered immutable source before re-auditing C178. Rebuild every1008-row LP from archived coordinate gradients with the independently implemented C175 arithmetic checker, raw-primal mapping, exact recorded direction and actual binary64 endpoint-difference certificates, all eight fixed updates,840 saved FP64 points,1176 records, final FP64 repeat and336 native gates. Retain all84 trajectories over nine frames/eight transitions, guard-part decomposition,4032 response-coordinate affine residuals and observed cap violations. Recompute all168 paired native state outcomes, full rescued/destroyed gate records and final failure trajectories without assuming the reported80/84 or monotonic improvement. No model/Torch/HF/SciPy import, new forward/backward/fit/training/LP/solve/Jacobian/SVD, state preparation, confirmation or excluded input. Archived gradients, outputs and KL are not independently measured. Exact recorded arithmetic does not certify derivative truth, finite error bounds, solver optimality/infeasibility, chronology, non-access, saved-point consumption, curvature, convergence, causality or generalisation. No head/step selection, deployment, changed thresholds or LC2 claim; original C178 decision retained.'

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source178=clean_json((ROOT/'trusted/C178_SOURCE_PINS.json').read_bytes()))
def no_compute():return prior.prior.no_compute()
def design():
 d=prior.prior.design()
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C178_zip_hash_known_before_runtime=True,anchor='Original C176 predetermined final local step8; exactly eight further C178 offline slots to its predetermined local final step8. No intermediate checkpoint or probe chosen.',arithmetic='All24 whole original ZIPs fixed by known hashes, original C177 independently rebound, delivered immutable C178 source, original C176-final initial head. Independent C175 problem/exact rational checker; no reuse of C174 production problem/certificate implementation. Reconstruct all1008-row problems, raw-primal mapping,16 exact certificates and unprojected0.001 updates. Retain original unknown solver-internal linear-system count as null; new solver calls0.',diagnostics='All84 states, nine FP64 frames, eight transitions and both endpoint native rounds. Report increases, zero-loss-to-active states, coordinate switches, all4032 observed cap comparisons, guard components, all168 paired native outcomes, all rescued/destroyed full-gate records and final failure trajectories. No assumption of80/84, no monotonicity requirement or future-step extrapolation.',scope=SCOPE)
 d.pop('C176_zip_hash_known_before_runtime',None)
 d['counts'].update(paired_native_state_outcomes=168)
 return d

# C178 production uses core174; this audit uses the independent core175 checker.
recorded_problem=prior.prior.recorded_problem
certificate=prior.prior.certificate
frame_objective=control.objective
chain_check=prior.prior.chain_check
SAFETY=.0018
STEP=.001
def stripped(rr):return [{k:v for k,v in r.items() if k not in ('stage','step')} for r in rr]
def guard(r):return sum(float(r['objective_parts'][k]) for k in PARTS[1:])
def contraction(g,d):return sum((rational(x)*rational(y) for x,y in zip(np.asarray(g).ravel(),np.asarray(d).ravel())),Fraction(0))
