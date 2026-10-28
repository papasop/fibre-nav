"""Read-only C176 fixed8 all-coordinate training audit; no model or solver."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor176'))
import core176 as prior
from core176 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
control=prior.control
PROTOCOL='GPT2_LC2_C177_READONLY_ALL_COORDINATE_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C176',)
TIMEOUT=3600
DECISION='C177_READONLY_ALL_COORDINATE_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C177_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy twenty-two-original archival audit of C176 fixed eight-step all-coordinate offline training. Rebuild every1008-row LP from archived coordinate gradients with the independently implemented C175 arithmetic checker, raw-primal mapping, exact recorded direction and actual binary64 endpoint-difference certificates, complete fixed updates, all840 saved FP64 points, all1176 records, final FP64 repeat and336 native gates. Describe all84 states over nine frames/eight transitions, including guard-part decomposition and all4032 response-coordinate affine prediction residuals. No model/Torch/HF/SciPy import, new forward/backward/fit/training/LP/solve/Jacobian/SVD, preparation, confirmation or excluded input. Archived gradients, model outputs and KL are not independently measured. Exact recorded arithmetic does not certify true derivatives, finite error bounds, solver optimality/infeasibility, chronology, non-access, saved-point consumption, curvature, convergence, causality or generalisation. No step/head selection, deployment, changed thresholds or LC2 claim; parent decision retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source176=clean_json((ROOT/'trusted/C176_SOURCE_PINS.json').read_bytes()))
def no_compute():return dict(new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_linear_program_calls=0,linear_systems_solved=0,new_Jacobians=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C176_zip_hash_known_before_runtime=True,training_registry=prior.design()['training_registry'],anchor='Original C170 fixed final step8; exactly eight C176 updates to its predetermined final step8.',arithmetic='All22 fixed originals and trusted C176 source; independent C175 response-coordinate problem and exact rational checker, no reuse of C174 production problem/certificate implementation. Each full1008-row registry,raw primal clipping/free-mask mapping,direction/actual-head certificates,scalar-gradient chain and unprojected0.001 update is reconstructed. Original solver internal linear solve count remains null; new solver calls0.',counts=dict(training_states=84,training_updates=8,ordered_records=1176,gradient_records=672,archived_coefficient_response_Jacobian_blocks=672,archived_response_coordinate_gradient_rows=4032,LP_signed_rows_each=1008,LP_signed_constraint_rows_reconstructed=8064,exact_certificates=16,exact_signed_certificate_rows=16128,actual_FP64_point_arrays=840,native_FP32_point_arrays=168,native_endpoint_records=336,native_repeat_pairs=168,FP64_final_repeat_pairs=84,trajectory_frame_rows=756,trajectory_transition_rows=672,response_coordinate_affine_rows=4032),diagnostics='All nine archived FP64 frames and eight observed transitions retained. Report all loss increases, zero-loss-to-active states, max-coordinate changes, guarded loss parts, actual versus recorded-affine residuals, initial five-failure trajectories, original endpoint gates and precision comparisons; no monotonicity assumption, finite-error certificate or future-step extrapolation.',thresholds='Original response0.002, all KL/read/radius/map/replay tolerances and coefficient box remain unchanged. max(0.0018,current response maximum) is a training linearisation cap only.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
# Production C176 uses core174; this read-only checker uses the independent core175 implementation.
recorded_problem=prior.prior.problem
certificate=prior.prior.certificate
frame_objective=control.objective
chain_check=prior.prior.chain_check
SAFETY=.0018
STEP=.001

def stripped(rr):return [{k:v for k,v in r.items() if k not in ('stage','step')} for r in rr]
def guard(r):return sum(float(r['objective_parts'][k]) for k in PARTS[1:])
def contraction(g,d):return sum((rational(x)*rational(y) for x,y in zip(np.asarray(g).ravel(),np.asarray(d).ravel())),Fraction(0))
