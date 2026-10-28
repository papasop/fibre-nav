"""Read-only C168 finite-probe audit; NumPy only, no model or solver imports."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor168'))
import core168 as prior
from core168 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe
base166=prior.prior.prior
mapping=base166.mapping
losses=base166.losses
PROTOCOL='GPT2_LC2_C169_READONLY_FINITE_PROBE_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C168',)
TIMEOUT=1800
DECISION='C169_READONLY_FINITE_PROBE_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C169_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy fourteen-original archive, tensor, gate and finite-difference arithmetic audit. No model/HF/Torch import, forward, backward, fitting, optimization, new direction, linear solve, Jacobian, SVD, preparation, confirmation or excluded-input evaluation. Recorded model outputs, KL and gradients are not independently re-evaluated. Saved point bindings cannot prove measurement accuracy, actual execution chronology, non-access or model consumption. Finite secants, observed-minus-linear residuals and precision gaps do not establish derivative error, certified curvature, physical error cause, convergence, feasibility, infeasibility or generalisation. Every registered state, arm and failure is retained, without step/head selection or deployment. All parent decisions retained; LC2 remains open.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source168=clean_json((ROOT/'trusted/C168_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=prior.design()['training_registry'],parent_protocol=prior.PROTOCOL,parent_decision='C168_FIXED_CERTIFIED_DIRECTION_PROBES_COMPLETE_NO_CANDIDATE_SELECTED',whole_C168_hash_known_before_runtime=True,all84_are_exposed_training=True,counts=dict(ordered_records=1848,actual_FP64_point_arrays=672,native_FP32_point_arrays=588,native_endpoint_records=1176,native_repeat_pairs=588,FP64_anchor_repeat_pairs=84,scalar_central_difference_rows=252,signed_loss_change_rows=504,response_coordinate_central_rows=1512,precision_comparisons=1176),audit='Bind all fourteen original ZIPs and the delivered C168 source. Reconstruct C167 exact positive certificate from original C165 archived gradients via the trusted C168 reader. Recompute C168 step panel, six complete coefficient heads, all NumPy/actual FP64/native FP32 point bindings, 1848 records, loss parts, all original native gates, all repeat checks and original summaries. Archived source is data only and never executed.',diagnostics='All84 states and all6 arms: signed response/guard loss changes, archived-gradient linear predictions and their finite residuals; all3 paired central secants and centered second secants; all6 signed response-coordinate budgets, active-coordinate/sign changes; FP64 versus native response/observable gaps and committed-point displacement/rounding records. No best arm, optimal step or extrapolated success selected.',tolerances='Reuse original source-pinned model-replay, map and native acceptance tolerances; no new empirical pass threshold for diagnostic secants or residuals.',worker_budget_seconds=TIMEOUT,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_LP_calls=0,LC2_closed=False,scope=SCOPE)
def no_compute():
 return dict(new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,new_linear_program_calls=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_states_prepared=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def guard(row):return sum(float(row['objective_parts'][k]) for k in PARTS[1:])
def dot(g,d):return float(np.sum(np.asarray(g,dtype=np.float64)*d))
def stripped(rows):return [{k:v for k,v in r.items() if k not in ('stage','arm')} for r in rows]
