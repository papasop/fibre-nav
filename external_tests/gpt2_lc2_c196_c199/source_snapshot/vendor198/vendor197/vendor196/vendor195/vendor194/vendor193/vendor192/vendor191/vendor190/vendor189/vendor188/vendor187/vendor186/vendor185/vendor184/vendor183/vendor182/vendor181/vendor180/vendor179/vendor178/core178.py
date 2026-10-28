"""Fixed eight-step continuation from the independently audited C176 final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor177'))
import core177 as prior
from core177 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
previous_training=prior.prior
source174=previous_training.source174
control=previous_training.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=previous_training.problem
exact_certificate=previous_training.exact_certificate
PROTOCOL='GPT2_LC2_C178_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C177',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
ANCHOR_SHA256='c4e6a00d16a11069c2f1d345092c76d80bec6b44a752e4c3fed89c4f3a8abdab'
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP continuation from the original C176 predetermined final step8 head, never a recovered or best checkpoint. Independently reconstruct C176 and bind the complete original C177 read-only audit before model loading. Same model/revision, frozen routes/V, prepared/raw states, original C155 response/logit references, coefficient box and native thresholds; no reference recentering. Each of eight current batches records all504 signed response-coordinate gradients and84 response/guard loss gradients; fixed eta0.001 and all1008 signed LP constraints, exact recorded-direction and actual-binary64-head-difference witnesses before every update. Read/KL/radius/regularizer guards are measured but not constrained by this response-only LP. All eight slots retained; empty active uses registered zero/no LP; unresolved witness stops incomplete with no fallback, step search or recovery qualification. Only predetermined local step8 (cumulative step16 since C170) receives original native full84 two-round qualification and a final FP64 repeat. No online correction-free execution, fresh confirmation, excluded input, candidate/best-checkpoint selection, deployment, changed thresholds, derivative truth, curvature/convergence, solver optimality/infeasibility, chronology/non-access, generalisation or LC2 claim.'

def contract():
 return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source177=clean_json((ROOT/'trusted/C177_SOURCE_PINS.json').read_bytes()))

def design():
 d=previous_training.design()
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),anchor='Exact original C176 predetermined final local step8; its final head is not selected from C176 intermediate checkpoints.',anchor_head_digest=ANCHOR_SHA256,parent_audit_protocol=prior.PROTOCOL,parent_audit_decision=prior.DECISION,prior_C176_training_steps=8,cumulative_training_steps_if_complete=16,parent_binding='All23 original ZIPs fixed by known whole SHA256. Reconstruct original C175 binding, all C176 updates/points/gates and C177 diagnostics with delivered immutable source; bind original C177 source, summary, ancestry and all independently reconstructed C176-arithmetic/diagnostic artifacts before loading a model.',measurement_order='Native initial C176-final84 round0; eight fresh full84 coordinate+loss AD batches and fixed updates; frozen local final8 full84 FP64 and repeat; native final84 round0; original C176-final84 round1; local final84 round1.',replay='Initial native both rounds and first full84 FP64 must match the archived original C176 fixed-final records under original tolerances and point identities; stop before LP/update on first-batch mismatch; no response/logit reference recentering.',scope=SCOPE)
 d['training']['steps']=STEPS
 d['training']['solver']='Original pinned SciPy1.17.0 highs-ds/options, at most one LP per active batch, at most8 total; solver internal linear-system count unknown/null.'
 d['frozen']='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached response/logit references and all native thresholds. Only shared coefficient head changes offline, starting from fixed C176 final head.'
 return d

def requests():return previous_training.requests()
def update(p,d):return previous_training.update(p,d)
