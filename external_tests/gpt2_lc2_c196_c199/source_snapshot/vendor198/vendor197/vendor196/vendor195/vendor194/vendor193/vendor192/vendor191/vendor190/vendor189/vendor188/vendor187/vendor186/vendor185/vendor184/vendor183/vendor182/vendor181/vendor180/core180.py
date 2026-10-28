"""Fixed eight-step continuation from the independently audited C178 final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor179'))
import core179 as prior
from core179 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
previous_training=prior.prior
source174=previous_training.source174
control=previous_training.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=previous_training.problem
exact_certificate=previous_training.exact_certificate
PROTOCOL='GPT2_LC2_C180_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C179',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
ANCHOR_SHA256='e7cbfd1e636869ff34e7c08b235900b560af2bed8607ed9f056cf2b0297daadd'
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP continuation from the original C178 predetermined final step8 head, never a recovered or best checkpoint. Independently reconstruct C178 and bind complete original C179 read-only audit before model loading. Same model/revision, frozen routes/V, prepared/raw states, original C155 response/logit references, coefficient box, native thresholds and0.001 fixed step. Each current full84 batch records all504 signed response-coordinate gradients and84 response/guard loss gradients; all1008 signed LP constraints and exact direction/actual-binary64-head-difference witnesses before each update. Read/KL/radius/regularizer guards measured but not constrained by the response-only LP. All eight slots retained; empty active uses registered zero/no LP; unresolved witness stops incomplete with no fallback, step search or recovery qualification. Only predetermined local step8 receives original native full84 two-round qualification and final FP64 repeat. Report all336 initial/final per-state native-vs-FP64 response margins and disagreements from these same measured endpoints, not extra forwards; native full acceptance unchanged, FP64 response-only is descriptive and does not become an added gate. C176/C178/C180 contribute24 offline update slots since the C170 fixed-final anchor, excluding C170 own eight slots. No online correction-free execution, fresh confirmation, excluded input, candidate/best-checkpoint selection, deployment, changed thresholds, derivative truth, curvature/convergence, solver optimality/infeasibility, chronology/non-access, generalisation or LC2 claim.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source179=clean_json((ROOT/'trusted/C179_SOURCE_PINS.json').read_bytes()))
def design():
 d=previous_training.design();d.pop('prior_C176_training_steps',None)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),anchor='Exact original C178 predetermined final local step8, not any earlier head or probe.',anchor_head_digest=ANCHOR_SHA256,parent_audit_protocol=prior.PROTOCOL,parent_audit_decision=prior.DECISION,prior_C176_C178_training_steps=16,cumulative_training_steps_if_complete=24,parent_binding='All25 original whole ZIP hashes fixed before analysis. Independently reconstruct original C178 through delivered immutable C179 checker and all24 upstream originals; bind original C179 source, complete summary, ancestry and every reconstructed arithmetic/diagnostic artifact before model loading.',measurement_order='Native initial C178-final84 round0; eight fresh full84 coordinate+loss AD batches and fixed updates; frozen local final8 full84 FP64 and repeat; native final84 round0; original C178-final84 round1; local final84 round1.',replay='Initial native both rounds and first full84 FP64 must match original C178 fixed-final archived measurements under original tolerances/point identities. Stop before any LP or update on initial FP64 mismatch; no reference recentering.',endpoint_precision_reporting='All84 states, initial and predetermined final heads, both native rounds compared with the corresponding already-measured FP64 endpoint. Export336 coordinate gaps, absolute response slacks, full native gates and response-only precision disagreement; do not add an FP64 full acceptance gate, change response0.002 or infer accuracy/error bounds from precision differences.',frozen='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached response/logit references and native acceptance thresholds. Only shared coefficient head changes offline from fixed C178 final head.',scope=SCOPE)
 d['counts']['endpoint_precision_comparison_rows']=336
 return d
def requests():return previous_training.requests()
def update(p,d):return previous_training.update(p,d)
