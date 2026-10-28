"""Fixed eight-step continuation from the independently audited C180 final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor181'))
import core181 as prior
from core181 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
previous_training=prior.prior
source174=previous_training.source174
control=previous_training.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=previous_training.problem
exact_certificate=previous_training.exact_certificate
PROTOCOL='GPT2_LC2_C182_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C181',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
ANCHOR_SHA256='3a49ff32146b08b145442f226a63d743e1aa1d5bb997e18004f678ec7e606eca'
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP continuation from original C180 predetermined final step8 head, never a recovered or best checkpoint. Independently reconstruct C180 and bind complete original C181 audit including its precision report before loading model. Same model/revision, frozen routes/V, prepared/raw states, original C155 response/logit references, coefficient box, native thresholds and0.001 fixed step. Every current full84 batch records all504 signed response-coordinate gradients and84 response/guard loss gradients; all1008 signed LP constraints and exact recorded direction/actual-binary64-head-difference witnesses before each update. Read/KL/radius/regularizer guards measured but not constrained by response-only LP. All eight slots retained; empty active uses registered zero/no LP; unresolved witness stops incomplete without fallback, step search or recovery qualification. Only predetermined local step8 receives original native full84 two-round qualification and final FP64 repeat. All336 initial/final per-state native-vs-FP64 response margins from existing measurements, no extra forwards or added FP64 full gate. C176/C178/C180/C182 contribute32 offline update slots since C170 fixed-final anchor, excluding C170 own eight slots. No online correction-free execution, fresh confirmation, excluded input, candidate/best-checkpoint selection, deployment, changed thresholds, derivative truth, curvature/convergence, solver optimality/infeasibility, chronology/non-access, generalisation or LC2 claim.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source181=clean_json((ROOT/'trusted/C181_SOURCE_PINS.json').read_bytes()))
def design():
 d=previous_training.design();d.pop('prior_C176_C178_training_steps',None)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),anchor='Exact original C180 predetermined final local step8, not earlier head, probe or recovery point.',anchor_head_digest=ANCHOR_SHA256,parent_audit_protocol=prior.PROTOCOL,parent_audit_decision=prior.DECISION,prior_C176_C178_C180_training_steps=24,cumulative_training_steps_if_complete=32,parent_binding='All27 original whole ZIP hashes fixed before payload. Independently reconstruct original C180 using immutable C181 checker and all26 upstream originals; bind original C181 source, full summary, ancestry, every reconstructed arithmetic/diagnostic artifact including all336 precision rows/CSV and168 transitions before model loading.',measurement_order='Native initial C180-final84 round0; eight fresh full84 coordinate+loss AD batches and fixed updates; predetermined final8 full84 FP64 and repeat; native final84 round0; original C180-final84 round1; final84 round1.',replay='Initial native both rounds and first full84 FP64 must match original C180 fixed-final archived measurements under unchanged replay tolerances/point identities. Stop before first LP/update on first-batch mismatch; no reference recentering.',endpoint_precision_reporting='All84 states, initial and predetermined final heads, both native rounds compared with corresponding already-measured FP64 endpoint; export336 signed response slacks, native full gates and response-only precision disagreements. No added FP64 acceptance gate or accuracy/error-bound inference.',frozen='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached references and native acceptance thresholds. Only shared coefficient head changes offline from fixed C180 final.',scope=SCOPE)
 return d
def requests():return previous_training.requests()
def update(p,d):return previous_training.update(p,d)
