"""Fixed eight-step continuation from the independently audited C188 final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor189'))
import core189 as prior
from core189 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
previous_training=prior.prior
source174=previous_training.source174
control=previous_training.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=previous_training.problem
exact_certificate=previous_training.exact_certificate
PROTOCOL='GPT2_LC2_C190_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C189',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
ANCHOR_SHA256='2bfa9158e2ea1e2175c948722f997036a2aa03446a075555989f96f8a71d7813'
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP continuation from original C188 predetermined final step8 head. Reconstruct original C188 and bind full original C189 audit including every precision/activity/focus and168 native/84 FP64 cross-run replay artifact before model loading. All35 original ZIP hashes and delivered source identities fixed. Same pinned model/revision, frozen V/routes/prepared/raw states and original C155 response/logit references. Step0.001, at most8 all1008-signed-row LPs, all672 coefficient-response Jacobian blocks/4032 rows and5376 scalar backwards explicitly offline. All840 actual FP64 points sealed before measurement,1176 records, predetermined final8 only, both native full84 rounds and final FP64 repeat. Distinguish same-run training change from cross-run output differences; original replay tolerances unchanged and no reference recentering. Response-only LP does not constrain measured guards; final native full gates remain mandatory. All84 states retained, no preset failure/rescue count, monotonicity, future steps or closure time. C176/C178/C180/C182/C184/C186/C188/C190 contribute64 offline update slots since C170 final, excluding C170 own8. No recovery or best-head selection, changed acceptance thresholds, deployment, new confirmation or excluded-input evaluation. No derivative truth, finite error bound, solver optimality/infeasibility, chronology/non-access, causal explanation, curvature/convergence, generalisation or LC2 closure claim.'

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source189=clean_json((ROOT/'trusted/C189_SOURCE_PINS.json').read_bytes()))
def design():
 d=previous_training.design()
 for k in list(d):
  if k.startswith('prior_C176_'):d.pop(k)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),anchor='Exact original C188 predetermined final local step8, not earlier head, probe or recovery point.',anchor_head_digest=ANCHOR_SHA256,parent_audit_protocol=prior.PROTOCOL,parent_audit_decision=prior.DECISION,prior_C176_C178_C180_C182_C184_C186_C188_training_steps=56,cumulative_training_steps_if_complete=64,parent_binding='All35 original whole ZIP hashes fixed before payload. Bind all33 upstream originals through immutable C188 loader; independently reconstruct original C188 using immutable C189 checker; bind original C189 source, full summary, ancestry and every reconstructed arithmetic/precision/activity/focus/replay artifact before model loading.',measurement_order='Native initial C188-final84 round0; eight fresh full84 coordinate+loss AD batches and fixed updates; predetermined final8 full84 FP64 and repeat; native final84 round0; original C188-final84 round1; final84 round1.',replay='Initial native both rounds and first full84 FP64 must match original C188 fixed-final archived measurements under unchanged replay tolerances/point identities. Stop before first LP/update on first-batch mismatch; no reference recentering.',endpoint_precision_reporting='All84 states, initial and predetermined final heads, both native rounds compared with corresponding already-measured FP64 endpoint; export336 signed response slacks, native full gates and response-only precision disagreements. No added FP64 acceptance gate or accuracy/error-bound inference.',frozen='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached references and native acceptance thresholds. Only shared coefficient head changes offline from fixed C188 final.',scope=SCOPE)
 return d
def requests():return previous_training.requests()
def update(p,d):return previous_training.update(p,d)
