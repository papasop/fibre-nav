"""Fixed eight-step continuation from the independently audited C186 final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor187'))
import core187 as prior
from core187 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
previous_training=prior.prior
source174=previous_training.source174
control=previous_training.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=previous_training.problem
exact_certificate=previous_training.exact_certificate
PROTOCOL='GPT2_LC2_C188_ALL_COORDINATE_CONTINUATION_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C187',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
ANCHOR_SHA256='005c3da405a9dff4f1168009b7857e5e8eb2247810647020ea49417d1e432c06'
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP continuation from original C186 predetermined final step8 head. Reconstruct original C186 and bind full original C187 audit including all precision/activity/focus and168 native/84 FP64 anchor replay pairs before model loading. All33 original ZIP hashes and source identities fixed. Same pinned model/revision, frozen V/routes/prepared/raw states and original C155 response/logit references. Step0.001, at most8 all1008-signed-row LPs, all672 coordinate Jacobian blocks/4032 rows and5376 scalar backwards explicitly offline. All840 actual FP64 points sealed before measurement,1176 records, predetermined final8 only, both native full84 rounds and final FP64 repeat. Distinguish same-run progress from cross-run observable differences; original replay tolerances unchanged, no reference recentering. Guard losses measured, not constrained by response-only LP. Retain all84 including every failed/destroyed state, no preset acceptance or rescue count. C176/C178/C180/C182/C184/C186/C188 contribute56 offline update slots since C170 final, excluding C170 own8. No recovery or best-head selection, threshold change, deployment, new confirmation or excluded input. No derivative truth, finite error bound, solver optimality/infeasibility, chronology/non-access, causal explanation, convergence, generalisation or LC2 closure claim.'

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source187=clean_json((ROOT/'trusted/C187_SOURCE_PINS.json').read_bytes()))
def design():
 d=previous_training.design()
 for k in list(d):
  if k.startswith('prior_C176_'):d.pop(k)
 d.update(protocol=PROTOCOL,inputs=list(INPUT_KEYS),anchor='Exact original C186 predetermined final local step8, not earlier head, probe or recovery point.',anchor_head_digest=ANCHOR_SHA256,parent_audit_protocol=prior.PROTOCOL,parent_audit_decision=prior.DECISION,prior_C176_C178_C180_C182_C184_C186_training_steps=48,cumulative_training_steps_if_complete=56,parent_binding='All33 original whole ZIP hashes fixed before payload. Bind all31 upstream originals through immutable C186 loader; independently reconstruct original C186 using immutable C187 checker; bind original C187 source, full summary, ancestry and every reconstructed arithmetic/precision/activity/focus/replay artifact before model loading.',measurement_order='Native initial C186-final84 round0; eight fresh full84 coordinate+loss AD batches and fixed updates; predetermined final8 full84 FP64 and repeat; native final84 round0; original C186-final84 round1; final84 round1.',replay='Initial native both rounds and first full84 FP64 must match original C186 fixed-final archived measurements under unchanged replay tolerances/point identities. Stop before first LP/update on first-batch mismatch; no reference recentering.',endpoint_precision_reporting='All84 states, initial and predetermined final heads, both native rounds compared with corresponding already-measured FP64 endpoint; export336 signed response slacks, native full gates and response-only precision disagreements. No added FP64 acceptance gate or accuracy/error-bound inference.',frozen='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached references and native acceptance thresholds. Only shared coefficient head changes offline from fixed C186 final.',scope=SCOPE)
 return d
def requests():return previous_training.requests()
def update(p,d):return previous_training.update(p,d)
