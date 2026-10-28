"""C198 frozen native replay: archived endpoints and freshly acquired proposals."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor197'))
import core197 as parent
from core197 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,archive155,IDS,N,DIM,parameter_hash,array_digest
import core148 as routing
PROTOCOL='GPT2_LC2_C198_FROZEN_END_TO_END_REPLAY_CPU_V1'
INPUT_KEYS=parent.INPUT_KEYS+('C197',)
HEAD_SHA256='7b4811602621506a568c7067bad6dd317c29a2f871a1dbca3684051097df0d4c'
TIMEOUT=14400
FORWARDS=1513
STAGES=('prepare0','features0','prepare1','features1','archived0','archived1','fresh0','fresh1')
SEMANTIC_PINS=('revision','seed','lora_A_sha256','chart_shapes','symbols','anchors','response_ids','sketch_sha256')
SCOPE='Frozen C196 final head, original84 exposed training states, fresh physical FP32 GPT-2 replay. Re-audit all42 C197 parent inputs and bind original C197 before model loading. Separately replay168 exact archived final points and168 newly generated proposals from168 fresh3147-feature captures, full3211 raw-writer inputs and recomputed prewrite-only routes. Recheck168 preparations; keep every failed gate and changed classification. Original C155 FP32 response/logit references retained; fresh zero-chart reference recorded but never substituted. No new head, fitting, optimizer, solve, AD/Jacobian/SVD, teacher, response repair, retry, selection, deployment, new states, confirmation or excluded input. This is same-programme model-level reproducibility evidence, not external-laboratory replication, derivative truth, error bounds, generalisation or LC2 closure.'
def contract():
 return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=parent.contract(),source197=clean_json((ROOT/'trusted/C197_SOURCE_PINS.json').read_bytes()))
def requests():return [dict(stage=stage,state_id=sid) for stage in STAGES for sid in IDS]
def design():
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),head_digest=HEAD_SHA256,head_shape=[3,2,36],training_registry=archive155.registry(),all84_exposed_training=True,stages=list(STAGES),requests=requests(),reference='Original C155 FP32 response reference and baseline logits; fresh initialization baseline diagnostic only, no recentering.',proposal='3147 fresh capture -> frozen raw writer and64 delta-sketch tail -> float32 raw point -> frozen mean/scale/prototypes soft router -> explicit left-sum3x2x36 coefficients -> diagonal plus positive roll1 shift -> float32 commit. IDs/case labels do not enter routing; requested XOR target enters the raw feature interface only.',independent_paths='Archived exact physical points and fresh end-to-end proposals reported separately; archived values never substitute for fresh features/raw/routes/points.',counts=dict(training_states=84,rounds=2,preparation_records=168,feature_records=168,archived_endpoint_records=168,fresh_endpoint_records=168,ordered_records=672,model_forwards=FORWARDS,initialization_forwards=1,feature_forwards=504,metric_forwards=1008,backwards=0,training_steps=0),acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),replay=dict(observed_Linf=.0002,KL_absolute=5e-6,equal_metric_gates=True,archived_point_identity=True),fresh_point_identity='Differences from archived points are diagnostic; fresh points need not be identical to pass the unchanged actual acceptance gates. Repeat point identity and native gates are required across the two fresh rounds.',seal='Both rounds of all168 fresh proposals saved and rehashed before any postwrite endpoint; each prepared state restored after calls; all endpoint measurements consume sealed point bytes.',pass_rule='All84 preparations and fresh proposals must pass both rounds; all168 archived endpoint controls must pass and match original gates/observables/point identity; both path repeats must match original repeat rules. Failures do not suppress remaining registered measurements. Completion is distinct from passing.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False,LC2_single_fold_passed=False)
def no_training():
 return dict(new_training_steps=0,new_head_fits=0,new_backward_calls=0,new_Jacobians=0,new_SVD_calls=0,new_linear_program_calls=0,linear_systems_solved=0,teacher_calls=0,online_response_repair_steps=0,backtracking_steps=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def check_head(p,production=True):
 need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.isfinite(p).all() and np.max(abs(p))<=.125,'Frozen finite3x2xrank coefficient head')
 if production:compare(p.shape,(3,2,36));compare(parameter_hash(p),HEAD_SHA256,'Only C196 predetermined final head')
def fresh_proposal(d,head,prepared,capture):
 # No state ID, class identifier, endpoint measurement or archived routing weight.
 raw,full=safe.reconstruct_raw(d['writer'],d['control']['delta_sketch'],prepared,capture)
 farray(full,(3211,));exact(full[:3147],capture,'Full capture prefix')
 V=d['V'];u=V@(raw.astype(np.float64)-prepared.astype(np.float64));w=routing.route(u,d['router']);e=(w[0]*head[0]+w[1]*head[1])+w[2]*head[2]
 shift=V.T@(e[0]*u+e[1]*np.roll(u,1));point=(raw.astype(np.float64)+shift).astype(np.float32)
 farray(point,(DIM,));return dict(raw=raw,full=full,weights=w,effective=e,point=point)
def summaries(rows):
 compare([r['state_id'] for r in rows],IDS,'Full84 ordered denominator')
 bad=Counter(k for r in rows for k in r['qualification']['failed_gates'])
 return dict(n_requested=84,n_evaluated=len(rows),passed=sum(r['qualification']['complete_pass'] for r in rows),failed_ids=[r['state_id'] for r in rows if not r['qualification']['complete_pass']],failed_gates=dict(bad),maximum_response_drift=max(r['metrics']['response_drift'] for r in rows),minimum_response_slack=min(.002-r['metrics']['response_drift'] for r in rows))
def metric_record(m,point,prepared,case,valid,write=True):
 farray(np.asarray(m['observed'],float),(22,));need(math.isfinite(m['KL']) and m['KL']>=0,'Finite native KL')
 if write:
  expected=archive155.full_metric(m['observed'],m['KL'],point,case);compare(m,expected,'Freshly recomputed native metric fields');q=archive155.endpoint_gates(expected,case,point,prepared,valid)
 else:
  gates,drift,margin=archive155.prev.native_gates(m,case,point,False);bad=[k for k,v in gates.items() if not v];q=dict(complete_pass=not bad,failed_gates=bad,response_drift=drift)
 return dict(metrics=m,qualification=q)
def replay_pair(a,b):return archive155.repeat_check(a,b)
def pair_pass(r):return r['within_replay_tolerance'] and r['same_point_identity']
def outcome(preps,ends,replays,repeats):
 controls=all(summaries(ends['archived'+str(j)])['passed']==84 for j in (0,1)) and all(pair_pass(r) for r in replays)
 fresh=all(summaries(ends['fresh'+str(j)])['passed']==84 for j in (0,1))
 prep=all(r['qualification']['complete_pass'] for rr in preps.values() for r in rr)
 repeated=all(pair_pass(r) for r in repeats)
 passed=bool(controls and fresh and prep and repeated)
 return dict(archived_control_replay_passed=bool(controls),fresh_end_to_end_all84_passed_both_rounds=bool(fresh),all84_preparations_passed_both_rounds=bool(prep),native_repeats_consistent=bool(repeated),frozen_training_replay_passed=passed,decision='C198_FROZEN_TRAINING_REPLAY_ALL84_PASS_NO_GENERALISATION' if passed else 'C198_COMPLETE_REPLAY_OR_TRAINING_CONSTRAINT_FAILURE_NO_GENERALISATION')
