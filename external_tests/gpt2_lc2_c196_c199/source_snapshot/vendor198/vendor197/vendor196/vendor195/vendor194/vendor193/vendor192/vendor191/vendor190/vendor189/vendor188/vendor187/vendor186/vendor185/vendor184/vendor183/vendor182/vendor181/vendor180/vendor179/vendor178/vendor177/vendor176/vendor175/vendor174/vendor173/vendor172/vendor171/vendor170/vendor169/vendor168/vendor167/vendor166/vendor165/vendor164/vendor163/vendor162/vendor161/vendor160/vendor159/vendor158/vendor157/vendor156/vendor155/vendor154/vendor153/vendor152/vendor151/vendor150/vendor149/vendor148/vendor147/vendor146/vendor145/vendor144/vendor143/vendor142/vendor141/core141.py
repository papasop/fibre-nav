"""Fixed48 new kernel directions; frozen C140 confirmation, no fitting."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor140'))
import core140 as parent
from core140 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,RADIUS,BUDGET
PROTOCOL='GPT2_LC2_C141_FROZEN_KERNEL_CONFIRMATION_CPU_V1'
INPUT_KEYS=parent.INPUT_KEYS+('C140',);TIMEOUT=3000;N=48;NORMS=(.125,.375);DIRECTIONS_PER_NORM=2;SEED_BASE=141082931
ARMS=('raw_control','C137_frozen_step24','C140_frozen_step12')

def contract():return dict(parent.contract(),pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source140=clean_json((ROOT/'trusted/C140_SOURCE_PINS.json').read_bytes()))

def registry():
    return [dict(state_id=f"{sp['case']}__c141_confirmation_p{i:02d}_n{k:02d}_d{j:02d}",case=sp['case'],parent_id=sp['state_id'],seed=SEED_BASE+1000*i+100*k+j,norm=norm,direction_index=j,cohort='prospective_seen_class_kernel_local_confirmation',dataset_role='confirmation',used_for_training=False) for i,sp in enumerate(parent.parent.registry()) for k,norm in enumerate(NORMS) for j in range(DIRECTIONS_PER_NORM)]

def design():
    return dict(protocol=PROTOCOL,registry=registry(),new_local_states=N,states_per_trained_class=16,known_training_states=36,new_parent_centers=0,
        candidate='Exact C140 fixed step12 a, original C132 V/raw writer/control/normalizers and feature formulas frozen; C137 fixed step24 only a descriptive control. No choice among the three arms.',
        preparation='u=standard_normal(fixed seed); t=u-V.T@(V@u); point=float32(known_parent_prepared + declared_norm*t/||t||). Use the unchanged frozen C132 shared row basis; no new Jacobian, SVD, solver, refit, retraction, shrinkage, repair, resampling or acceptance-based selection.',
        geometry_scope='The fixed shared row-space orthogonal complement is not a certified local tangent space or a guarantee of finite-step acceptance. Native preparation gates decide validity.',
        norms=list(NORMS),directions_per_parent_and_norm=DIRECTIONS_PER_NORM,seed_base=SEED_BASE,
        freeze='Freeze design and candidate tensors before generating all48 coordinates; freeze coordinates before model initialisation; freeze all144 raw/C137/C140 proposals before any new writer endpoint. New preparation outcomes never alter any proposal formula, seed, norm, denominator or head.',
        novelty='New IDs/seeds relative to the bound declared C138/C139/C140 registries; disjoint coordinate hashes from all bound training inputs, archived final points and explicit coordinate-hash fields collected from the nine originals. No exhaustive programme-wide historical non-access or chronology certification.',
        preflight='Recompute C14012-update/432-row arithmetic and216 native records from nine original archives; re-extract all36 training prewrite feature maps and replay144 known preparation/raw/C137/C140 points. Point/gate identity, feature/raw map2e-6, native observations2e-4 and KL5e-6. Drift stops before new-state measurements.',
        new_preparations='Measure all48 preparations twice at fixed coordinates before features or writer endpoints. Retain invalid preparations and repeat drift. For writer qualification a state requires both preparation rounds valid.',
        new_proposals='Extract prewrite features for all48 states; generate original raw, C137 and C140 proposals unchanged, even for invalid preparations. No postwrite features enter proposals.',
        native_evaluation='All3 arms on all48 states,2 rounds,288 writer records; no deletion/replacement of failed preparations or failed writes. Repeats are the same48 states, not96 independent samples.',
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        training_safety_line_not_an_acceptance_threshold=.00125,
        fixed_requests=dict(known_features=36,known_native_points=144,new_preparation_records=96,new_feature_records=48,new_writer_endpoints=288,metric_calls=528,feature_calls=84,model_forwards_including_initialisation=1309),
        pass_rule='All48 preparations valid in both rounds; frozen C140 complete native gates pass48/48 in both rounds; all requested records complete and known-point/repeat consistency. Raw/C137 are descriptive, never replacements for C140.',
        next_rule='If complete confirmation passes, preregister a separate excluded-input split/provenance and evaluation contract. If it fails, retain the negative result and mark all exposed C141 states development-only for future work; no reuse as unseen tests after adaptation.',
        scope='Prospectively declared local new-direction confirmation in the same fixed shared-kernel preparation family around known training centers, for xor00/01/11. Not new-center, excluded-input, long-sequence or cross-model generalisation; cannot close LC2 or a single excluded fold.',
        new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,
        excluded_input_states_evaluated=0,candidate_selected=False,writer_deployed=False,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT)

def head_digests(d):return dict(parent.head_digests(d),C140_a=row_digest(d['a140']))

def fresh_coordinates(d,rows=None):
    rows=registry() if rows is None else rows;compare(rows,registry(),'Exact48 registered confirmation states');v=farray(d['V'],(len(d['V']),DIM));need(len(v)>0 and np.max(abs(v@v.T-np.eye(len(v))))<=1e-8,'Frozen basis not orthonormal; no repair')
    oldids={sp['state_id'] for sp in parent.registry()};oldseeds={sp['seed'] for sp in parent.previous.registry()}|{sp['seed'] for sp in parent.previous.previous.registry()};known=set(d['historical_hashes']);points={};records=[]
    for r in rows:
        need(r['state_id'] not in oldids and r['seed'] not in oldseeds and r['case'] in ('xor_00','xor_01','xor_11'),'Old seed/ID or excluded input forbidden');base=d['points'][r['parent_id']]['prepared'];farray(base,(DIM,));need(base.dtype==np.float32,'Float32 parent required')
        u=np.random.default_rng(r['seed']).normal(size=DIM);t=u-v.T@(v@u);nn=float(np.linalg.norm(t));need(np.isfinite(nn) and nn>1e-12,'Collapsed kernel direction; no resampling');delta=r['norm']*t/nn;point=(base.astype(float)+delta).astype(np.float32);h=safe.tensor_hash(point);need(h not in known,'Historical/batch coordinate collision; no replacement');known.add(h)
        actual=point.astype(float)-base.astype(float);distance=float(np.linalg.norm(actual));need(abs(distance-r['norm'])<=1e-5,'Committed perturbation radius differs');points[r['state_id']]=point;records.append(dict(**r,point_sha256=h,parent_sha256=safe.tensor_hash(base),actual_distance=distance,ideal_shared_row_component_Linf=float(np.max(abs(v@delta))),committed_shared_row_component_Linf=float(np.max(abs(v@actual))),float32_commit_error_Linf=float(np.max(abs(actual-delta)))))
    need(len(points)==len({r['seed'] for r in rows})==N,'All48 distinct states/seeds required');return points,records

def gate_record(m,case,point,prepared,valid):
    g=parent.parent.gate_record(m,case,point,prepared)
    if not valid:g['complete_pass']=False;g['failed_gates'].insert(0,'preparation')
    return g

def summaries(records):
    result={};pairs={}
    for group in ('all48','xor_00','xor_01','xor_11'):
        result[group]={};pairs[group]={}
        for rnd in (0,1):
            subset=[r for r in records if r['round']==rnd and (group=='all48' or r['case']==group)];by={(r['state_id'],r['arm']):r for r in subset};result[group][str(rnd)]={};pairs[group][str(rnd)]={}
            for arm in ARMS:
                rr=[r for r in subset if r['arm']==arm];bad=Counter(k for r in rr for k in r['recomputed_gates']['failed_gates']);result[group][str(rnd)][arm]=dict(n_requested=N if group=='all48' else 16,n_evaluated=len(rr),passed=sum(r['recomputed_gates']['complete_pass'] for r in rr),failed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass']],failed_gates=dict(sorted(bad.items())),maximum_response_drift=max((r['metrics']['response_drift'] for r in rr),default=None),within_C140_training_response_safety=sum(r['metrics']['response_drift']<=.00125 for r in rr));pairs[group][str(rnd)][arm]={}
                for control in ARMS[:2]:pairs[group][str(rnd)][arm][control]=dict(rescued_ids=[r['state_id'] for r in rr if r['recomputed_gates']['complete_pass'] and not by[(r['state_id'],control)]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass'] and by[(r['state_id'],control)]['recomputed_gates']['complete_pass']])
    return result,pairs

def preparation_summary(rows):return dict(n_requested=N,n_evaluated=len(rows),passed=sum(r['preparation_valid'] for r in rows),failed_ids=[r['state_id'] for r in rows if not r['preparation_valid']],failed_gates=dict(Counter(k for r in rows for k,v in r['metrics']['gates'].items() if not v)),maximum_response_drift=max((r['metrics']['response_drift'] for r in rows),default=None))

class Phase:
    def __init__(self,d,fresh):
        self.phase='controls';self.pending=None;self.pending_feature=None;self.used=set();self.table={};self.features={}
        for sp in parent.registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('C137',d['C140_points'][(sid,parent.EVAL_ARMS[1])],True),('C140',d['C140_points'][(sid,parent.EVAL_ARMS[2])],True)):self.table[('controls',(sid,kind))]=(safe.tensor_hash(point),sp['case'],write)
            self.features[('controls',sid)]=(safe.tensor_hash(p['prepared']),int(sp['case'][-2])^int(sp['case'][-1]))
        for sp in registry():
            sid=sp['state_id']
            for rnd in (0,1):self.table[('preparation',(rnd,sid))]=(safe.tensor_hash(fresh[sid]),sp['case'],False)
            self.features[('prewrite',sid)]=(safe.tensor_hash(fresh[sid]),int(sp['case'][-2])^int(sp['case'][-1]))
    def metric(self,key,point,case,write):
        tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write);need(self.pending is None and self.pending_feature is None and tag not in self.used and self.table.get(tag)==expected,'Unregistered/repeated native point or phase');self.pending=expected;self.used.add(tag)
    def consume(self,point,case,write):need(self.pending==(safe.tensor_hash(point),case,write),'Unauthorized native point');self.pending=None
    def feature(self,sid,point,desired):
        tag=(self.phase,sid);expected=(safe.tensor_hash(point),desired);need(self.phase in ('controls','prewrite') and self.pending is None and self.pending_feature is None and ('feature',tag) not in self.used and self.features.get(tag)==expected,'Unregistered/duplicate prewrite feature');self.used.add(('feature',tag));self.pending_feature=expected
    def consume_feature(self,point,desired):need(self.pending_feature==(safe.tensor_hash(point),desired),'Unauthorized feature point');self.pending_feature=None
    def begin_preparations(self):need(self.phase=='controls' and self.pending is None and self.pending_feature is None and sum(k[0]=='controls' for k in self.used)==144 and sum(k[0]=='feature' for k in self.used)==36,'All36 known feature maps and144 controls required');self.phase='preparation'
    def begin_prewrite(self):need(self.phase=='preparation' and self.pending is None and self.pending_feature is None and sum(k[0]=='preparation' for k in self.used)==96,'All96 preparation records required');self.phase='prewrite'
    def freeze(self,points):
        need(self.phase=='prewrite' and self.pending is None and self.pending_feature is None and sum(k[0]=='feature' for k in self.used)==84,'All48 new feature records required');need(set(points)=={(r['state_id'],arm) for r in registry() for arm in ARMS},'All144 proposals required')
        for (sid,arm),point in points.items():
            for rnd in (0,1):self.table[('evaluation',(rnd,sid,arm))]=(safe.tensor_hash(point),sid.split('__')[0],True)
        self.phase='evaluation'
    def finish(self):need(self.phase=='evaluation' and self.pending is None and self.pending_feature is None and sum(k[0]=='evaluation' for k in self.used)==288,'All288 writer endpoints required')
