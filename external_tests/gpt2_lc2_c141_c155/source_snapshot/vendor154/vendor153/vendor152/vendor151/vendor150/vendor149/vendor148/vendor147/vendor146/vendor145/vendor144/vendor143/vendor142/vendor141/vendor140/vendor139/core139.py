"""Paired historical preparation diagnostic; no independent generalisation claim."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor138'))
import core138 as previous
from core138 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,RADIUS,BUDGET,head_digests
parent=previous.parent
PROTOCOL='GPT2_LC2_C139_KERNEL_PREPARATION_DIAGNOSTIC_CPU_V1'
INPUT_KEYS=previous.INPUT_KEYS+('C138',);TIMEOUT=3000;ARMS=previous.ARMS

def contract():
    return dict(previous.contract(),pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source138=clean_json((ROOT/'trusted/C138_SOURCE_PINS.json').read_bytes()))

def registry():
    return [dict(state_id=r['state_id'].replace('__c138_local_','__c139_kernel_'),source_state_id=r['state_id'],case=r['case'],parent_id=r['parent_id'],seed=r['seed'],norm=r['norm'],cohort='exposed_C138_direction_paired_diagnostic',dataset_role='development',independent_validation_eligible=False) for r in previous.registry()]

def design():
    return dict(protocol=PROTOCOL,registry=registry(),fixed_development_directions=24,independent_confirmation_states=0,
        inputs='Seven original scientific ZIPs C132-C138, whole-file hashes and C138 delivered source fixed before analysis. Reconstruct C138 coordinates, proposals, preparation gates, endpoint gates, summaries and stored seals. Do not execute archived source.',
        original_diagnostic='Retain all24 C138 states, including invalid preparations; tabulate every preparation gate and separately describe writer endpoints in valid and invalid preparation strata. Strata never replace the all24 denominator.',
        preparation='For each SAME C138 seed: u=standard_normal(seed); t=u-V.T@(V@u); new=float32(parent_prepared + declared_norm*t/||t||). Use the existing frozen C132 shared36-row basis exactly. No new Jacobian, SVD, solver, retraction, gradient, teacher, rejection sampling, shrinkage or resampling. Collapsed directions or collisions stop incomplete; no replacement.',
        geometry_scope='Euclidean orthogonal complement of the fixed C132 shared response row space. The basis is not a certified tangent space at all12 parents or new points; first-order suppression does not guarantee finite preparation acceptance.',
        norms=list(previous.NORMS),seeds='Exactly the24 C138 seeds, now exposed development data',
        freeze='Candidate and design before coordinate generation; all24 kernel-prepared coordinates before model; all48 proposals before any kernel writer endpoint. Preparation outcomes never select a direction, norm, head or evaluation subset.',
        preflight='Replay all12 known parent feature maps and36 native parent points, then all24 archived C138 preparations; compare exact point/gate identity, observed2e-4, KL5e-6. Drift blocks kernel measurements.',
        preparation_repeats='Measure all24 kernel preparations twice at identical coordinates before feature extraction or writer endpoints. Report all gate failures in each round; writer qualification requires both preparations valid. Repeat drift is retained and blocks a positive diagnostic.',
        execution='Extract only prewrite features for all24 kernel points; raw original writer and C137 fixed step24 run unchanged; evaluate both arms twice at48 presealed points. No postwrite features enter proposals. Invalid preparation states remain in all96 endpoint requests.',
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        fixed_requests=dict(known_parent_features=12,known_parent_native_points=36,C138_preparation_replays=24,kernel_preparations=48,kernel_features=24,kernel_writer_endpoints=96,metric_calls=204,feature_calls=36,model_forwards_including_initialisation=517),
        decision_rule='Diagnostic completion is distinct from success. Report kernel preparation all24 pass only if both rounds all24 pass and repeatable. Report frozen writer all24 gates only if preparation all24 and both endpoint rounds all24 pass with consistent controls/repeats. Even success remains exposed-direction development evidence, no candidate selection or deployment.',
        next_rule='If preparation is incomplete or fails, stop this fixed preparation family and retain failures; do not proceed to excluded inputs. If preparations all pass but writer fails, resolve seen-class writer failures on development data. Only a complete development pass permits designing separate prospectively fixed unseen confirmation; LC2 requires its own isolated excluded-input protocol.',
        scope='Historical paired preparation intervention with fixed exposed C138 directions; actual CPU model replay. No fresh confirmation, new-center or excluded-input generalisation claim; no causal attribution of hardware error; chronology/non-access not independently proved.',
        new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,
        candidate_selected=False,writer_deployed=False,excluded_input_states_evaluated=0,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT)

def preparation_gates(m,case,point):
    valid=safe.metric_gates(m,case,point,False)
    return dict(complete_pass=bool(valid),failed_gates=[k for k,v in m['gates'].items() if not v],response_drift=m['response_drift'],KL=m['KL'],minimum_exact_margin=min(m['exact_margins']),minimum_development_margin=min(m['development_margins']))

def gate_table(records):
    bad=Counter(k for r in records for k in r['gates']['failed_gates'])
    return dict(n_requested=24,n_evaluated=len(records),passed=sum(r['gates']['complete_pass'] for r in records),failed_gates=dict(sorted(bad.items())),failed_ids=[r['state_id'] for r in records if not r['gates']['complete_pass']],maximum_response_drift=max((r['metrics']['response_drift'] for r in records),default=None))

def freeze_kernel_coordinates(data,rows=None):
    rows=registry() if rows is None else rows;compare(rows,registry(),'Exact24 diagnostic registry')
    v=farray(data['V'],(len(data['V']),DIM));need(len(v)>0,'Empty frozen basis');need(np.max(abs(v@v.T-np.eye(len(v))))<=1e-8,'Frozen row basis not orthonormal; no repair')
    seen=set(data['historical_hashes'])|{safe.tensor_hash(p) for p in data['C138_fresh'].values()};points={};records=[]
    for r in rows:
        base=data['points'][r['parent_id']]['prepared'];farray(base,(DIM,));need(base.dtype==np.float32,'Float32 parent')
        u=np.random.default_rng(r['seed']).normal(size=DIM);t=u-v.T@(v@u);n=float(np.linalg.norm(t));need(np.isfinite(n) and n>1e-12,'Collapsed kernel direction; no resampling')
        delta=r['norm']*t/n;point=(base.astype(float)+delta).astype(np.float32);h=safe.tensor_hash(point);need(h not in seen,'Kernel coordinate collision; no replacement');seen.add(h)
        actual=point.astype(float)-base.astype(float);distance=float(np.linalg.norm(actual));need(abs(distance-r['norm'])<=1e-5,'Kernel committed radius differs')
        points[r['state_id']]=point;records.append(dict(**r,parent_sha256=safe.tensor_hash(base),source_full_point_sha256=safe.tensor_hash(data['C138_fresh'][r['source_state_id']]),point_sha256=h,actual_distance=distance,ideal_shared_row_component_Linf=float(np.max(abs(v@delta))),committed_shared_row_component_Linf=float(np.max(abs(v@actual))),float32_commit_error_Linf=float(np.max(abs(actual-delta))),normal_projection_L2=float(np.linalg.norm(u-t))))
    need(len(points)==24,'All24 kernel points required');return points,records

def endpoint_summaries(records):
    result={};pairs={}
    for case in ('all24','xor_00','xor_01','xor_11'):
        result[case]={};pairs[case]={}
        for rnd in (0,1):
            q=[r for r in records if r['round']==rnd and (case=='all24' or r['case']==case)];raw={r['state_id']:r for r in q if r['arm']==ARMS[0]};result[case][str(rnd)]={};pairs[case][str(rnd)]={}
            for arm in ARMS:
                rr=[r for r in q if r['arm']==arm];bad=Counter(k for r in rr for k in r['recomputed_gates']['failed_gates'])
                result[case][str(rnd)][arm]=dict(n_requested=24 if case=='all24' else 8,n_evaluated=len(rr),passed=sum(r['recomputed_gates']['complete_pass'] for r in rr),failed_gates=dict(sorted(bad.items())),failed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass']],maximum_response_drift=max((r['metrics']['response_drift'] for r in rr),default=None))
                pairs[case][str(rnd)][arm]=dict(rescued_ids=[r['state_id'] for r in rr if r['recomputed_gates']['complete_pass'] and not raw[r['state_id']]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass'] and raw[r['state_id']]['recomputed_gates']['complete_pass']])
    return result,pairs

class Phase:
    def __init__(self,data,points):
        self.phase='controls';self.pending=None;self.pending_feature=None;self.used=set();self.table={};self.features={};self.points=points
        for sp in parent.registry():
            sid=sp['state_id'];p=data['points'][sid]
            for k,x,w in (('prepared',p['prepared'],False),('raw',p['raw'],True),('candidate',data['C137_points'][sid],True)):self.table[('controls',(sid,k))]=(safe.tensor_hash(x),sp['case'],w)
            self.features[('controls',sid)]=(safe.tensor_hash(p['prepared']),int(sp['case'][-2])^int(sp['case'][-1]))
        for sp in registry():
            sid=sp['state_id'];case=sp['case'];self.table[('full_replay',sid)]=(safe.tensor_hash(data['C138_fresh'][sp['source_state_id']]),case,False)
            for rnd in (0,1):self.table[('kernel_preparation',(rnd,sid))]=(safe.tensor_hash(points[sid]),case,False)
            self.features[('prewrite',sid)]=(safe.tensor_hash(points[sid]),int(case[-2])^int(case[-1]))
    def metric(self,key,point,case,write):
        tag=(self.phase,key);expected=(safe.tensor_hash(point),case,write)
        need(self.pending is None and self.pending_feature is None and tag not in self.used and self.table.get(tag)==expected,'Unregistered/repeated native point or phase');self.used.add(tag);self.pending=expected
    def consume(self,point,case,write):need(self.pending==(safe.tensor_hash(point),case,write),'Unauthorized native point');self.pending=None
    def feature(self,sid,point,desired):
        tag=(self.phase,sid);expected=(safe.tensor_hash(point),desired)
        need(self.phase in ('controls','prewrite') and self.pending is None and self.pending_feature is None and self.features.get(tag)==expected and ('feature',tag) not in self.used,'Unregistered prewrite feature');self.used.add(('feature',tag));self.pending_feature=expected
    def consume_feature(self,point,desired):need(self.pending_feature==(safe.tensor_hash(point),desired),'Unauthorized feature point');self.pending_feature=None
    def advance(self,to):
        order={'controls':('full_replay',36,12),'full_replay':('kernel_preparation',24,12),'kernel_preparation':('prewrite',48,12)}
        need(self.phase in order,'Invalid phase transition');n,metrics,features=order[self.phase]
        need(to==n and self.pending is None and self.pending_feature is None and sum(k[0]==self.phase for k in self.used)==metrics and sum(k[0]=='feature' for k in self.used)==features,'Incomplete previous phase');self.phase=to
    def freeze(self,points):
        need(self.phase=='prewrite' and self.pending is None and self.pending_feature is None and sum(k[0]=='feature' for k in self.used)==36,'All24 prewrite features required')
        need(set(points)=={(r['state_id'],arm) for r in registry() for arm in ARMS},'All48 proposals required')
        for (sid,arm),p in points.items():
            for rnd in (0,1):self.table[('evaluation',(rnd,sid,arm))]=(safe.tensor_hash(p),sid.split('__')[0],True)
        self.phase='evaluation'
    def finish(self):need(self.phase=='evaluation' and self.pending is None and self.pending_feature is None and sum(k[0]=='evaluation' for k in self.used)==96,'All96 endpoints required')
