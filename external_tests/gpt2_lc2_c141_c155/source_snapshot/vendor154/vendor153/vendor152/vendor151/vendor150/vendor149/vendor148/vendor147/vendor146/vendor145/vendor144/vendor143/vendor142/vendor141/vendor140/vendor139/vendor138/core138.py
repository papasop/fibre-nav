"""Frozen-writer local new-state confirmation; no training or state repair."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor137'))
import core137 as parent
from core137 import safe,need,sha,dump,clean_json,Archive,farray,exact,close,compare,row_digest,DIM,RADIUS,BUDGET
PROTOCOL='GPT2_LC2_C138_FROZEN_LOCAL_CONFIRMATION_CPU_V2'
INPUT_KEYS=('C132','C133','C134','C135','C136','C137');TIMEOUT=3000
ARMS=('raw_control','C137_frozen_step24');NORMS=(.125,.375);SEED_BASE=138082931

def contract():
    c=parent.contract();c.update(pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),source137=clean_json((ROOT/'trusted/C137_SOURCE_PINS.json').read_bytes()));return c

def registry():
    return [dict(state_id=f"{r['state_id'].split('__')[0]}__c138_local_{i:02d}_n{k:02d}",case=r['case'],parent_id=r['state_id'],seed=SEED_BASE+100*i+k,norm=norm,cohort='prospective_seen_class_local_confirmation',dataset_role='confirmation') for i,r in enumerate(parent.registry()) for k,norm in enumerate(NORMS)]

def design():
    return dict(protocol=PROTOCOL,reader_revision='V2: preserve and validate the original optional geometry fields; no scientific rule changes',registry=registry(),new_local_states=24,states_per_trained_class=8,known_parent_centers=12,
        frozen_candidate='C137 fixed step24 a; original C132 V, raw writer and delta sketch, normalizers and feature formulas unchanged',
        preparation='float32(parent_prepared + norm * standard_normal(seed) / L2_norm); full-chart Euclidean noise, not projected; no derivatives, solving, retraction, teacher, repair, resampling or preparation acceptance selection',
        norms=list(NORMS),seed_base=SEED_BASE,
        generation_freeze='Candidate tensors and design sealed before generating all24 coordinates; all24 coordinates sealed before model initialization; all48 raw/candidate proposals sealed before any new endpoint observation',
        novelty='IDs and seeds new within the bound registries; coordinate hashes disjoint from every prepared/raw/projected/C136/C137 endpoint available in the six bound originals and from this batch. Exhaustive comparison against every historical programme state is not claimed.',
        evaluation='All24 preparations and prewrite features measured; invalid preparations are retained, never replaced. All finite proposals evaluated at fixed points twice for both arms; invalid preparation forces endpoint qualification failure.',
        acceptance_budgets=dict(response=.002,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),
        proposal='Same frozen C137 formula: float32(raw+V.T@(a*(V@(raw-prepared)))); no radial clipping or postwrite response correction',
        controls='All12 known preparations/features/raw/C137 endpoints before new-state model access; feature/raw map2e-6, observables2e-4, KL5e-6, point and gate identity; control drift blocks new-state measurements',
        repeats='Two native endpoint rounds; observational repeat tolerances2e-4 and KL5e-6 plus exact coordinate and gate identity; repetitions are not independent states',
        endpoint_requests=96,preparation_requests=24,feature_requests=24,known_feature_controls=12,known_native_controls=36,requested_model_forwards=421,
        pass_rule='All24 requested states valid and complete gates pass twice; controls and repeats consistent. Raw comparison is descriptive, no choice between arms or parameter adaptation.',
        scope='Local new-state confirmation around known training centers, on xor00/01/11. Not new-center, excluded-input, cross-model, long-sequence or LC2 evidence. Chronology and global historical non-access are not independently certified.',
        new_training_steps=0,new_backward_calls=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,
        teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,excluded_input_states_evaluated=0,LC2_closed=False,LC2_single_fold_passed=False,worker_budget_seconds=TIMEOUT)

def head_digests(data):return dict(a=row_digest(data['a']),V=row_digest(data['V']),writer=safe.head_digest(data['writer']),control=safe.head_digest(data['control']))

def fresh_coordinates(data,rows=None):
    rows=registry() if rows is None else rows;need(rows==registry(),'Unregistered new-state registry')
    known=set(data['historical_hashes']);batch=set();points={};records=[]
    for spec in rows:
        need(spec['case'] in ('xor_00','xor_01','xor_11'),'Excluded input forbidden');base=data['points'][spec['parent_id']]['prepared'];farray(base,(DIM,));need(base.dtype==np.float32,'Float32 parent required')
        direction=np.random.default_rng(spec['seed']).normal(size=DIM);norm=float(np.linalg.norm(direction));need(np.isfinite(norm) and norm>1e-12,'Collapsed noise; no resampling')
        point=(base.astype(float)+spec['norm']*direction/norm).astype(np.float32);h=safe.tensor_hash(point);distance=float(np.linalg.norm(point.astype(float)-base.astype(float)))
        need(h not in known and h not in batch,'Historical or batch coordinate collision; no resampling');need(abs(distance-spec['norm'])<=1e-5,'Committed perturbation radius differs')
        batch.add(h);points[spec['state_id']]=point;records.append(dict(**spec,point_sha256=h,parent_sha256=safe.tensor_hash(base),actual_distance=distance))
    need(len(points)==24 and len({r['seed'] for r in rows})==24,'All24 new states required');return points,records

def gate_record(m,case,point,prepared,valid):
    g=parent.gate_record(m,case,point,prepared)
    if not valid:g['complete_pass']=False;g['failed_gates'].insert(0,'preparation')
    return g

def summaries(records):
    result={};paired={}
    for case in ('all24','xor_00','xor_01','xor_11'):
        result[case]={};paired[case]={}
        for rnd in (0,1):
            subset=[r for r in records if r['round']==rnd and (case=='all24' or r['case']==case)];raw={r['state_id']:r for r in subset if r['arm']==ARMS[0]};result[case][str(rnd)]={};paired[case][str(rnd)]={}
            for arm in ARMS:
                q=[r for r in subset if r['arm']==arm];bad=Counter(k for r in q for k in r['recomputed_gates']['failed_gates'])
                result[case][str(rnd)][arm]=dict(n_requested=24 if case=='all24' else 8,n_evaluated=len(q),passed=sum(r['recomputed_gates']['complete_pass'] for r in q),maximum_response_drift=max((r['metrics']['response_drift'] for r in q),default=None),failed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass']],failed_gates={k:bad[k] for k in ('preparation','bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin','proposal_radius')})
                paired[case][str(rnd)][arm]=dict(rescued_ids=[r['state_id'] for r in q if r['recomputed_gates']['complete_pass'] and not raw[r['state_id']]['recomputed_gates']['complete_pass']],destroyed_ids=[r['state_id'] for r in q if not r['recomputed_gates']['complete_pass'] and raw[r['state_id']]['recomputed_gates']['complete_pass']])
    return result,paired

class Phase:
    def __init__(self,data,fresh):
        self.phase='controls';self.pending=None;self.pending_feature=None;self.used=set();self.controls={};self.features={};self.fresh=fresh;self.proposals={}
        self.fresh_expected={r['state_id']:(safe.tensor_hash(fresh[r['state_id']]),r['case'],False) for r in registry()}
        for sp in parent.registry():
            sid=sp['state_id'];p=data['points'][sid]
            for kind,point,write in (('prepared',p['prepared'],False),('raw',p['raw'],True),('candidate',data['C137_points'][sid],True)):
                self.controls[(sid,kind)]=(safe.tensor_hash(point),sp['case'],write)
            self.features[('controls',sid)]=(safe.tensor_hash(p['prepared']),int(sp['case'][-2])^int(sp['case'][-1]))
        for sp in registry():self.features[('prewrite',sp['state_id'])]=(safe.tensor_hash(fresh[sp['state_id']]),int(sp['case'][-2])^int(sp['case'][-1]))
    def metric(self,key,point,case,write):
        need(self.pending is None and self.pending_feature is None,'Pending access');tag=(self.phase,key);need(tag not in self.used,'Duplicate metric request')
        if self.phase=='controls':expected=self.controls.get(key)
        elif self.phase=='prewrite':expected=self.fresh_expected.get(key)
        else:expected=self.proposals.get(key)
        need(expected==(safe.tensor_hash(point),case,write),'Unregistered native point/phase');self.pending=expected;self.used.add(tag)
    def consume(self,point,case,write):need(self.pending==(safe.tensor_hash(point),case,write),'Unauthorized native model point');self.pending=None
    def feature(self,sid,point,desired):
        key=(self.phase,sid);need(self.pending is None and self.pending_feature is None and self.phase in ('controls','prewrite') and key in self.features and ('feature',key) not in self.used,'Unregistered or repeated feature request')
        need(self.features[key]==(safe.tensor_hash(point),desired),'Wrong prewrite feature point');self.used.add(('feature',key));self.pending_feature=self.features[key]
    def consume_feature(self,point,desired):need(self.pending_feature==(safe.tensor_hash(point),desired),'Unauthorized model feature point');self.pending_feature=None
    def begin_prewrite(self):
        need(self.phase=='controls' and self.pending is None and self.pending_feature is None and sum(k[0]=='controls' for k in self.used)==36 and sum(k[0]=='feature' for k in self.used)==12,'All known controls required');self.phase='prewrite'
    def freeze(self,points):
        need(self.phase=='prewrite' and self.pending is None and self.pending_feature is None and sum(k[0]=='prewrite' for k in self.used)==24 and sum(k[0]=='feature' for k in self.used)==36,'All24 prewrite records required')
        need(set(points)=={(r['state_id'],a) for r in registry() for a in ARMS},'All48 proposals required')
        self.proposals={(rnd,sid,arm):(safe.tensor_hash(p),sid.split('__')[0],True) for (sid,arm),p in points.items() for rnd in (0,1)};self.phase='evaluation'
    def finish(self):need(self.phase=='evaluation' and self.pending is None and sum(k[0]=='evaluation' for k in self.used)==96,'All96 endpoints required')
