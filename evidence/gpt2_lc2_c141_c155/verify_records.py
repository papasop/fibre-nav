"""CPU compact-evidence checks. No Torch/HF, model replay, training or archived code execution."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,re,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
HERE=Path(__file__).resolve().parent
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def tensor_digest(a):return digest(np.asarray(a).tobytes())
def same(a,b,msg):need(a==b,msg)
def close(a,b,msg):need(np.allclose(a,b,rtol=1e-10,atol=1e-10),msg)
def member(base,index,stage,name):
    row=index[stage][name];p=(base/row['path']).resolve();need(p.is_relative_to(base.resolve()),'Member path outside evidence root');raw=p.read_bytes();same(digest(raw),row['sha256'],'Compact member hash: '+stage+'/'+name);return raw
def jmember(base,index,stage,name):return json.loads(member(base,index,stage,name))
def arrays(base,index,stage,name):
    import io
    with np.load(io.BytesIO(member(base,index,stage,name)),allow_pickle=False) as f:o={k:f[k] for k in f.files}
    need(all(v.dtype.kind in 'fbiu' and np.isfinite(v).all() for v in o.values()),'Finite numeric tensor only: '+stage+'/'+name);return o
def native_gates(m,case,point,write=True):
    need(case in ('xor_00','xor_01','xor_11'),'No excluded class relabel');x,y=int(case[-2]),int(case[-1]);truth=[x,y,1,x^y if write else 1-(x^y)];o=np.asarray(m['observed'],float);need(o.shape==(22,) and np.isfinite(o).all(),'Finite22 observables');need(np.isfinite(m['KL']) and m['KL']>=0,'Finite KL');bits=(o[6:10]>=0).astype(int).tolist();margins=(2*np.tile(truth,4)-1)*o[6:];drift=float(max(abs(o[:6])));same(m['theta_sha256'],tensor_digest(point),'Actual physical coordinate binding');same(m['bits'],bits,'Bits');close(m['exact_margins'],margins[:4],'Exact margins');close(m['development_margins'],margins[4:],'Development margins');same(m['response_drift'],drift,'Response drift')
    g=dict(bits_match=bits==truth,response=drift<=.002,KL=m['KL']<=.005,exact_read_margin=bool(np.all(margins[:4]>=.05)),development_read_margin=bool(np.all(margins[4:]>=.05)))
    if write:g.update(output_flip=bits[3]!=1-(x^y),unselected_preserved=bits[:3]==[x,y,1])
    same(m['gates'],g,'Original native gates');same(m['pass'],all(g.values()),'Original native pass');return g,drift,margins
def endpoint(m,case,point,prepared,valid):
    g,drift,margin=native_gates(m,case,point);radius=float(np.linalg.norm(point.astype(float)-prepared.astype(float)));within=radius<=20.+1e-5;failed=[k for k in ['bits_match','output_flip','unselected_preserved','response','KL','exact_read_margin','development_read_margin'] if not g[k]]+([] if within else ['proposal_radius'])
    if not valid:failed.insert(0,'preparation')
    return dict(complete_pass=not failed,failed_gates=failed,response_drift=drift,response_residual=list(m['observed'][:6]),KL=m['KL'],minimum_exact_margin=float(min(margin[:4])),minimum_development_margin=float(min(margin[4:])),actual_delta_norm=radius,proposal_radius_passed=within)
def assert_gate_record(g,want):
    same(set(g),set(want),'Gate-record field inventory')
    for k,v in want.items():
        if isinstance(v,(float,list)) and k!='failed_gates':close(g[k],v,'Gate-record '+k)
        else:same(g[k],v,'Gate-record '+k)
def check_c141(base,index):
    st='C141';spec=jmember(base,index,st,'ALL48_CONFIRMATION_REGISTRY.json');same(len(spec),48,'All48 declared');same(len({s['state_id'] for s in spec}),48,'Unique48');prepared=arrays(base,index,st,'ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz');points=arrays(base,index,st,'ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz');same(set(prepared),{s['state_id'] for s in spec},'All48 physical preparations');arms=['raw_control','C137_frozen_step24','C140_frozen_step12'];same(set(points),{s['state_id']+'__'+a for s in spec for a in arms},'All144 physical proposals');pre=jmember(base,index,st,'ALL96_NEW_PREPARATIONS.json');same([(r['round'],r['state_id']) for r in pre],[(n,s['state_id']) for n in (0,1) for s in spec],'All96 preparations retained');valid={s['state_id']:True for s in spec}
    for r in pre:
        g,_,_=native_gates(r['metrics'],r['case'],prepared[r['state_id']],False);same(r['preparation_valid'],all(g.values()),'Preparation status');valid[r['state_id']] &= r['preparation_valid']
    rows=jmember(base,index,st,'ALL288_CONFIRMATION_ENDPOINTS.json');same([(r['round'],r['state_id'],r['arm']) for r in rows],[(n,s['state_id'],a) for n in (0,1) for s in spec for a in arms],'All288 endpoints retained');summary=jmember(base,index,st,'SUMMARY.json');recomputed={}
    for r in rows:
        sid=r['state_id'];g=endpoint(r['metrics'],r['case'],points[sid+'__'+r['arm']],prepared[sid],valid[sid]);assert_gate_record(r['recomputed_gates'],g)
    for rnd in (0,1):
        recomputed[str(rnd)]={}
        for arm in arms:
            rr=[r for r in rows if r['round']==rnd and r['arm']==arm];want=summary['confirmation_summaries']['all48'][str(rnd)][arm];same(len(rr),48,'48 per arm/round');same(want['passed'],sum(r['recomputed_gates']['complete_pass'] for r in rr),'C141 pass count');same(want['failed_ids'],[r['state_id'] for r in rr if not r['recomputed_gates']['complete_pass']],'C141 failure denominator');same(want['failed_gates'],dict(Counter(k for r in rr for k in r['recomputed_gates']['failed_gates'])),'C141 failed gates');recomputed[str(rnd)][arm]=want['passed']
    same(summary['local_confirmation_passed'],False,'Retain C141 failure');same(summary['LC2_closed'],False,'LC2 open');return recomputed
def check_training(base,index,stage):
    spec=jmember(base,index,stage,'ALL84_TRAINING_REGISTRY_FROZEN.json');same(len(spec),84,'All84 training');need(all(s['used_for_training'] and not s['independent_validation_eligible'] for s in spec),'Exposed training identities retained');data=arrays(base,index,stage,'ALL84_TRAINING_INPUTS.npz');V=arrays(base,index,stage,'FROZEN_SHARED_ROW_BASIS.npz')['V'];need(V.shape==(36,36864),'Original V shape');W=arrays(base,index,stage,'ALL84_PREWRITE_ROUTING.npz')['weights'];need(W.shape==(84,3),'Original weights');need(np.all(W>=0) and np.max(abs(W.sum(1)-1))<1e-14,'Convex routing');same(set(data),{s['state_id']+'__'+k for s in spec for k in ['prepared','raw','features']},'Full84 input arrays')
    if stage=='C148':
        arms=['raw_control','C146_frozen_step12','global_continuation','context_routed'];init=arrays(base,index,stage,'INITIAL_C146_CONTROL.npz')['a'];final={a:arrays(base,index,stage,'FINAL_'+a+'_STEP12.npz')['parameters'] for a in arms[2:]}
    else:
        arms=['raw_control','C148_frozen_routed_step12','routed_diagonal_continuation','routed_cyclic_coupling'];init=arrays(base,index,stage,'INITIAL_C148_ROUTED_CONTROL.npz')['A'];final={a:arrays(base,index,stage,'FINAL_'+a+'_STEP12.npz')['parameters'] for a in arms[2:]}
    points={};effective={}
    for i,s in enumerate(spec):
        sid=s['state_id'];pre,raw=data[sid+'__prepared'],data[sid+'__raw'];need(pre.shape==raw.shape==(36864,) and pre.dtype==raw.dtype==np.float32,'Physical input dtype/shape');need(data[sid+'__features'].shape==(3211,),'Full3211 feature vector');u=V@(raw.astype(float)-pre.astype(float))
        for arm in arms:
            if arm==arms[0]:e=np.zeros(36)
            elif arm==arms[1]:e=init if stage=='C148' else W[i]@init
            elif stage=='C148':e=final[arm] if arm=='global_continuation' else W[i]@final[arm]
            elif arm=='routed_diagonal_continuation':e=W[i]@final[arm]
            else:e=np.stack([W[i]@final[arm][:,j,:] for j in range(2)])
            need(np.max(abs(e))<=.125+1e-15,'Frozen effective box');shift=e*u if e.ndim==1 else e[0]*u+e[1]*np.roll(u,1);point=raw.copy() if arm==arms[0] else (raw.astype(float)+V.T@shift).astype(np.float32);points[(sid,arm)]=point;effective[(sid,arm)]=e
    rows=jmember(base,index,stage,'ALL672_FINAL_TRAINING_ENDPOINTS.json');same([(r['round'],r['state_id'],r['arm']) for r in rows],[(n,s['state_id'],a) for n in (0,1) for s in spec for a in arms],'All672 endpoint denominator');summary=jmember(base,index,stage,'SUMMARY.json');result={}
    for r in rows:
        sid=r['state_id'];g=endpoint(r['metrics'],r['case'],points[(sid,r['arm'])],data[sid+'__prepared'],True);assert_gate_record(r['gates'],g);same(r['used_for_training'],True,'No confirmation relabel')
    for rnd in (0,1):
        result[str(rnd)]={}
        for arm in arms:
            rr=[r for r in rows if r['round']==rnd and r['arm']==arm];want=summary['final_training_summaries']['all84'][str(rnd)][arm];same(len(rr),84,'All84 in each arm');same(want['passed'],sum(r['gates']['complete_pass'] for r in rr),'Training pass count');same(want['failed_ids'],[r['state_id'] for r in rr if not r['gates']['complete_pass']],'All failed states retained');same(want['failed_gates'],dict(Counter(k for r in rr for k in r['gates']['failed_gates'])),'All failed gates');result[str(rnd)][arm]=want['passed']
    need(not any(summary['training_qualified'].values()),'Retain failed training qualification');same(summary['LC2_closed'],False,'LC2 open');return result
def verify(base=HERE):
    base=Path(base);idx=read(base/'MEMBER_INDEX.json');prov=read(base/'ARCHIVE_PROVENANCE.json')
    for stage,files in idx.items():
        for n,row in files.items():member(base,idx,stage,n)
        if files:
            original_manifest=jmember(base,idx,stage,'RESULT_MANIFEST.json')
            for n,row in files.items():
                if n!='RESULT_MANIFEST.json':same(original_manifest[n],row['sha256'],'Original manifest/member relation')
    results=dict(C141=check_c141(base,idx),C148=check_training(base,idx,'C148'),C150=check_training(base,idx,'C150'))
    a=jmember(base,idx,'C141','SUMMARY.json');b=jmember(base,idx,'C142','SUMMARY.json');same(b['confirmation_summaries'],jmember(base,idx,'C142','RECOMPUTED_CONFIRMATION_SUMMARIES.json'),'C142 original summary/duplicate');same(b['confirmation_summaries']['all48'],a['confirmation_summaries']['all48'],'C142 C141 all48 linkage');same(b['parent_decision'],a['decision'],'C142 parent decision');same(b['local_confirmation_passed'],False,'C142 retains failure');same(b['new_model_forwards'],0,'C142 is archive-only')
    for stage in ['C154','C155']:
        folder=base/'records'/stage.lower();raw=(folder/'execution_log.txt').read_bytes();p=read(folder/'LOG_PROVENANCE.json');same(digest(raw),p['original_upload_sha256'],'Original supplied log hash');text=raw.decode();m=re.search(r'\{\s*"protocol"\s*:\s*"GPT2_LC2_'+stage+'_',text);need(m is not None,'Log protocol');summary=json.JSONDecoder().raw_decode(text[m.start():])[0];same(summary,read(folder/'SUMMARY_FROM_LOG.json'),'Log-derived summary');same(summary['complete'],True,'Logged completion');same(summary['LC2_closed'],False,'No LC2 closure');same(prov[stage]['evidence_tier'],'source_plus_execution_log_only','Do not upgrade missing archives')
    return dict(complete=True,compact_records_checked=4,log_only_records_checked=2,original_ZIPs_rehashed_by_this_command=False,original_ZIPs_rehashed_during_packaging=['C141','C142','C148','C150'],new_model_forwards=0,raw_writer_independently_reevaluated=False,gradients_independently_verified=False,LC2_closed=False,complete_model_reproduction_established=False,counts=results)
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
