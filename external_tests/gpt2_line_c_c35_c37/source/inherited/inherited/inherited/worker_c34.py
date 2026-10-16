"""Actual finite-endpoint oracle compensation, never correction-free writing."""
import argparse,fcntl
import numpy as np
import torch
from c34_protocol import *
from c34_parent import validate
from c34_math import construct,proposal,aggregate
from c34_runtime import Runtime,load,vector,save_tensor,tensor_hash

def saved_state(out,spec):
    out=Path(out);folder=out/'states'/spec['cohort']/spec['id'];rec=read(folder/'COMPLETE.json')
    require(rec['spec']==spec and rec['writer_frozen_sha256']==sha(out/'WRITER_FROZEN.json'),'Saved state identity differs')
    require(rec['generation_retries']==0 and rec['post_write_correction_steps']==0,'C31 state policy differs')
    for n in ('INPUT.pt','prepared.pt','RAW_WRITTEN.pt'):require(sha(folder/n)==rec['files'][n],'Saved C31 tensor bytes differ')
    raw=load(folder/'RAW_WRITTEN.pt');prep=vector(load(folder/'prepared.pt'));inp=load(folder/'INPUT.pt');theta=vector(raw)
    require(tensor_hash(theta)==rec['raw_written']['theta_sha256'] and tensor_hash(prep)==rec['preparation']['theta_sha256'],'Saved C31 endpoint tensor digest differs')
    require(raw['prepared_sha256']==tensor_hash(prep) and raw['input_sha256']==tensor_hash(inp['input']),'Written-state preparation/input binding differs')
    require(inp['spec']==spec and inp['prepared_sha256']==tensor_hash(prep),'Saved input specification differs')
    require(type(rec['preparation_valid']) is bool,'Invalid preparation-validity flag')
    return theta,rec

def check_metrics(m,case):
    e=np.asarray(m['response_residual']);require(e.shape==(6,) and np.isfinite(e).all(),'Invalid measured residual')
    require(set(m['gates'])==set(GATES) and all(type(v) is bool for v in m['gates'].values()),'Measured gate schema differs')
    require(abs(m['drift']-float(np.max(np.abs(e))))<=1e-7 and np.isfinite(m['kl']),'Measured response/KL invalid')
    require(m['gates']['response']==(m['drift']<=.002) and m['gates']['KL']==(m['kl']<=.005),'Measured budget gates differ')
    start=STARTS[case];target=start[:3]+[start[0]^start[1]]
    em=m['evaluation']['exact_margins'];dm=m['evaluation']['heldout_margins']
    require(len(em)==4 and len(dm)==12 and np.isfinite(em+dm).all(),'Measured margin schema differs')
    expected={'bits_match':m['bits']==target,'output_flip':m['bits'][3]!=start[3],
              'unselected_preserved':m['bits'][:3]==start[:3],'exact_read_margin':min(em)>=.05,'development_read_margin':min(dm)>=.05}
    require(all(m['gates'][k]==v for k,v in expected.items()) and m['pass']==all(m['gates'].values()),'Measured endpoint gates inconsistent')

def measure_raw(rt,out,spec):
    theta,old=saved_state(out,spec);folder=Path(out)/'evaluation'/spec['id'];folder.mkdir(parents=True,exist_ok=True);p=folder/'RAW_REPLAY.json'
    binding={'spec':spec,'theta_sha256':tensor_hash(theta),'parent_writer_sha256':sha(Path(out)/'WRITER_FROZEN.json')}
    if p.exists():
        record=read(p);require(record['binding']==binding,'Raw replay cache binding differs');m=record['metrics']
    else:
        start=STARTS[spec['case']];target=start[:3]+[start[0]^start[1]]
        m=rt.evaluate(theta,target,start);check_metrics(m,spec['case'])
        previous=old['raw_written'];err=max(abs(a-b) for a,b in zip(m['response_residual'],previous['response_residual']))
        require(err<=SETTINGS['raw_replay_response_tolerance'] and m['bits']==previous['bits'] and m['gates']==previous['gates'],'Actual C31 raw endpoint replay differs; stop without replacing state')
        dump(p,{'binding':binding,'metrics':m,'parent_response_replay_error':err})
    check_metrics(m,spec['case']);require(m['theta_sha256']==binding['theta_sha256'],'Raw replay endpoint differs')
    return theta,old,m

def build_bases(rt,out,allspec,backup):
    out=Path(out);byid={s['id']:s for s in allspec};basis_ids=SETTINGS['basis_training_ids']
    require(all(s in byid for s in basis_ids),'Declared basis states missing')
    if (out/'BASES_FROZEN.json').exists():return check_frozen(out)
    for case in CASES:
        folder=out/'basis'/case;folder.mkdir(parents=True,exist_ok=True);cp=folder/'COMPLETE.json'
        if cp.exists():
            rec=read(cp)
            require(rec['training_ids']==[x for x in basis_ids if x.startswith(case+'__')],'Basis training IDs changed')
            for n,h in rec['files'].items():require(sha(out/n)==h,'Cached basis/geometry changed')
            print('C34 RESTORE BASIS',case,flush=True);continue
        js=[];bind=[]
        for sid in [x for x in basis_ids if x.startswith(case+'__')]:
            spec=byid[sid];theta,old,m=measure_raw(rt,out,spec)
            path=folder/(sid+'_J.pt');J=rt.jacobian(theta,path);js.append(J)
            require(J.shape==(6,36864) and np.isfinite(J).all(),'Finite6x36864 response Jacobian required')
            bind.append({'id':sid,'theta_sha256':tensor_hash(theta),'J_sha256':sha(path)})
            print('C34 OFFLINE WRITTEN-STATE GEOMETRY',sid,'raw_R',m['drift'],flush=True);snapshot(out,backup)
        bases,diagnostic=construct(js)
        save_tensor(folder/'BASES.pt',{k:torch.from_numpy(v.copy()) for k,v in bases.items()})
        files={p.relative_to(out).as_posix():sha(p) for p in folder.glob('*.pt')}
        dump(cp,{'training_ids':[x['id'] for x in bind],'raw_state_binding':bind,'diagnostic':diagnostic,'files':files})
        snapshot(out,backup)
    files={p.relative_to(out).as_posix():sha(p) for p in (out/'basis').rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    dump(out/'BASES_FROZEN.json',{'settings':SETTINGS,'files':files,'basis_training_ids':basis_ids,
         'writer_frozen_sha256':sha(out/'WRITER_FROZEN.json'),'parent_provenance_sha256':sha(out/'PARENT_PROVENANCE.json'),
         'frozen_before_nonbasis_endpoint_evaluation':True,'nonbasis_geometry_used':False,
         'coefficient_learning_performed':False,'privileged_measured_response_required':True})
    rt.invariant();fr=check_frozen(out);snapshot(out,backup);return fr

def check_frozen(out):
    out=Path(out);fr=read(out/'BASES_FROZEN.json')
    require(fr['settings']==SETTINGS and fr['basis_training_ids']==SETTINGS['basis_training_ids'],'Frozen basis specification differs')
    require(fr['writer_frozen_sha256']==sha(out/'WRITER_FROZEN.json') and fr['parent_provenance_sha256']==sha(out/'PARENT_PROVENANCE.json'),'Frozen writer/parent binding differs')
    require(fr['frozen_before_nonbasis_endpoint_evaluation'] and not fr['nonbasis_geometry_used'],'Basis fit/evaluation separation changed')
    files={p.relative_to(out).as_posix() for p in (out/'basis').rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    require(files==set(fr['files']),'Basis file coverage differs')
    for n,h in fr['files'].items():require(sha(out/n)==h,'Frozen basis or geometry changed: '+n)
    return fr

def evaluate_state(rt,out,spec):
    out=Path(out);check_frozen(out);folder=out/'evaluation'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    frozen_sha=sha(out/'BASES_FROZEN.json');cp=folder/'COMPLETE.json'
    if cp.exists():
        rec=read(cp);require(rec['spec']==spec and rec['bases_frozen_sha256']==frozen_sha,'Completed state cache binding differs')
        for n,h in rec['files'].items():require(sha(out/n)==h,'Saved actual endpoint/metrics changed')
        for arm in ARMS:check_metrics(rec['arms'][arm]['metrics'],spec['case'])
        return rec
    theta,old,raw=measure_raw(rt,out,spec)
    bt=load(out/'basis'/spec['case']/'BASES.pt')
    require(set(bt)=={'common1','response6'} and bt['common1'].shape==(36864,1) and bt['response6'].shape==(36864,6),'Frozen basis tensor shape differs')
    require(all(t.dtype==torch.float64 and bool(torch.isfinite(t).all()) for t in bt.values()),'Frozen basis dtype/finiteness differs')
    bases={k:t.numpy() for k,t in bt.items()};arms={'raw':{'metrics':raw,'post_write_correction_steps':0}}
    files={};start=STARTS[spec['case']];target=start[:3]+[start[0]^start[1]]
    for arm in ARMS[1:]:
        delta,info=proposal(bases,arm,raw['response_residual']);dest=folder/(arm+'.pt');md=folder/(arm+'.json')
        expected=vector(theta+torch.from_numpy(delta.astype(np.float32)))
        binding={'spec':spec,'raw_theta_sha256':tensor_hash(theta),'bases_frozen_sha256':frozen_sha,'arm':arm,'expected_theta_sha256':tensor_hash(expected)}
        if dest.exists():
            saved=load(dest);require(saved['binding']==binding and tensor_hash(vector(saved))==tensor_hash(expected),'Saved corrected endpoint differs from fixed oracle rule')
        else:save_tensor(dest,{'theta':expected,'binding':binding,'proposal':info})
        if md.exists():
            r=read(md);require(r['binding']==binding and r['endpoint_file_sha256']==sha(dest),'Saved corrected metric binding differs');m=r['metrics']
        else:
            m=rt.evaluate(expected,target,start);check_metrics(m,spec['case'])
            dump(md,{'binding':binding,'endpoint_file_sha256':sha(dest),'metrics':m})
        require(m['theta_sha256']==tensor_hash(expected),'Metric/corrected state digest differs');check_metrics(m,spec['case'])
        displacement=expected.double()-theta.double()
        record={'metrics':m,'proposal':info,'actual_float32_displacement_norm':float(displacement.norm()),
                'measured_raw_residual_used_for_coefficients':True,'offline_response_correction_steps':1}
        jfile=out/'basis'/spec['case']/(spec['id']+'_J.pt')
        if jfile.exists():
            J=load(jfile)['J'].numpy();linear=np.array(raw['response_residual'])+J@displacement.numpy()
            record['basis_training_only_linearised_residual']=linear.tolist()
            record['basis_training_only_nonlinear_remainder_Linf']=float(np.max(np.abs(np.asarray(m['response_residual'])-linear)))
        arms[arm]=record
        for p in (dest,md):files[p.relative_to(out).as_posix()]=sha(p)
    files[(folder/'RAW_REPLAY.json').relative_to(out).as_posix()]=sha(folder/'RAW_REPLAY.json')
    rec={'spec':spec,'cohort':group(spec),'preparation_valid':old['preparation_valid'],
         'reference_validity_scope':'C31 preparation validity only; no teacher-write admissibility certificate',
         'bases_frozen_sha256':frozen_sha,'arms':arms,'files':files,
         'privileged_raw_response_used':True,'correction_free_writer_tested':False}
    dump(cp,rec);rt.invariant()
    print('C34 FINITE ENDPOINT',spec['id'],group(spec),'R',
          {a:arms[a]['metrics']['drift'] for a in ARMS},'pass',
          {a:bool(rec['preparation_valid'] and arms[a]['metrics']['pass']) for a in ARMS},flush=True)
    return rec

def main(out,backup='',runtime_factory=Runtime):
    out=Path(out);root=Path(__file__).resolve().parent;check_sources(root)
    require(read(out/'identity.json')=={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)},'C34 identity/source/settings differ')
    specs=validate(out,root)
    if (out/'SUMMARY.json').exists():
        check_frozen(out)
        for s in specs:
            r=read(out/'evaluation'/s['id']/'COMPLETE.json')
            for n,h in r['files'].items():require(sha(out/n)==h,'Completed archive state file changed')
        print((out/'SUMMARY.json').read_text(),flush=True);return read(out/'SUMMARY.json')
    rt=runtime_factory(out,root/'inherited');build_bases(rt,out,specs,backup);records=[]
    for i,spec in enumerate(specs,1):
        records.append(evaluate_state(rt,out,spec))
        if i%4==0:snapshot(out,backup)
    cohorts={name:aggregate([r for r in records if r['cohort']==name]) for name in ('basis_training','other_training','inspected_development')}
    require([cohorts[n]['raw']['n_requested'] for n in cohorts]==[12,36,16],'Requested cohorts changed')
    expected=read(out/'parent/C31_SUMMARY.json')
    require(cohorts['basis_training']['raw']['complete_passed']+cohorts['other_training']['raw']['complete_passed']==expected['new_training']['all']['raw_complete_write_passed'],'C31 training raw pass recount differs')
    require(cohorts['inspected_development']['raw']['complete_passed']==expected['prospective_local_development']['all']['raw_complete_write_passed'],'C31 development raw pass recount differs')
    rt.invariant();check_frozen(out)
    summary={'protocol':PROTOCOL,'complete':True,'decision':'C34_LOW_DIMENSIONAL_ORACLE_COMPARISON_COMPLETE',
       'cohorts':cohorts,'by_case':{g:{c:aggregate([r for r in records if r['cohort']==g and r['spec']['case']==c]) for c in CASES} for g in cohorts},
       'basis_diagnostics':{c:read(out/'basis'/c/'COMPLETE.json')['diagnostic'] for c in CASES},
       'bases_frozen_sha256':sha(out/'BASES_FROZEN.json'),'original_writer_sha256':sha(out/'WRITER_FROZEN.json'),
       'all64_requested_states_retained':len(records)==64,'original_C25_C27_writer_unchanged':True,
       'offline_written_state_response_Jacobians':12,'offline_basis_SVD_calls':4,
       'new_writer_training_steps':0,'learned_coefficient_model_tested':False,
       'privileged_raw_endpoint_response_used_for_coefficients':True,
       'actual_finite_oracle_corrected_endpoints':128,'response_correction_attempts_per_state_arm':1,
       'backtracking_steps':0,'new_states_generated':0,'fresh_evaluation':False,
       'correction_free_writer_tested':False,'internal_parameter_update':False,'controller_dispatch_tested':False,'LC1_closed':False,
       'scope':SETTINGS['scope'],
       'interpretation':'Tests finite correction along training-derived fixed bases when actual raw residuals are supplied. Success does not establish learned coefficients, correction-free writes, unseen input classes, new local-state confirmation or native updates.'}
    dump(out/'SUMMARY.json',summary)
    concise={k:v for k,v in summary.items() if k not in ('by_case','basis_diagnostics')}
    dump(out/'CONSOLE_SUMMARY.json',concise);snapshot(out,backup);print(json.dumps(concise,indent=2),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--drive-dir',default='');a=p.parse_args()
    lock=open(Path(a.out).parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a.out,a.drive_dir)
