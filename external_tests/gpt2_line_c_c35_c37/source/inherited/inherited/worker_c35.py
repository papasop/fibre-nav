"""Prewrite-only frozen proposals, followed by separate physical endpoint audits."""
import argparse,fcntl,time
import numpy as np
import torch
from c35_protocol import *
from c35_parent import validate
from c35_fit import fit_all,check_frozen,records,input_data
from c35_writer import ComposedWriter
from c31_runtime import Runtime,load,save_tensor,vector,tensor_hash
from worker_c34 import saved_state,check_metrics
from geometry import null_project

def prepare_new(rt,out,spec):
    out=Path(out);check_frozen(out);folder=out/'new_states'/spec['id'];folder.mkdir(parents=True,exist_ok=True);cp=folder/'COMPLETE.json'
    if cp.exists():
        rec=read(cp);require(rec['spec']==spec and rec['coefficients_frozen_sha256']==sha(out/'COEFFICIENTS_FROZEN.json'),'New preparation freeze/spec differs')
        for n,h in rec['files'].items():require(sha(folder/n)==h,'Saved new preparation changed')
        return vector(load(folder/'prepared.pt')),load(folder/'INPUT.pt'),rec['preparation_valid']
    basefolder=out/'states/new_training'/spec['base_id'];base=vector(load(basefolder/'prepared.pt'));qpath=folder/'PERTURBATION.pt'
    if qpath.exists():
        saved=load(qpath);require(saved['spec']==spec and saved['base_sha256']==tensor_hash(base),'Saved perturbation differs');q=vector(saved['theta'])
    else:
        J=rt.jacobian(base,out/'preparation_geometry'/(spec['base_id']+'.pt'))
        raw=np.random.default_rng(spec['seed']).normal(size=36864)
        q,info=null_project(J,raw,SETTINGS['preparation_null_relative_cutoff'],SETTINGS['preparation_null_absolute_cutoff'])
        norm=float(np.linalg.norm(q));require(norm>0,'Zero fixed perturbation, do not resample')
        q=torch.from_numpy((q*(spec['norm']/norm)).astype(np.float32))
        save_tensor(qpath,{'theta':q,'spec':spec,'base_sha256':tensor_hash(base),'null_projection':info})
    ppath=folder/'prepared.pt';initial=base+q
    if ppath.exists():
        p=load(ppath);require(p['initial_sha256']==tensor_hash(initial) and p['spec']==spec,'Saved new preparation differs');prep=vector(p)
    else:
        prep,history,seconds=rt.correct_preparation(initial,folder/'preparation_correction')
        save_tensor(ppath,{'theta':prep,'initial_sha256':tensor_hash(initial),'spec':spec})
        dump(folder/'PREPARATION_HISTORY.json',{'history':history,'seconds':seconds,'prewrite_state_preparation_only':True})
    start=STARTS[spec['case']];pm=rt.evaluate(prep,start)
    desired=start[0]^start[1];ipath=folder/'INPUT.pt'
    if ipath.exists():
        inp=load(ipath);require(inp['spec']==spec and inp['prepared_sha256']==tensor_hash(prep) and inp['desired_bit']==desired,'Saved new input differs')
    else:
        inp={**rt.fresh_input(prep,desired),'spec':spec,'prepared_sha256':tensor_hash(prep),'desired_bit':desired};save_tensor(ipath,inp)
    valid=bool(pm['pass'] and int(inp['route'])==CASES.index(spec['case']))
    rec={'spec':spec,'coefficients_frozen_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),'preparation':pm,'preparation_valid':valid,
         'generation_retries':0,'replacement_used':False,'teacher_reference_constructed':False,
         'files':{n:sha(folder/n) for n in ('prepared.pt','INPUT.pt','PERTURBATION.pt')}}
    dump(cp,rec);print('C35 NEW PREPARATION',spec['id'],'valid',valid,flush=True)
    return prep,inp,valid

def proposals(rt,head,out,spec,prep,inp):
    """All three points are computed and sealed before the caller may audit any."""
    out=Path(out);check_frozen(out);folder=out/'proposals'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    require(inp['input'].shape==(3147,) and inp['input'].dtype==torch.float32 and bool(torch.isfinite(inp['input']).all()),'Invalid prewrite input')
    require(inp['extras'].shape==(70,) and inp['extras'].dtype==torch.float32 and bool(torch.isfinite(inp['extras']).all()),'Invalid original prewrite extras')
    require(inp['prepared_sha256']==tensor_hash(prep),'Input belongs to another preparation')
    binding={'spec':spec,'prepared_sha256':tensor_hash(prep),'input_sha256':tensor_hash(inp['input']),
             'extras_sha256':tensor_hash(inp['extras']),'coefficients_frozen_sha256':sha(out/'COEFFICIENTS_FROZEN.json')}
    path=folder/'POINTS.pt';ready=folder/'READY.json'
    if ready.exists():
        mark=read(ready);require(mark['binding']==binding and mark['points_sha256']==sha(path),'Sealed proposal binding differs')
        value=load(path)
    else:
        rt.sync();t0=time.perf_counter()
        with torch.no_grad():
            points,diag=head(prep.to(rt.device),inp['input'].to(rt.device),inp['route'].to(rt.device),
                             inp['extras'][:6].to(rt.device).double()*.002)
        rt.sync();seconds=time.perf_counter()-t0;rt.invariant()
        require(set(points)==set(ARMS) and all(t.shape==(36864,) and t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in points.values()),'Invalid finite composed proposal')
        value={'binding':binding,'points':{a:t.detach().cpu().clone() for a,t in points.items()},
               'diagnostics':{a:{k:(v.detach().cpu().double().tolist() if v.ndim else float(v)) for k,v in dd.items()} for a,dd in diag.items()},
               'proposal_seconds_all_arms':seconds,'core_forward_calls':1,'all_points_generated_before_endpoint_audit':True,
               'intermediate_candidate_forward':False,'postwrite_response_used':False}
        save_tensor(path,value)
        dump(ready,{'binding':binding,'points_sha256':sha(path),'no_endpoint_audit_yet':True,
                    'all_points_generated_before_endpoint_audit':True})
    require(value['binding']==binding and value['all_points_generated_before_endpoint_audit'] and not value['postwrite_response_used'],'Proposal policy differs')
    for a,t in value['points'].items():vector(t)
    return value,path,ready

def evaluate(rt,head,out,spec,prep,inp,valid,cohort,archived_raw=None):
    out=Path(out);folder=out/'results'/spec['id'];folder.mkdir(parents=True,exist_ok=True);cp=folder/'COMPLETE.json'
    if cp.exists():
        rec=read(cp);require(rec['spec']==spec and rec['coefficients_frozen_sha256']==sha(out/'COEFFICIENTS_FROZEN.json') and rec['preparation_valid']==valid,'Saved result binding differs')
        for n,h in rec['files'].items():require(sha(out/n)==h,'Saved result/actual proposal changed')
        for arm in ARMS:check_metrics(rec['arms'][arm]['metrics'],spec['case'])
        return rec
    value,pfile,ready=proposals(rt,head,out,spec,prep,inp)
    if archived_raw is not None:require(torch.equal(value['points']['raw'],archived_raw),'Frozen core proposal does not exactly reproduce archived C31 point; stop without replacing it')
    frozen_ready_sha=sha(ready);start=STARTS[spec['case']];target=start[:3]+[start[0]^start[1]];arms={};files={}
    for arm in ARMS:
        theta=vector(value['points'][arm]);mp=folder/(arm+'.json')
        if mp.exists():
            rec=read(mp);require(rec['theta_sha256']==tensor_hash(theta) and rec['ready_sha256']==frozen_ready_sha,'Saved audit binding differs');m=rec['metrics']
        else:
            m=rt.evaluate(theta,target,start);check_metrics(m,spec['case'])
            dump(mp,{'theta_sha256':tensor_hash(theta),'ready_sha256':frozen_ready_sha,'metrics':m})
        check_metrics(m,spec['case']);require(m['theta_sha256']==tensor_hash(theta),'Actual audited endpoint differs')
        arms[arm]={'metrics':m,'complete_accepted':bool(valid and m['pass'])}
        if arm in value['diagnostics']:arms[arm]['proposal_diagnostic']=value['diagnostics'][arm]
        files[mp.relative_to(out).as_posix()]=sha(mp)
    require(sha(ready)==frozen_ready_sha,'Endpoint audit changed a sealed proposal')
    for p in (pfile,ready):files[p.relative_to(out).as_posix()]=sha(p)
    rec={'spec':spec,'cohort':cohort,'preparation_valid':valid,'arms':arms,'files':files,
         'coefficients_frozen_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),
         'one_proposal_per_arm':True,'online_write_Jacobians':0,'online_write_SVD_calls':0,
         'online_write_correction_steps':0,'online_backtracking':0,'teacher_fallback_used':False,
         'all_candidates_sealed_before_first_endpoint_forward':True,'postwrite_response_used_for_proposal':False,
         'reference_validity_scope':'Preparation validity; no new teacher-write admissibility certificate'}
    dump(cp,rec);rt.invariant()
    print('C35 ENDPOINT',cohort,spec['id'],'R',{a:arms[a]['metrics']['drift'] for a in ARMS},'pass',{a:arms[a]['complete_accepted'] for a in ARMS},flush=True)
    return rec

def counts(rs):
    require(rs,'Empty requested cohort');ans={}
    for arm in ARMS:
        ms=[r['arms'][arm]['metrics'] for r in rs]
        ans[arm]={'n_requested':len(rs),'valid_preparations':sum(r['preparation_valid'] for r in rs),
          'complete_passed':sum(r['arms'][arm]['complete_accepted'] for r in rs),
          'gate_counts':{g:sum(m['gates'][g] for m in ms) for g in GATES},
          'maximum_response_drift':max(m['drift'] for m in ms),'median_response_drift':float(np.median([m['drift'] for m in ms])),
          'clipped_compensations':sum(r['arms'][arm].get('proposal_diagnostic',{}).get('radial_scale',1.)<1. for r in rs)}
    return ans

def completed(out,specs_):
    out=Path(out);s=read(out/'SUMMARY.json');check_frozen(out)
    allrecords=[]
    for spec in specs_:
        rec=read(out/'results'/spec['id']/'COMPLETE.json');require(rec['spec']==spec,'Completed result specification differs')
        for n,h in rec['files'].items():require(sha(out/n)==h,'Completed result bytes changed')
        allrecords.append(rec)
    for group_ in ('training','inspected_development','new_local'):
        require(counts([r for r in allrecords if r['cohort']==group_])==s['cohorts'][group_],'Completed summary recount differs')
    return s

def main(out,backup='',runtime_factory=Runtime):
    out=Path(out);root=Path(__file__).resolve().parent;check_sources(root)
    require(read(out/'identity.json')=={'protocol':PROTOCOL,'settings':SETTINGS,'source_hashes':sources(root)},'C35 identity/source/settings differ')
    fixed=validate(out,root)
    require(read(out/'NEW_STATE_SPECS.json')==specs(),'New state seeds/specifications changed')
    if (out/'SUMMARY.json').exists():
        s=completed(out,fixed['new_training']+fixed['prospective']+specs());print((out/'CONSOLE_SUMMARY.json').read_text(),flush=True);return s
    fit_all(out,backup);check_frozen(out)
    rt=runtime_factory(out,root/'inherited/inherited');head=ComposedWriter(rt.writer,records(out),rt.device);head_versions={n:t._version for n,t in head.named_buffers()}
    allrecords=[]
    for cohort,ss in (('training',fixed['new_training']),('inspected_development',fixed['prospective'])):
        for i,spec in enumerate(ss,1):
            raw,old=saved_state(out,spec);prep=vector(load(out/'states'/spec['cohort']/spec['id']/'prepared.pt'));inp=input_data(out,spec)
            allrecords.append(evaluate(rt,head,out,spec,prep,inp,old['preparation_valid'],cohort,raw))
            if i%4==0:snapshot(out,backup)
    for i,spec in enumerate(specs(),1):
        prep,inp,valid=prepare_new(rt,out,spec)
        allrecords.append(evaluate(rt,head,out,spec,prep,inp,valid,'new_local'))
        if i%4==0:snapshot(out,backup)
    rt.invariant();check_frozen(out);require(all(t._version==head_versions[n] for n,t in head.named_buffers()),'Frozen composed-writer buffer changed')
    co={g:counts([r for r in allrecords if r['cohort']==g]) for g in ('training','inspected_development','new_local')}
    require([co[g]['raw']['n_requested'] for g in co]==[48,16,16],'Fixed denominators changed')
    tr=co['training']['selected6']['complete_passed']==48;fresh=co['new_local']['selected6']['complete_passed']==16
    decision='C35_FROZEN_SINGLE_PROPOSAL_LOCAL_DEVELOPMENT_PASS' if tr and fresh else ('C35_RAW_TRAINING_FUNCTION_FAILURE' if not tr else 'C35_NEW_LOCAL_ACCEPTANCE_FAILURE')
    s={'protocol':PROTOCOL,'complete':True,'decision':decision,'cohorts':co,
       'new_local_by_case':{c:counts([r for r in allrecords if r['cohort']=='new_local' and r['spec']['case']==c]) for c in CASES},
       'selected_heads':{c:read(out/'fit'/c/'COMPLETE.json')['selected_spec'] for c in CASES},
       'calibration_training_n':36,'fixed_geometry_prior_n':12,'coefficients_frozen_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),
       'all80_requested_states_retained':len(allrecords)==80,'new_state_generation_retries':0,
       'old16_used_for_training_or_selection':False,'all_heads_frozen_before_new_state_generation':True,
       'candidate_points_sealed_before_all_endpoint_audits':True,
       'online_write_Jacobians':0,'online_write_SVD_calls':0,'online_write_correction_steps':0,'online_write_backtracking':0,
       'postwrite_response_used_for_proposal':False,'teacher_fallback_used':False,
       'offline_training_uses_archived_response_targets_and_Jacobians':True,'offline_new_state_preparation_uses_geometry':True,
       'one_frozen_core_forward_per_state_and_fixed_affine_head_operations':True,
       'preparation_Jacobian_files':len(list((out/'preparation_geometry').glob('*.pt'))),
       'preparation_correction_Jacobian_files':len(list((out/'new_states').rglob('J_*.pt'))),
       'original_C25_C27_writer_unchanged':True,'fresh_state_seeds':True,'fresh_evaluation':False,
       'same_model_chart_access_prompts':True,'requested_output_bit_supplied':True,
       'LC1_local_development_protocol_passed':bool(tr and fresh),'LC1_independently_confirmed':False,
       'full_Line_C_complete':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
       'scope':SETTINGS['scope'],
       'interpretation':'All three frozen proposals are produced from prewrite data before any written-endpoint audit. Fixed offline Jacobians are features, not rebuilt online. This is externally committed same-model/chart development; native updates, independent model replication and broad generalisation are not established.'}
    dump(out/'SUMMARY.json',s);dump(out/'CONSOLE_SUMMARY.json',s);snapshot(out,backup);print(json.dumps(s,indent=2),flush=True);return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--drive-dir',default='');a=p.parse_args()
    lock=open(Path(a.out).parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(a.out,a.drive_dir)
