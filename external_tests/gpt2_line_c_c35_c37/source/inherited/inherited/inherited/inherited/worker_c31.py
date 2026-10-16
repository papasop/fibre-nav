"""Frozen C27 writes label new training coverage; final predictors never write."""
import argparse, fcntl, hashlib, os
from pathlib import Path
import numpy as np
import torch
from c31_protocol import *
from c31_prediction import fit_arm, freeze_all, check_frozen, compare
from c30_data import load_cohort
from c31_runtime import Runtime, load, vector, save_tensor, tensor_hash
from geometry import null_project


def validate_cache(out,record,spec):
    require(record['spec']==spec and record['writer_frozen_sha256']==sha(Path(out)/'WRITER_FROZEN.json'),'Saved state spec/writer differs')
    folder=Path(out)/'states'/spec['cohort']/spec['id']
    for n,h in record['files'].items():require(sha(folder/n)==h,'Saved actual state/input/endpoint changed: '+n)
    require(record['generation_retries']==0 and record['post_write_correction_steps']==0,'State protocol changed')
    return record


def historical_replay(rt,out):
    out=Path(out);p=out/'HISTORICAL_INPUT_REPLAY.json'
    if p.exists():
        r=read(p);require(r['writer_frozen_sha256']==sha(out/'WRITER_FROZEN.json') and r['all12_passed'],'Historical replay binding differs')
        return r
    rows=[]
    for case in CASES:
        for i in range(3):
            sid=case+'__v%d'%i;folder=out/'data/training'/sid
            prep=vector(load(folder/'prepared.pt'));old=load(folder/'INPUT.pt');desired=STARTS[case][0]^STARTS[case][1]
            fresh=rt.fresh_input(prep,desired);delta=float((fresh['input']-old['input']).abs().max())
            require(delta<=SETTINGS['historical_input_replay_tolerance'] and int(fresh['route'])==CASES.index(case),'Original fresh input/sketch recipe replay failed: '+sid)
            # Reproduce the archived endpoint with its authoritative unchanged input;
            # fresh extraction is a separate tolerance check, not a silent replacement.
            theta,timing=rt.raw_write(prep,old);m=rt.evaluate(theta,STARTS[case][:3]+[desired],STARTS[case])
            oldm=read(folder/'C29_COMPLETE.json')['endpoints']['c27']['metrics']
            err=max(abs(a-b) for a,b in zip(m['response_residual'],oldm['response_residual']))
            require(err<=SETTINGS['historical_endpoint_replay_tolerance'] and m['bits']==oldm['bits'] and m['gates']==oldm['gates'],'Original C27 endpoint replay differs: '+sid)
            rows.append({'id':sid,'input_max_error':delta,'written_residual_replay_error':err,'passed':True})
            print('C31 HISTORICAL INPUT + WRITE REPLAY',sid,delta,err,flush=True)
    r={'all12_passed':True,'n':12,'writer_frozen_sha256':sha(out/'WRITER_FROZEN.json'),'rows':rows,
       'sketch_recreated_from_pinned_original_source':True,'original_archived_sketch_tensor_independently_available':False}
    dump(p,r);return r


def collect_state(rt,out,spec):
    out=Path(out);folder=out/'states'/spec['cohort']/spec['id'];folder.mkdir(parents=True,exist_ok=True)
    if spec['cohort']=='prospective':check_frozen(out)
    cp=folder/'COMPLETE.json'
    if cp.exists():return validate_cache(out,read(cp),spec)
    base=vector(load(out/'data/training'/spec['base_id']/'prepared.pt'))
    qfile=folder/'PERTURBATION.pt'
    if qfile.exists():
        qcp=load(qfile);require(qcp['spec']==spec and qcp['base_sha256']==tensor_hash(base),'Saved perturbation identity differs')
        q=vector(qcp['theta']);geom=qcp['geometry']
    else:
        J=rt.jacobian(base,out/'base_preparation_geometry'/(spec['base_id']+'.pt'))
        raw=np.random.default_rng(spec['seed']).normal(size=36864)
        projected,geom=null_project(J,raw,SETTINGS['null_relative_cutoff'],SETTINGS['null_absolute_cutoff'])
        norm=float(np.linalg.norm(projected));require(norm>0,'Zero fixed perturbation: no replacement allowed')
        q=torch.from_numpy((projected*(spec['norm']/norm)).astype(np.float32))
        save_tensor(qfile,{'theta':q,'spec':spec,'base_sha256':tensor_hash(base),'geometry':geom})
    prepfile=folder/'prepared.pt'
    if prepfile.exists():
        saved=load(prepfile)
        require(saved['initial_sha256']==tensor_hash(base+q) and saved['spec']==spec,'Saved preparation initial point/spec changed')
        prep=vector(saved)
    else:
        prep,history,seconds=rt.correct_preparation(base+q,folder/'preparation_correction')
        save_tensor(prepfile,{'theta':prep,'initial_sha256':tensor_hash(base+q),'spec':spec});dump(folder/'PREPARATION_CORRECTION.json',{'history':history,'seconds':seconds,'only_pre_write_preparation':True})
    pm=rt.evaluate(prep,STARTS[spec['case']]);desired=STARTS[spec['case']][0]^STARTS[spec['case']][1]
    inpfile=folder/'INPUT.pt'
    if inpfile.exists():
        inp=load(inpfile)
        require(inp['prepared_sha256']==tensor_hash(prep) and inp['desired_bit']==desired and inp['spec']==spec,'Saved feature preparation/target/spec differs')
    else:
        inp={**rt.fresh_input(prep,desired),'prepared_sha256':tensor_hash(prep),'desired_bit':desired,'spec':spec};save_tensor(inpfile,inp)
    require(inp['input'].shape==(3147,) and inp['input'].dtype==torch.float32 and bool(torch.isfinite(inp['input']).all()),'Nonfinite generated input; abort incomplete experiment, never replace state')
    route_match=int(inp['route'])==CASES.index(spec['case']);preparation_valid=bool(pm['pass'] and route_match)
    endpointfile=folder/'RAW_WRITTEN.pt'
    if endpointfile.exists():
        endpoint=load(endpointfile);require(endpoint['prepared_sha256']==tensor_hash(prep) and endpoint['input_sha256']==tensor_hash(inp['input']),'Raw endpoint preparation/input differs')
        theta=vector(endpoint);timing=endpoint['timing']
    else:
        theta,timing=rt.raw_write(prep,inp)
        save_tensor(endpointfile,{'theta':theta,'prepared_sha256':tensor_hash(prep),'input_sha256':tensor_hash(inp['input']),'timing':timing})
    require(timing['proposal_count']==1 and timing['online_Jacobians']==0 and timing['online_SVD_calls']==0 and timing['post_write_correction_steps']==0,'Raw one-proposal protocol changed')
    target=STARTS[spec['case']][:3]+[desired];wm=rt.evaluate(theta,target,STARTS[spec['case']]);rt.invariant()
    require(len(wm['response_residual'])==6 and np.isfinite(wm['response_residual']).all(),'Nonfinite raw residual: do not drop state')
    record={'spec':spec,'writer_frozen_sha256':sha(out/'WRITER_FROZEN.json'),
       'preparation_valid':preparation_valid,'preparation':pm,'route_match':route_match,'raw_written':wm,
       'teacher_reference_constructed':False,'reference_validity_scope':'preparation only; no positive teacher-write feasibility check',
       'generation_retries':0,'replacement_used':False,'post_write_correction_steps':0,'timing':timing,
       'files':{n:sha(folder/n) for n in ('PERTURBATION.pt','prepared.pt','INPUT.pt','RAW_WRITTEN.pt')}}
    dump(cp,record)
    print('C31 RAW STATE',spec['cohort'],spec['id'],'prep',preparation_valid,'raw_R',wm['drift'],'writer_pass',bool(preparation_valid and wm['pass']),flush=True)
    return record


def arrays(out,records):
    X=[];Y=[];rows=[]
    for rec in records:
        t=rec['spec'];validate_cache(out,rec,t)
        inp=load(Path(out)/'states'/t['cohort']/t['id']/'INPUT.pt')
        X.append(inp['input'].double().numpy());Y.append(rec['raw_written']['response_residual'])
        rows.append({'id':t['id'],'case':t['case'],'preparation_valid':rec['preparation_valid']})
    return np.stack(X),np.asarray(Y,dtype=np.float64),rows


def coverage(records):
    result={}
    for case in (*CASES,'all'):
        rs=[r for r in records if case=='all' or r['spec']['case']==case]
        e=np.asarray([r['raw_written']['response_residual'] for r in rs]);m=e.mean(1)
        result[case]={'n_requested':len(rs),'valid_preparations':sum(r['preparation_valid'] for r in rs),
           'raw_complete_write_passed':sum(r['preparation_valid'] and r['raw_written']['pass'] for r in rs),
           'raw_response_gate_passed':sum(r['raw_written']['gates']['response'] for r in rs),
           'maximum_raw_response_drift':float(np.max(np.abs(e))),
           'common_residual_minimum':float(m.min()),'common_residual_maximum':float(m.max()),
           'nonresponse_gates_passed':{g:sum(r['raw_written']['gates'][g] for r in rs) for g in GATES if g!='response'},
           'all_failures_retained':True,'positive_teacher_reference_available':False}
    return result


def main(out,backup='',runtime_factory=Runtime):
    out=Path(out);root=Path(__file__).resolve().parent;check_sources(root)
    identity=read(out/'identity.json');require(identity['settings']==SETTINGS and identity['source_hashes']==sources(root),'Runtime source/settings mismatch')
    expected_specs={'new_training':specs('new_training'),'prospective':specs('prospective'),'old_inspected16_used':False,'resampling_allowed':False}
    if (out/'FIXED_STATE_SPECS.json').exists():require(read(out/'FIXED_STATE_SPECS.json')==expected_specs,'Fixed seeds/specs changed')
    else:dump(out/'FIXED_STATE_SPECS.json',expected_specs)
    for n,r in read(out/'PARENT_PROVENANCE.json')['members'].items():require(sha(out/n)==r['sha256'],'Imported parent member differs: '+n)
    if (out/'SUMMARY.json').exists():
        check_frozen(out);print((out/'SUMMARY.json').read_text(),flush=True);return read(out/'SUMMARY.json')
    torch.set_num_threads(1)
    X,Y,oldrows=load_cohort(out,'training',torch)
    historical=[{'id':r['id'],'case':r['case'],'preparation_valid':r['valid']} for r in oldrows]
    rt=runtime_factory(out,root);historical_replay(rt,out)
    # Baseline is fixed before even the new *training* states are generated.
    baseline=fit_arm(out,'historical76',X,Y,historical);snapshot(out,backup)
    new=[]
    for i,t in enumerate(specs('new_training'),1):
        new.append(collect_state(rt,out,t))
        if i%SETTINGS['archive_every_states']==0:snapshot(out,backup)
    NX,NY,nrows=arrays(out,new)
    expanded=fit_arm(out,'expanded124',np.concatenate([X,NX]),np.concatenate([Y,NY]),historical+nrows)
    freeze_all(out);snapshot(out,backup)
    prospective=[]
    for i,t in enumerate(specs('prospective'),1):
        prospective.append(collect_state(rt,out,t))
        if i%SETTINGS['archive_every_states']==0:snapshot(out,backup)
    DX,DY,drows=arrays(out,prospective);predictions=compare(out,DX,DY,drows)
    rt.invariant();check_frozen(out)
    oldcommon=Y.mean(1)
    summary={'protocol':PROTOCOL,'complete':True,'decision':'C31_RESIDUAL_COVERAGE_COMPARISON_COMPLETE',
      'historical_training':{'n_requested':76,'raw_write_passed_from_C29':sum(r['writer_endpoint_pass'] for r in oldrows),
          'maximum_raw_response_drift':float(np.max(np.abs(Y))),'common_residual_minimum':float(oldcommon.min()),'common_residual_maximum':float(oldcommon.max()),'not_newly_replayed_all76':True},
      'new_training':coverage(new),'prospective_local_development':coverage(prospective),
      'prospective_prediction_comparison':predictions,
      'frozen_predictor_sha256':sha(out/'PREDICTORS_FROZEN.json'),'C27_writer_unchanged':True,
      'new_writer_training_steps':0,'post_write_correction_steps':0,'teacher_fallback_used':False,
      'offline_geometry_saved':{'base_preparation_Jacobians':len(list((out/'base_preparation_geometry').glob('*.pt'))),
          'correction_Jacobians':len(list((out/'states').rglob('J_*.pt'))),
          'fixed_perturbation_records':len(list((out/'states').rglob('PERTURBATION.pt')))},
      'all48_training_and16_prospective_states_retained':True,'generation_retries':0,
      'old_inspected16_used_for_fitting_or_selection':False,'all_predictors_frozen_before_prospective_generation':True,
      'offline_preparation_uses_autograd_SVD_response_correction':True,
      'requested_output_bit_supplied':True,'fresh_state_seeds':True,'fresh_evaluation':False,
      'independent_model_or_chart_seed':False,'internal_parameter_update':False,'controller_dispatch_tested':False,'LC1_closed':False,
      'scope':'Same frozen C27 writer, new state coverage for residual prediction. Preparation validity is checked; no teacher-write admissibility certificate is constructed. Both residual predictors are frozen before generation of16 fixed-seed prospective states. No predicted residual is converted into a parameter correction; raw write acceptance and forecast quality are separate.'}
    dump(out/'SUMMARY.json',summary);snapshot(out,backup);print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--drive-dir',default='');a=p.parse_args()
    lock=open(Path(a.out).parent.parent/'worker.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    main(a.out,a.drive_dir)
