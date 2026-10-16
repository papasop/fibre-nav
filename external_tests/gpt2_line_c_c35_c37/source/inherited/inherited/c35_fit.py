"""No old16 targets are parsed before these coefficient heads are frozen."""
import numpy as np
import torch
import platform
from c35_protocol import *
from c35_math import geometric,select,predict,compile_affine
from c31_runtime import load,save_tensor,vector,tensor_hash
from worker_c34 import saved_state,check_metrics

def prior_geometry(out,case):
    out=Path(out);ids=[s for s in SETTINGS['prior_basis_states'] if s.startswith(case+'__')]
    points=[];ys=[];js=[]
    from c31_protocol import specs as old_specs
    byid={s['id']:s for s in old_specs('new_training')}
    for sid in ids:
        theta,_=saved_state(out,byid[sid]);m=read(out/'evaluation'/sid/'COMPLETE.json')['arms']['raw']['metrics'];check_metrics(m,case)
        require(m['theta_sha256']==tensor_hash(theta),'Prior geometry/raw metric identity differs')
        cp=load(out/'basis'/case/(sid+'_J.pt'));J=cp['J']
        require(cp['theta_sha256']==tensor_hash(theta) and cp['geometry_stage']=='C34_saved_raw_written_endpoint','Prior Jacobian binding differs')
        require(J.dtype==torch.float64 and J.shape==(6,36864) and bool(torch.isfinite(J).all()),'Invalid prior Jacobian')
        points.append(theta.double().numpy());ys.append(m['response_residual']);js.append(J.numpy())
    basis=load(out/'basis'/case/'BASES.pt')['response6']
    require(basis.dtype==torch.float64 and basis.shape==(36864,6) and bool(torch.isfinite(basis).all()),'Invalid pinned6D basis')
    return {'J_mean':np.mean(js,axis=0),'point_center':np.mean(points,axis=0),
            'response_center':np.mean(ys,axis=0),'B6':basis.numpy().copy()}

def input_data(out,spec):
    folder=Path(out)/'states'/spec['cohort']/spec['id'];f=load(folder/'INPUT.pt')
    require(f['spec']==spec and f['input'].dtype==torch.float32 and f['input'].shape==(3147,), 'Original cached input/spec differs')
    require(f['extras'].dtype==torch.float32 and f['extras'].shape==(70,) and bool(torch.isfinite(f['extras']).all()),'Original C31 extras required, not reconstructed')
    require(bool(torch.isfinite(f['input']).all()) and f['route'].numel()==1,'Invalid cached input/route')
    return f

def training_arrays(out,case,map_):
    from c31_protocol import specs as old_specs
    ss=[s for s in old_specs('new_training') if s['case']==case and s['id'] not in SETTINGS['prior_basis_states']]
    gs=[];pre=[];ys=[]
    for s in ss:
        point,_=saved_state(out,s);inp=input_data(out,s)
        require(int(inp['route'])==CASES.index(case),'Training route differs')
        m=read(Path(out)/'evaluation'/s['id']/'COMPLETE.json')['arms']['raw']['metrics'];check_metrics(m,case)
        require(m['theta_sha256']==tensor_hash(point),'Calibration target endpoint differs')
        gs.append(geometric(map_,point.double().numpy()));pre.append(inp['extras'][:6].double().numpy()*.002);ys.append(m['response_residual'])
    return np.asarray(gs),np.asarray(pre),np.asarray(ys),[s['id'] for s in ss]

def check_frozen(out):
    out=Path(out);fr=read(out/'COEFFICIENTS_FROZEN.json')
    require(fr['settings']==SETTINGS and fr['new_state_generation_started'] is False and fr['old16_targets_used_for_fitting'] is False,'Coefficient freeze protocol differs')
    require(fr['parent_provenance_sha256']==sha(out/'PARENT_PROVENANCE.json') and fr['writer_frozen_sha256']==sha(out/'WRITER_FROZEN.json'),'Coefficient parent/writer binding differs')
    require(fr['fit_runtime_sha256']==sha(out/'FIT_RUNTIME.json'),'Fitting runtime record changed')
    files={p.relative_to(out).as_posix() for p in (out/'fit').rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    require(files==set(fr['files']),'Frozen coefficient file coverage differs')
    for n,h in fr['files'].items():require(sha(out/n)==h,'Frozen coefficient file changed: '+n)
    for n,h in read(out/'parent/C34_BASES_FROZEN.json')['files'].items():require(sha(out/n)==h,'Parent basis/J changed after coefficient freeze')
    return fr

def fit_all(out,backup=''):
    out=Path(out)
    if (out/'COEFFICIENTS_FROZEN.json').exists():check_frozen(out);return
    runtime={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),'fit_algebra':'float64'}
    if (out/'FIT_RUNTIME.json').exists():require(read(out/'FIT_RUNTIME.json')==runtime,'Partial coefficient fitting runtime differs')
    else:dump(out/'FIT_RUNTIME.json',runtime)
    for case in CASES:
        folder=out/'fit'/case;folder.mkdir(parents=True,exist_ok=True);cp=folder/'COMPLETE.json'
        if cp.exists():
            r=read(cp);require(sha(folder/'HEAD.pt')==r['head_file_sha256'],'Cached coefficient head changed');continue
        map_=prior_geometry(out,case);g,p,y,ids=training_arrays(out,case,map_);model,audit=select(g,p,y,ids)
        record={**map_,**{k:model[k] for k in ('mean','scale','active','offset','beta')}}
        record.update(compile_affine(record))
        save_tensor(folder/'HEAD.pt',{k:torch.from_numpy(v.copy()) for k,v in record.items()})
        prediction=predict(model,g,p)
        dump(cp,{'case':case,'calibration_ids':ids,'fixed_prior_ids':[x for x in SETTINGS['prior_basis_states'] if x.startswith(case+'__')],
             'head_file_sha256':sha(folder/'HEAD.pt'),'selected_spec':model['spec'],'CV':audit,
             'calibration_training_prediction_RMSE':float(np.sqrt(np.mean((prediction-y)**2))),
             'calibration_training_prediction_error_Linf':float(np.max(np.abs(prediction-y))),
             'target_sha256':hashlib.sha256(g.tobytes()+p.tobytes()+y.tobytes()+json.dumps(ids).encode()).hexdigest()})
        print('C35 TRAIN-ONLY COEFFICIENT HEAD',case,model['spec'],'fit_prediction_RMSE',float(np.sqrt(np.mean((prediction-y)**2))),flush=True)
        snapshot(out,backup)
    files={p.relative_to(out).as_posix():sha(p) for p in (out/'fit').rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    dump(out/'COEFFICIENTS_FROZEN.json',{'settings':SETTINGS,'files':files,'prior_reference_n':12,'calibration_n':36,
         'parent_provenance_sha256':sha(out/'PARENT_PROVENANCE.json'),'writer_frozen_sha256':sha(out/'WRITER_FROZEN.json'),
         'fit_runtime_sha256':sha(out/'FIT_RUNTIME.json'),
         'old16_targets_used_for_fitting':False,'new_state_generation_started':False,
         'fixed_basis_prior_conditioned_calibration_CV':True,'independent_base_preparation_holdout':False})
    check_frozen(out);snapshot(out,backup)

def records(out):
    check_frozen(out);ans=[]
    shapes={'J_mean':(6,36864),'point_center':(36864,),'response_center':(6,),'B6':(36864,6),
            'mean':(12,),'scale':(12,),'active':(12,),'offset':(6,),'beta':(12,6),
            'geometry_W':(6,36864),'geometry_bias':(6,),'coefficient_W':(6,36864),'pre_response_W':(6,6),'coefficient_bias':(6,)}
    for case in CASES:
        r=load(Path(out)/'fit'/case/'HEAD.pt');require(set(r)==set(shapes),'Actual coefficient head tensor coverage differs')
        for k,shape in shapes.items():
            require(r[k].shape==shape and r[k].dtype==(torch.bool if k=='active' else torch.float64),'Actual coefficient head shape/dtype differs: '+k)
            require(bool(torch.isfinite(r[k]).all()),'Nonfinite coefficient head')
        require(bool((r['scale']>0).all()),'Invalid frozen feature scale')
        ans.append({k:v.numpy().copy() for k,v in r.items()})
    return ans
