"""C30's fixed predictor family and train-only CV, with new state coverage."""
import os, hashlib
import numpy as np
from c31_protocol import *


def fit_arm(out,name,X,Y,rows):
    out=Path(out);folder=out/'predictors'/name
    binding=hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest()
    ids=[r['id'] for r in rows]
    for case in CASES:
        mask=np.array([r['case']==case for r in rows]);ci=[s for s,f in zip(ids,mask) if f]
        dest=folder/case;cp=dest/'COMPLETE.json'
        if cp.exists():
            s=read(cp);require(s['training_ids']==ci and s['dataset_sha256']==binding,'Cached predictor dataset differs')
            for f in predictor.FAMILIES:load_model(dest/(f+'.npz'),s['models'][f])
            continue
        selection=predictor.tune(X[mask],Y[mask],ci)
        models={}
        for f in predictor.FAMILIES:
            spec=selection['by_family'][f];model=predictor.fit(X[mask],Y[mask],f,spec['ridge'])
            p=dest/(f+'.npz');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.npz.tmp')
            with tmp.open('wb') as fh:np.savez_compressed(fh,**{k:model[k] for k in ('ix','mean','scale','active','beta','target_mean')})
            os.replace(tmp,p)
            models[f]={'family':f,'ridge':spec['ridge'],'fit_n':len(ci),'training_ids':ci,'file_sha256':sha(p)}
        dump(cp,{'training_ids':ci,'dataset_sha256':binding,'selection':selection,'models':models})
        print('C31 FROZEN PREDICTOR ROUTE',name,case,len(ci),selection['winner'],flush=True)
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
    freeze={'arm':name,'training_ids':ids,'dataset_sha256':binding,'files':files,'fitted_only_on_training':True}
    fp=out/(name.upper()+'_PREDICTOR_FROZEN.json')
    if fp.exists():require(read(fp)==freeze,'Cached arm freeze differs')
    else:dump(fp,freeze)
    return freeze


def load_model(path,meta):
    require(sha(path)==meta['file_sha256'],'Actual predictor coefficients differ')
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'ix','mean','scale','active','beta','target_mean'},'Predictor fields differ')
        model={k:z[k].copy() for k in z.files}
    d=len(predictor.columns(meta['family']))
    require(model['ix'].tolist()==predictor.columns(meta['family']) and model['beta'].shape==(d,6) and model['target_mean'].shape==(6,),'Predictor shape differs')
    require(all(model[k].shape==(d,) for k in ('mean','scale','active')) and model['active'].dtype==np.bool_ and (model['scale']>0).all(),'Predictor scales differ')
    require(all(np.isfinite(v).all() for v in model.values()),'Nonfinite predictor')
    return {**model,'family':meta['family'],'ridge':meta['ridge'],'fit_n':meta['fit_n']}


def freeze_all(out):
    out=Path(out)
    fs={name:read(out/(name.upper()+'_PREDICTOR_FROZEN.json')) for name in ('historical76','expanded124')}
    fr={'arms':fs,'prospective_states_generated_at_freeze':0,'all_frozen_before_prospective_generation':True,
        'writer_frozen_sha256':sha(out/'WRITER_FROZEN.json')}
    p=out/'PREDICTORS_FROZEN.json'
    if p.exists():require(read(p)==fr,'Combined freeze changed')
    else:
        require(not any((out/'states/prospective').rglob('*.pt')),'Cannot first freeze predictors after prospective state generation')
        dump(p,fr)
    check_frozen(out)
    return fr


def check_frozen(out):
    out=Path(out);fr=read(out/'PREDICTORS_FROZEN.json')
    require(fr['all_frozen_before_prospective_generation'] and fr['prospective_states_generated_at_freeze']==0,'Prospective freeze order changed')
    require(fr['writer_frozen_sha256']==sha(out/'WRITER_FROZEN.json'),'Writer binding differs')
    for name,s in fr['arms'].items():
        actual={p.relative_to(out).as_posix() for p in (out/'predictors'/name).rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
        require(actual==set(s['files']),'Predictor file coverage differs')
        for n,h in s['files'].items():require(sha(out/n)==h,'Frozen predictor modified: '+n)
    return fr


def forecast(out,name,X,rows):
    ans={f:np.empty((len(rows),6)) for f in (*predictor.FAMILIES,'train_selected')}
    for case in CASES:
        mask=np.array([r['case']==case for r in rows]);folder=Path(out)/'predictors'/name/case;cp=read(folder/'COMPLETE.json')
        for f in predictor.FAMILIES:ans[f][mask]=predictor.predict(load_model(folder/(f+'.npz'),cp['models'][f]),X[mask])
        ans['train_selected'][mask]=ans[cp['selection']['winner']['family']][mask]
    return ans


def compare(out,X,Y,rows):
    check_frozen(out);ans={};ids=[r['id'] for r in rows];valid=[r['preparation_valid'] for r in rows]
    for name in ('historical76','expanded124'):
        pp=forecast(out,name,X,rows);reports={f:predictor.report(Y,p,ids,valid) for f,p in pp.items()}
        for f,r in reports.items():
            r['all_requested_prediction_errors_within_budget']=sum(v['error_Linf']<=SETTINGS['response_budget'] for v in r['rows'])
            r['reference_valid_means_preparation_only_no_teacher']=True
        dump(Path(out)/(name.upper()+'_PROSPECTIVE_PREDICTIONS.json'),reports)
        ans[name]={f:{k:v for k,v in r.items() if k!='rows'} for f,r in reports.items()}
    return ans
