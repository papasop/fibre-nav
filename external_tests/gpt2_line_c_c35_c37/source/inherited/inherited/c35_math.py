"""Twelve-feature affine calibration with genuinely held-out calibration targets."""
import numpy as np
from c35_protocol import *
def choices():return [{'kind':'geometry','ridge':None}]+[{'kind':'ridge','ridge':x} for x in SETTINGS['ridge_grid']]
def checked(g,pre,y=None):
    g=np.asarray(g,dtype=np.float64);pre=np.asarray(pre,dtype=np.float64)
    require(g.ndim==2 and g.shape==pre.shape and g.shape[1]==6 and np.isfinite(g).all() and np.isfinite(pre).all(),'Finite paired n-by6 geometric/prewrite features required')
    if y is not None:
        y=np.asarray(y,dtype=np.float64);require(y.shape==g.shape and np.isfinite(y).all(),'Finite target denominator differs');return g,pre,y
    return g,pre
def fit(g,pre,y,spec):
    g,pre,y=checked(g,pre,y);require(len(g)>=2 and spec in choices(),'Invalid training count/spec')
    x=np.concatenate([g,pre],1)/.002;mu=x.mean(0);sd=x.std(0);active=sd>SETTINGS['std_active_floor'];scale=np.maximum(sd,SETTINGS['std_scale_floor'])
    z=(x-mu)/scale;z[:,~active]=0.;d=y-g;mean=d.mean(0)
    if spec['kind']=='geometry':beta=np.zeros((12,6));mean=np.zeros(6)
    else:beta=np.linalg.solve(z.T@z+len(x)*spec['ridge']*np.eye(12),z.T@(d-mean))
    require(np.isfinite(beta).all(),'Nonfinite calibrated coefficients')
    return {'spec':spec,'mean':mu,'scale':scale,'active':active,'offset':mean,'beta':beta}
def predict(model,g,pre):
    g,pre=checked(g,pre);x=np.concatenate([g,pre],1)/.002;z=(x-model['mean'])/model['scale'];z[:,~model['active']]=0
    p=g+model['offset']+z@model['beta'];require(np.isfinite(p).all(),'Nonfinite coefficient prediction');return p
def select(g,pre,y,ids):
    g,pre,y=checked(g,pre,y);require(len(ids)==9 and len(set(ids))==9 and not set(ids)&set(SETTINGS['prior_basis_states']),'Exactly9 nonprior calibration IDs per route required')
    ff=np.array(folds(ids,3));records=[]
    for spec in choices():
        p=np.empty_like(y)
        for k in range(3):
            tr=ff!=k;va=ff==k;p[va]=predict(fit(g[tr],pre[tr],y[tr],spec),g[va],pre[va])
        records.append({'spec':spec,'CV_loss':float(np.mean(((p-y)/.002)**2)),'calibration_CV_predictions':p.tolist()})
    winner=min([r for r in records if r['spec']['kind']=='ridge'],key=lambda t:t['CV_loss'])
    return fit(g,pre,y,winner['spec']),{'ids':ids,'folds':dict(zip(ids,ff.tolist())),'candidates':records,'selected':winner['spec'],'best_including_geometry_control':min(records,key=lambda t:t['CV_loss'])['spec'],
         'prior_geometry_fixed_outside_calibration_CV':True,'validation_ids_not_in_prior_basis':True}
def geometric(map_,point):
    point=np.asarray(point,dtype=np.float64);require(point.shape==(36864,) and np.isfinite(point).all(),'Finite raw coordinate point required')
    return map_['response_center']+map_['J_mean']@(point-map_['point_center'])

def compile_affine(record):
    """Offline algebra: absorb geometric features/standardisation into fixed weights."""
    q=record['beta']/record['scale'][:,None];q=np.where(record['active'][:,None],q,0.)
    transform=np.eye(6)+q[:6].T/.002
    gb=record['response_center']-record['J_mean']@record['point_center']
    return {'geometry_W':record['J_mean'].copy(),'geometry_bias':gb,
            'coefficient_W':transform@record['J_mean'],'pre_response_W':q[6:].T/.002,
            'coefficient_bias':transform@gb+record['offset']-record['mean']@q}
