"""Training-only shared dual ridge and compiled geometric maps; no routes."""
import numpy as np
from lc2_protocol import SETTINGS,require
def finite(x,ndim=None):
 a=np.asarray(x,dtype=np.float64)
 require(np.isfinite(a).all() and (ndim is None or a.ndim==ndim),'Finite shape mismatch');return a
def inverse(J,ridge=0.):
 J=finite(J,2);u,s,vt=np.linalg.svd(J,full_matrices=False)
 cut=max(SETTINGS['svd_absolute_cutoff'],SETTINGS['svd_relative_cutoff']*(s[0] if len(s) else 0))
 f=np.where(s>cut,s/(s*s+max(ridge,1e-30)),0.)
 return (vt.T*f)@u.T,{'singular_values':s.tolist(),'numeric_rank':int((s>cut).sum()),'cutoff':cut,'rank_certified':False}
def limited(v,radius):
 v=finite(v,1);norm=float(np.linalg.norm(v));return v*min(1.,radius/max(norm,1e-30))
def fit_shared(X,Y,records):
 from lc2_protocol import validate_training
 ids=validate_training(records);X,Y=finite(X,2),finite(Y,2)
 require(X.shape[0]==Y.shape[0]==len(ids),'Shared fit denominator mismatch')
 mu=X.mean(0);scale=np.maximum(X.std(0),SETTINGS['feature_scale_floor']);z=(X-mu)/scale
 ym=Y.mean(0);K=z@z.T/X.shape[1]
 coefficients=np.linalg.solve(K+SETTINGS['writer_ridge']*np.eye(len(z)),Y-ym)
 return {'mean':mu,'scale':scale,'training_z':z,'coefficients':coefficients,'delta_mean':ym}
def predict(head,x):
 x=finite(x,1);z=(x-head['mean'])/head['scale']
 return limited(head['delta_mean']+(z@head['training_z'].T/len(x))@head['coefficients'],SETTINGS['proposal_radius'])
def rhs_for_write(observed,desired):
 o=finite(observed,1);require(o.shape==(22,) and desired in (0,1),'Expected6 responses +16 read margins')
 rhs=np.zeros(22);rhs[:6]=-o[:6]/SETTINGS['response_budget']
 for i in (9,13,17,21):rhs[i]=(2*desired-1)*max(abs(o[i]),SETTINGS['teacher_target_margin'])-o[i]
 return rhs
def compose(head,theta,features,observed):
 theta=finite(theta,1);features=finite(features,1);o=finite(observed,1)
 require(features.shape==(3147,) and features[3076] in (0.,1.),'Requested bit/feature schema differs')
 delta=predict(head,features)
 predicted=o[:6]+head['J_mean'][:6]@delta+head['response_offset']
 compensation=limited(-head['B_response']@predicted,SETTINGS['compensation_radius'])
 geo=limited(head['B_all']@rhs_for_write(o,int(features[3076])),SETTINGS['proposal_radius'])
 return {'learned_raw':theta+delta,'geometry_shared':theta+geo,'learned_composite':theta+delta+compensation}
