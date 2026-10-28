"""Explicit precommit FP64 coefficient map; no model or solver."""
from core159 import *
def torch_constants(c):
 import torch
 return {n:torch.from_numpy(a.copy()) for n,a in c.items()}
def torch_map(c,sid,alpha):
 import torch
 need(alpha.dtype==torch.float64 and alpha.shape==() and bool(torch.isfinite(alpha)),'Finite FP64 scalar alpha');p=c['P0']+alpha*c['deltaP'];w=c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);point=c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*torch.roll(u,1));need(point.dtype==e.dtype==p.dtype==torch.float64,'Continuous FP64 map only');return point,p,e

def chain_arithmetic(ga,ge,gp,c,sid):
 ge=farray(np.asarray(ge,float),(2,c['V'].shape[0]));gp=farray(np.asarray(gp,float),c['P0'].shape);need(np.isfinite(ga),'Finite alpha derivative');pull=c[sid+'__weights'][:,None,None]*ge[None,...];explicit=float(np.sum(pull*c['deltaP']));direct=float(np.sum(gp*c['deltaP']));return dict(alpha_AD=float(ga),effective_gradient=ge.tolist(),direct_coefficient_gradient=gp.tolist(),explicit_pullback_gradient=pull.tolist(),pullback_vs_direct_Linf=float(max(abs(pull-gp).reshape(-1))),explicit_pullback_dot_delta=explicit,direct_coefficient_dot_delta=direct,explicit_dot_minus_alpha=explicit-float(ga),direct_dot_minus_alpha=direct-float(ga),derivative_agreement_threshold=None)
