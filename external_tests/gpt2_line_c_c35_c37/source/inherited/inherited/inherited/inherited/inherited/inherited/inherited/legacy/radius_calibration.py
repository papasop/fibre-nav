"""Persist compact ridge coefficients; reconstruct directions from fixed geometry."""
import numpy as np

def ridge_coefficients(system,ridge):
 if not np.isfinite(ridge) or ridge<=0:raise ValueError('Positive ridge required')
 v=system['vectors'];b=-system['E'].reshape(-1)/system['den']
 coefficients=(v@((v.T@b)/(system['values']+ridge)))/system['den']
 if not np.isfinite(coefficients).all():raise ValueError('Nonfinite compact ridge coefficients')
 return coefficients

def reconstruct_direction(system,coefficients):
 c=np.asarray(coefficients,dtype=np.float64)
 if c.shape!=system['den'].shape or not np.isfinite(c).all():raise ValueError('Invalid compact coefficient shape/values')
 D=system['scale']*(system['j'].T@(c[:,None]*system['h']))
 if not np.isfinite(D).all():raise ValueError('Nonfinite reconstructed direction')
 return D

def scale_direction(system,base,ridge,layer_radius,increment_radius):
 if any(not np.isfinite(x) or x<=0 for x in (ridge,layer_radius,increment_radius)):raise ValueError('Positive ridge/radii required')
 D=np.asarray(base,dtype=np.float64).copy()
 if D.shape!=(system['J'].shape[2],system['H'].shape[1]) or not np.isfinite(D).all():raise ValueError('Invalid unscaled output-layer direction')
 layer_norm=float(np.linalg.norm(D));increments=system['scale']*(D@system['H'].T).T
 maximum=float(np.max(np.linalg.norm(increments,axis=1)))
 unscaled=system['E']+np.einsum('npd,nd->np',system['J'],increments)
 factor=min(1.,layer_radius/max(layer_norm,1e-300),increment_radius/max(maximum,1e-300));D*=factor
 predicted=system['E']+np.einsum('npd,nd->np',system['J'],system['scale']*(D@system['H'].T).T)
 return D,{**system['info'],'ridge':ridge,'increment_radius':increment_radius,'layer_radius':layer_radius,
  'unclipped_layer_norm':layer_norm,'unclipped_max_training_increment_norm':maximum,'trust_clip_factor':factor,
  'layer_direction_norm':float(np.linalg.norm(D)),'predicted_full_step_residuals':predicted.tolist(),
  'predicted_maximum_drift':float(np.max(np.abs(predicted))),'unclipped_linear_predicted_R':float(np.max(np.abs(unscaled))),
  'linear_solve_uses_float64':True,'autograd_Jacobian_precision':'source float32; cast to float64 for solve'}
