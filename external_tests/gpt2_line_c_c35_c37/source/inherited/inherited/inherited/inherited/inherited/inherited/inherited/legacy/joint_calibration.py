"""Float64 Kronecker least-squares output-layer calibration; no rank certificate."""
import numpy as np

def checked(J,H,E,scale):
 J=np.asarray(J,dtype=np.float64);H=np.asarray(H,dtype=np.float64);E=np.asarray(E,dtype=np.float64)
 if J.ndim!=3 or H.ndim!=2 or E.shape!=J.shape[:2] or H.shape[0]!=J.shape[0]:raise ValueError('J/H/E shape mismatch')
 if not all(x>0 for x in J.shape) or not H.shape[1]>0:raise ValueError('Nonempty calibration inputs required')
 if not np.isfinite(scale) or scale<=0 or any(not np.isfinite(x).all() for x in (J,H,E)):raise ValueError('Nonfinite/invalid calibration input')
 return J,H,E

def prepare_system(J,H,E,scale,normalisation_floor=1e-12,rank_cutoff=1e-8):
 if not np.isfinite(normalisation_floor) or normalisation_floor<=0 or not 0<rank_cutoff<=1:raise ValueError('Invalid normalisation/rank convention')
 J,H,E=checked(J,H,E,scale);n,p,d=J.shape;j=J.reshape(n*p,d);h=np.repeat(H,p,axis=0)
 row_norm=scale*np.linalg.norm(j,axis=1)*np.linalg.norm(h,axis=1);den=np.maximum(row_norm,normalisation_floor)
 gram=(scale**2)*(j@j.T)*(h@h.T)/(den[:,None]*den[None,:]);gram=(gram+gram.T)/2
 if not np.isfinite(gram).all():raise ValueError('Nonfinite Gram')
 values,vectors=np.linalg.eigh(gram);positive=np.maximum(values,0.)
 info={'Gram_eigenvalues':values.tolist(),'negative_eigenvalues':int((values<0).sum()),'zero_derivative_rows':int((row_norm<=normalisation_floor).sum()),
  'numerical_Gram_rank':int((positive>rank_cutoff*max(float(positive.max()),1.)).sum()),'rank_certified':False,
  'n_states':n,'n_response_equations':n*p,'output_layer_variables':d*H.shape[1],
  'finite_response_preservation_certified':False,'normalised_Gram':True}
 return {'J':J,'H':H,'E':E,'scale':scale,'j':j,'h':h,'den':den,'values':positive,'vectors':vectors,'info':info}

def solve_direction(system,ridge,layer_radius,increment_radius):
 if any(not np.isfinite(x) or x<=0 for x in (ridge,layer_radius,increment_radius)):raise ValueError('Positive ridge/radii required')
 b=-system['E'].reshape(-1)/system['den'];v=system['vectors'];coeff=v@((v.T@b)/(system['values']+ridge))
 coeff=coeff/system['den'];D=system['scale']*(system['j'].T@(coeff[:,None]*system['h']))
 if not np.isfinite(D).all():raise ValueError('Nonfinite layer direction')
 before_norm=float(np.linalg.norm(D));increments=system['scale']*(D@system['H'].T).T
 maximum=float(np.max(np.linalg.norm(increments,axis=1)))
 factor=min(1.,layer_radius/max(before_norm,1e-300),increment_radius/max(maximum,1e-300));D*=factor
 predicted=system['E']+np.einsum('npd,nd->np',system['J'],system['scale']*(D@system['H'].T).T)
 return D,{**system['info'],'ridge':ridge,'unclipped_layer_norm':before_norm,'unclipped_max_training_increment_norm':maximum,
  'trust_clip_factor':factor,'layer_direction_norm':float(np.linalg.norm(D)),
  'predicted_full_step_residuals':predicted.tolist(),'predicted_maximum_drift':float(np.max(np.abs(predicted))),
  'linear_solve_uses_float64':True,'autograd_Jacobian_precision':'source float32; cast to float64 for solve'}

def response_prediction(J,E,actual_displacements):
 J=np.asarray(J,dtype=np.float64);E=np.asarray(E,dtype=np.float64);D=np.asarray(actual_displacements,dtype=np.float64)
 if J.ndim!=3 or E.shape!=J.shape[:2] or D.shape!=(J.shape[0],J.shape[2]) or any(not np.isfinite(x).all() for x in (J,E,D)):raise ValueError('Prediction shape/nonfinite')
 return E+np.einsum('npd,nd->np',J,D)

def linear_protection(predicted,passing_indices,budget):
 predicted=np.asarray(predicted,dtype=np.float64)
 return bool(np.isfinite(predicted).all() and all(np.max(np.abs(predicted[i]))<=budget for i in passing_indices))
